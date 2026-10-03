"""Deterministic input checks, batch execution, aggregation and cost reporting."""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.request import urlopen

import pandas as pd

from classifier import base_result

REQUIRED = ['return_id', 'sku_id', 'vendor_id', 'category', 'size_ordered',
            'return_date', 'dropdown_reason', 'other_text']
GROUPS = ['vendor_id', 'category', 'size_ordered']
REASONS = {
    'fit': 'fit', 'size issue': 'fit', 'too small': 'fit', 'too large': 'fit',
    'size too small': 'fit', 'size too large': 'fit',
    'quality': 'quality', 'poor quality': 'quality',
    'quality not as expected': 'quality',
    'not as shown': 'not_as_shown', 'not_as_shown': 'not_as_shown',
    'different from image': 'not_as_shown',
    'wrong or damaged item': 'wrong_or_damaged', 'wrong_or_damaged': 'wrong_or_damaged',
    'wrong item': 'wrong_or_damaged', 'damaged item': 'wrong_or_damaged',
    'wrong item received': 'wrong_or_damaged', 'damaged or defective': 'wrong_or_damaged',
    'delivery': 'delivery', 'late delivery': 'delivery',
    'delivered late': 'delivery',
    'changed mind': 'changed_mind', 'changed_mind': 'changed_mind',
    'changed my mind': 'changed_mind',
    'unclear': 'unclear',
}
OUTPUT_COLUMNS = ['reason', 'status', 'confidence', 'evidence', 'direction', 'area',
                  'detail', 'multiple_reasons', 'route', 'note', 'trace', 'calls']
TRACKING_COLUMNS = ['row_id', 'data_quality_note']


def load_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError('CSV exceeds the 10 MB limit. Export a smaller period.')
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise ValueError('Cannot read this CSV. Use a UTF-8 CSV with a header row.') from None
    missing = sorted(set(REQUIRED) - {'return_id'} - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    collisions = sorted(set(OUTPUT_COLUMNS + TRACKING_COLUMNS) & set(frame.columns))
    if collisions:
        raise ValueError('Upload original input, not a labelled export. Reserved columns: ' + ', '.join(collisions))
    if frame.empty or len(frame) > 10000:
        raise ValueError('Upload between 1 and 10,000 rows.')
    if 'return_id' not in frame:
        frame.insert(0, 'return_id', '')
    for column in REQUIRED:
        if column != 'other_text':
            frame[column] = frame[column].str.strip()
    blank_ids = frame.return_id.eq('')
    repeated_ids = ~blank_ids & frame.return_id.duplicated(keep=False)
    duplicate_rows = frame.duplicated(keep=False)
    notes = [[] for _ in range(len(frame))]
    for mask, message in [
        (blank_ids, 'Missing return ID; use row_id for tracking.'),
        (repeated_ids, 'Repeated return ID; check whether these are separate items.'),
        (duplicate_rows, 'Possible duplicate row; retained for review.'),
    ]:
        for index in frame.index[mask]:
            notes[index].append(message)
    frame['row_id'] = [f'row-{index + 1:05d}' for index in range(len(frame))]
    frame['data_quality_note'] = [' '.join(items) for items in notes]
    dates = pd.to_datetime(frame.return_date, format='%Y-%m-%d', errors='coerce')
    if dates.isna().any() or not frame.return_date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('Every return_date must be a valid date in YYYY-MM-DD format.')
    values = frame.dropdown_reason.str.casefold()
    unknown = sorted(set(values) - set(REASONS) - {'other'})
    if unknown:
        raise ValueError('Unmapped dropdown reasons: ' + ', '.join(repr(x) for x in unknown)
                         + '. Map these labels before running.')
    if frame.other_text.str.len().gt(4000).any():
        raise ValueError('A comment exceeds 4,000 characters. Review it before uploading.')
    return frame


def classify_row(row, classifier):
    dropdown = row['dropdown_reason'].strip().casefold()
    if dropdown != 'other':
        reason = REASONS[dropdown]
        return base_result(reason=reason, status='Unclear' if reason == 'unclear' else 'Passed through',
                           note='Original dropdown mapped in code; details not inferred.')
    return classifier.classify(row['other_text'])


def run_batch(frame, classifier, progress=None, workers=4):
    records = frame.to_dict('records')
    outputs = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(classify_row, row, classifier): i for i, row in enumerate(records)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                result = future.result()
            except Exception:
                result = base_result(reason='unclassified', status='Failed',
                                     note='Unexpected row processing failure. Inspect configuration and retry.')
            outputs[index] = {**records[index], **result}
            if progress:
                progress(done, len(records))
    return pd.DataFrame(outputs)


def aggregate(frame, minimum=3):
    """Rank counts, not return rates. Unknown directions remain in the denominator."""
    fit = frame[(frame.reason == 'fit') & frame.status.isin(['Classified', 'Passed through'])].copy()
    missing = fit[GROUPS].eq('').any(axis=1)
    missing_count = int(missing.sum())
    rows = []
    for key, group in fit[~missing].groupby(GROUPS, sort=True):
        n = len(group)
        small = int(group.direction.eq('too_small').sum())
        large = int(group.direction.eq('too_large').sum())
        quotes = list(dict.fromkeys(group.evidence[group.evidence.ne('')].tolist()))[:3]
        rows.append(dict(zip(GROUPS, key), fit_complaints=n, too_small=small, too_large=large,
                         unknown_direction=n-small-large, too_small_share=small/n,
                         too_large_share=large/n, evidence=' | '.join(quotes),
                         eligible=n >= minimum))
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(['fit_complaints', *GROUPS], ascending=[False, True, True, True])
    return table, missing_count


def export_csv(frame):
    safe = frame.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda value: json.dumps(value, ensure_ascii=False)
                                 if isinstance(value, (list, dict)) else value)
        # CSV quoting alone does not prevent formulas executing in spreadsheet software.
        safe[col] = safe[col].map(lambda value: "'" + value
                                 if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@'))
                                 else value)
    return safe.to_csv(index=False).encode('utf-8-sig')


def fetch_prices(model_ids):
    """Public catalogue only; no key or customer data in this request."""
    try:
        with urlopen('https://openrouter.ai/api/v1/models', timeout=10) as response:
            catalogue = json.load(response)['data']
        rates = {}
        for model in catalogue:
            if model['id'] in model_ids:
                pricing = model['pricing']
                rates[model['id']] = {'input': float(pricing['prompt']), 'output': float(pricing['completion'])}
        return rates, datetime.now(timezone.utc).isoformat()
    except Exception:
        return {}, None


def cost_report(results, prices):
    calls = [call for result in results for call in result['calls']]
    rows = []
    total = 0.0
    complete = True
    for model in sorted({c['model'] for c in calls}):
        group = [c for c in calls if c['model'] == model]
        known = all(c['input_tokens'] is not None and c['output_tokens'] is not None for c in group)
        input_tokens = sum(c['input_tokens'] or 0 for c in group)
        output_tokens = sum(c['output_tokens'] or 0 for c in group)
        rates = prices.get(model)
        cost = (input_tokens * rates['input'] + output_tokens * rates['output']) if rates and known else None
        complete &= cost is not None
        total += cost or 0
        rows.append(dict(model=model, calls=len(group), reported_input_tokens=input_tokens,
                         reported_output_tokens=output_tokens, usage_complete=known,
                         estimated_usd=cost, invalid_outputs=sum(c['outcome'] == 'invalid_output' for c in group)))
    return pd.DataFrame(rows), total if complete else None
