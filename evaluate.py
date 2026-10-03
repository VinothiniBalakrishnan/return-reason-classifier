"""Score manually labelled examples. This command makes real, billable API calls."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from classifier import Classifier, Settings
from processing import cost_report, export_csv, fetch_prices, run_batch


def score(gold, predictions):
    acceptable = predictions.status.isin(['Classified', 'Unclear'])
    correct = predictions.reason.eq(gold.expected_reason) & acceptable
    fit = gold.expected_reason.eq('fit') & gold.expected_direction.isin(['too_small', 'too_large'])
    direction_correct = (predictions.direction.eq(gold.expected_direction)
                         & predictions.reason.eq('fit') & predictions.status.eq('Classified'))
    specific = predictions.status.eq('Classified')
    per_reason = {}
    for reason in sorted(gold.expected_reason.unique()):
        mask = gold.expected_reason.eq(reason)
        per_reason[reason] = {'rows': int(mask.sum()), 'accuracy': float(correct[mask].mean())}
    return dict(rows=len(gold), top_level_accuracy=float(correct.mean()),
                specific_label_coverage=float(specific.mean()), fit_direction_rows=int(fit.sum()),
                fit_direction_accuracy=float(direction_correct[fit].mean()) if fit.any() else None,
                status_counts={str(k): int(v) for k, v in predictions.status.value_counts().items()},
                per_reason=per_reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold_csv', help='Columns: other_text, expected_reason, expected_direction')
    parser.add_argument('--output', default='evaluation_results.csv')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent / '.env')
    gold = pd.read_csv(args.gold_csv, dtype=str, keep_default_na=False)
    required = {'other_text', 'expected_reason', 'expected_direction'}
    if not required.issubset(gold.columns) or gold.empty:
        parser.error('A nonempty CSV with other_text, expected_reason, expected_direction is required.')
    allowed = {'fit', 'quality', 'not_as_shown', 'wrong_or_damaged', 'delivery', 'changed_mind', 'unclear'}
    if not set(gold.expected_reason).issubset(allowed):
        parser.error('Gold labels must use the documented reason taxonomy.')
    if len(gold) > 1000 or gold.other_text.str.len().gt(4000).any():
        parser.error('Maximum 1,000 examples and 4,000 characters per comment.')
    try:
        engine = Classifier(Settings.from_env())
    except ValueError as exc:
        parser.error(str(exc))
    rates, timestamp = fetch_prices([engine.settings.small_model, engine.settings.strong_model])
    frame = gold[['other_text']].assign(dropdown_reason='Other', return_id=[f'EVAL-{i}' for i in range(len(gold))])
    started = time.monotonic()
    predictions = run_batch(frame, engine)
    metrics = score(gold, predictions)
    _, estimated_cost = cost_report(predictions.to_dict('records'), rates)
    metrics.update(estimated_usd=estimated_cost, price_snapshot=timestamp,
                   elapsed_seconds=round(time.monotonic()-started, 2))
    predictions['expected_reason'] = gold.expected_reason
    predictions['expected_direction'] = gold.expected_direction
    Path(args.output).write_bytes(export_csv(predictions))
    print(json.dumps(metrics, indent=2))
    print('Bundled smoke cases are development examples, not an untouched gold set.')


if __name__ == '__main__':
    main()
