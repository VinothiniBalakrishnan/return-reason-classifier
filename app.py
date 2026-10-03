"""Run with: streamlit run app.py"""
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from classifier import Classifier, Settings
from processing import aggregate, cost_report, export_csv, fetch_prices, load_csv, run_batch

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='Dhaga & Co. | Returns', page_icon='🧵', layout='wide')
st.title('Dhaga & Co. · Returns Reason Classifier')
st.caption('Turn customer comments into evidence for your next category review.')

with st.expander('Run setup', expanded=False):
    configured_key = os.getenv('OPENROUTER_API_KEY', '')
    try:
        configured_key = configured_key or st.secrets.get('OPENROUTER_API_KEY', '')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    supplied_key = ''
    if configured_key:
        st.success('OpenRouter key configured')
    else:
        st.info('Add your key to .env or enter it below for this session.')
        supplied_key = st.text_input('OpenRouter API key', type='password', key='session_api_key')
    settings = Settings.from_env(configured_key or supplied_key)
    st.caption('Bulk model')
    st.code(settings.small_model, language=None)
    st.caption('Escalation model')
    st.code(settings.strong_model, language=None)
    minimum = st.number_input('Minimum fit complaints per ranked group', min_value=1, max_value=100,
                              value=3, help='Demo threshold, not a statistical significance test.')
    inr_rate = st.number_input('Optional INR per USD (0 = USD only)', min_value=0.0, value=0.0, step=1.0)
    st.caption('Comments are sent to OpenRouter only when you press a classify button.')
    if st.button('Clear results'):
        for key in ['batch', 'single']:
            st.session_state.pop(key, None)


def get_classifier():
    try:
        return Classifier(settings)
    except ValueError as exc:
        st.error(str(exc))
        return None
    except Exception:
        st.error('Could not initialize the model integration. Check model settings and installed dependencies.')
        return None


def show_cost(results, rates, rate_time, elapsed, other_rows):
    table, total = cost_report(results, rates)
    with st.expander('Run report', expanded=True):
        left, right = st.columns(2)
        left.metric('Elapsed time', f'{elapsed:.1f} s')
        right.metric('Estimated token cost (USD)', 'Unavailable' if total is None else f'${total:.6f}')
        if not table.empty:
            st.dataframe(table, hide_index=True, width='stretch')
        else:
            st.caption('No model calls were needed.')
        if total is None:
            st.warning('Some usage or model prices are missing. Cost is unknown, not zero; check OpenRouter activity.')
        elif inr_rate:
            st.caption(f'Estimated ₹{total * inr_rate:.4f} using your supplied rate of ₹{inr_rate:g}/USD.')
        if total is not None and other_rows:
            weekly = total / other_rows * 6547
            st.caption(f'Conditional weekly projection: ${total:.6f} ÷ {other_rows} processed Other rows '
                       f'× 6,547 ≈ ${weekly:.4f}. Assumes this sample represents the weekly mix.')
        if rate_time:
            st.caption(f'Catalogue pricing snapshot: {rate_time}. Token estimates exclude account fees and may '
                       'differ from provider billing, caching and routing. Verify actual spend in OpenRouter.')
        st.caption('Tokens include returned responses rejected during validation. Network failures can have unreported usage.')


analyze, review, single = st.tabs(['Analyze returns', 'Review', 'Try a comment'])
with analyze:
    source = st.radio('Input source', ['Upload CSV', 'Bundled synthetic sample'], horizontal=True)
    raw = None
    if source == 'Bundled synthetic sample':
        st.warning('Synthetic demo data written for development. This is not the client dataset or an accuracy benchmark.')
        raw = (ROOT / 'data' / 'sample_returns.csv').read_bytes()
    else:
        uploaded = st.file_uploader('Returns CSV', type=['csv'])
        if uploaded is not None:
            raw = uploaded.getvalue()
        else:
            st.info('Upload your teammate’s CSV, or preview the bundled synthetic sample.')
    fingerprint = hashlib.sha256(raw).hexdigest() if raw is not None else None
    if st.session_state.get('input_hash') != fingerprint:
        st.session_state.pop('batch', None)
        st.session_state['input_hash'] = fingerprint
    frame = None
    if raw is not None:
        try:
            frame = load_csv(raw)
        except ValueError as exc:
            st.error(str(exc))
        if frame is not None:
            st.caption(f'{len(frame):,} rows · {frame.return_date.min()} to {frame.return_date.max()}')
            flagged = frame[frame.data_quality_note.ne('')]
            if not flagged.empty:
                st.warning(f'{len(flagged):,} rows have missing/repeated return IDs or possible duplicates. '
                           'All rows are retained and receive a unique tracking ID within this file. '
                           'Original return IDs are kept after trimming whitespace. '
                           'Counts include retained duplicates; check them before interpreting results.')
                with st.expander('Data cleaning review', expanded=True):
                    st.dataframe(flagged, hide_index=True, width='stretch')
            st.dataframe(frame.head(10), hide_index=True, width='stretch')
            process_all = st.checkbox('Process all rows', key=f'all_{fingerprint[:12]}')
            if process_all:
                limit = len(frame)
            else:
                limit = st.number_input('Rows to process from the start of this file', min_value=1,
                                        max_value=min(len(frame), 1000), value=min(25, len(frame)),
                                        key=f'limit_{fingerprint[:12]}')
            st.caption(f'Classify returns will process {limit:,} of {len(frame):,} rows. '
                       'Only Other comments need model classification; existing dropdown reasons are mapped automatically.')
            if st.button('Classify returns', type='primary'):
                engine = get_classifier()
                if engine:
                    st.session_state.pop('batch', None)
                    progress = st.progress(0.0, text='Preparing run…')
                    rates, rate_time = fetch_prices([settings.small_model, settings.strong_model])
                    started = time.monotonic()
                    results = run_batch(frame.head(int(limit)), engine,
                                        lambda done, total: progress.progress(done / total, text=f'{done}/{total} rows processed'))
                    st.session_state['batch'] = dict(results=results, prices=rates, rate_time=rate_time,
                                                    elapsed=time.monotonic() - started, source=source, total=len(frame))
    batch = st.session_state.get('batch')
    if batch:
        results = batch['results']
        st.subheader('Results')
        st.caption(f"{batch['source']} · {len(results)} of {batch['total']} input rows processed")
        cols = st.columns(3)
        cols[0].metric('Processed', len(results))
        cols[1].metric('Ready to use', int(results.status.isin(['Classified', 'Passed through']).sum()))
        cols[2].metric('Needs review', int((~results.status.isin(['Classified', 'Passed through'])
                                          | results.data_quality_note.ne('')).sum()))
        other = results[results.dropdown_reason.str.casefold().eq('other')]
        if len(other):
            st.write('How Other comments were labelled')
            counts = other.groupby('reason').size().rename('Count')
            st.bar_chart(counts)
            coverage = other.status.eq('Classified').sum() / len(other)
            st.caption(f'Specific trusted labels: {coverage:.1%} of {len(other)} processed Other rows. '
                       'Unclassified includes validation/provider failures; see status counts below.')
        st.dataframe(results.groupby('status').size().rename('Rows').reset_index(), hide_index=True)
        st.subheader('Fit complaints to investigate')
        st.caption('Ranked by complaint count, not return rate. These observations do not prove a cause or prescribe a fix.')
        groups, missing = aggregate(results, int(minimum))
        if groups.empty:
            st.info('No trusted fit complaints with complete grouping fields.')
        else:
            ranked = groups[groups.eligible].drop(columns='eligible')
            if ranked.empty:
                st.info('Not enough rows: no group meets the minimum fit-complaint count.')
            else:
                st.dataframe(ranked, hide_index=True, width='stretch', column_config={
                    'too_small_share': st.column_config.NumberColumn('Too small share', format='percent'),
                    'too_large_share': st.column_config.NumberColumn('Too large share', format='percent'),
                })
            st.caption(f'{int((~groups.eligible).sum())} smaller groups excluded; '
                       f'{missing} fit rows lack vendor, category or size. Shares include unknown directions.')
        st.download_button('Download labelled CSV', export_csv(results), 'labelled_returns.csv', 'text/csv')
        show_cost(results.to_dict('records'), batch['prices'], batch['rate_time'], batch['elapsed'], len(other))

with review:
    batch = st.session_state.get('batch')
    if not batch:
        st.info('Run a CSV in Analyze returns to populate the review queue.')
    else:
        results = batch['results']
        queue = results[~results.status.isin(['Classified', 'Passed through'])
                        | results.data_quality_note.ne('')]
        if queue.empty:
            st.success('No rows need review in this run.')
        else:
            st.dataframe(queue[['row_id', 'return_id', 'data_quality_note', 'other_text', 'status', 'reason', 'evidence', 'note']],
                         hide_index=True, width='stretch')
            selected = st.selectbox('Inspect a row', queue.row_id.tolist())
            row = queue[queue.row_id.eq(selected)].iloc[0]
            if row['data_quality_note']:
                st.warning(row['data_quality_note'])
            st.write(row['note'])
            st.json(row['trace'])
            st.caption('Untrusted classifications are excluded from rankings. Data-quality flags alone do not '
                       'exclude rows: repeated IDs may represent separate items. Corrections are not saved in this MVP.')

with single:
    with st.form('single_comment'):
        comment = st.text_area('Customer return comment', placeholder='size chota hai, kamar pe tight', max_chars=4000)
        submitted = st.form_submit_button('Classify comment')
    st.caption('Try “theek nahi tha” to demonstrate ambiguity handling.')
    if submitted:
        engine = get_classifier()
        if engine:
            rates, rate_time = fetch_prices([settings.small_model, settings.strong_model])
            started = time.monotonic()
            with st.spinner('Reading the comment…'):
                result = engine.classify(comment)
            st.session_state['single'] = dict(result=result, comment=comment, prices=rates,
                                              rate_time=rate_time, elapsed=time.monotonic()-started)
    saved = st.session_state.get('single')
    if saved:
        result = saved['result']
        st.caption('Result for the last submitted comment')
        st.text(saved['comment'])
        st.write(f"**{result['reason'].replace('_', ' ').title()}** · {result['status']}")
        if result['evidence']:
            st.text('Evidence: ' + result['evidence'])
        if result['direction']:
            st.write(f"Fit: {result['direction']} · {result['area']}")
        if result['detail']:
            st.write('Detail: ' + result['detail'])
        st.caption('Route: ' + result['route'])
        if result['note']:
            st.info(result['note'])
        with st.expander('Model trace'):
            st.json(result['trace'])
        show_cost([result], saved['prices'], saved['rate_time'], saved['elapsed'], 1)
