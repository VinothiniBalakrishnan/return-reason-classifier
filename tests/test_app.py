"""Offline checks of failure paths, accounting and Streamlit session behavior."""
import unittest
import io
import json
import httpx
from openrouter import OpenRouter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from classifier import Classifier, Settings, Triage, FitDetail, QualityDetail, MismatchDetail, base_result
from processing import load_csv, run_batch, aggregate, cost_report, export_csv
from evaluate import score

ROOT = Path(__file__).resolve().parents[1]


def response(parsed, invalid=False):
    return dict(parsed=parsed, parsing_error='bad json' if invalid else None,
                raw=SimpleNamespace(usage_metadata={'input_tokens': 100, 'output_tokens': 20}))


def triage(reason='fit', evidence='tight', confidence='high', multiple=False):
    return Triage(reason=reason, evidence=evidence, confidence=confidence, multiple_reasons=multiple)


def fit(evidence='tight', direction='too_small', multiple=False):
    return FitDetail(direction=direction, area='waist', confidence='high', evidence=evidence,
                     multiple_observations=multiple)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Classifier(Settings(api_key='offline-placeholder'))
        self.small = self.engine.settings.small_model
        self.strong = self.engine.settings.strong_model
        for key in self.engine.chains:
            self.engine.chains[key] = RunnableLambda(lambda _: (_ for _ in ()).throw(AssertionError('Unexpected model call')))

    def answer(self, model, stage, parsed):
        self.engine.chains[model, stage] = RunnableLambda(lambda _: response(parsed))

    def test_blank_never_calls_models(self):
        for text in ['', '  ', '😕!', '...']:
            result = self.engine.classify(text)
            self.assertEqual(result['status'], 'Unclear')
            self.assertEqual(result['calls'], [])

    def test_fit_routes_and_counts_both_calls(self):
        self.answer(self.small, 'triage', triage())
        self.answer(self.small, 'fit', fit())
        result = self.engine.classify('waist tight')
        self.assertEqual(result['status'], 'Classified')
        self.assertEqual(result['direction'], 'too_small')
        self.assertEqual([c['stage'] for c in result['calls']], ['triage', 'fit'])

    def test_quality_and_mismatch_routes(self):
        for reason, detail in [('quality', QualityDetail(issue='stitching', confidence='high', evidence='broken')),
                               ('not_as_shown', MismatchDetail(issue='colour', confidence='high', evidence='blue'))]:
            with self.subTest(reason=reason):
                self.answer(self.small, 'triage', triage(reason=reason, evidence=detail.evidence))
                self.answer(self.small, reason, detail)
                result = self.engine.classify(detail.evidence)
                self.assertEqual(result['detail'], detail.issue)
                self.assertEqual(result['status'], 'Classified')

    def test_invalid_evidence_retries_then_unclassified(self):
        self.answer(self.small, 'triage', triage(evidence='not actually in text'))
        result = self.engine.classify('tight')
        self.assertEqual(result['status'], 'Unclassified')
        self.assertEqual(len(result['calls']), 2)
        self.assertTrue(all(c['outcome'] == 'invalid_output' for c in result['calls']))

    def test_low_confidence_escalates_once(self):
        self.answer(self.small, 'triage', triage(confidence='low'))
        self.answer(self.strong, 'triage', triage())
        self.answer(self.strong, 'fit', fit())
        result = self.engine.classify('tight')
        self.assertEqual(result['status'], 'Classified')
        self.assertEqual(result['route'], 'small → strong')
        self.assertEqual(len(result['calls']), 3)

    def test_multiple_reasons_never_ranked(self):
        self.answer(self.small, 'triage', triage(multiple=True))
        self.answer(self.strong, 'triage', triage(multiple=True))
        result = self.engine.classify('tight and poor stitching')
        self.assertEqual(result['status'], 'Review')
        self.assertEqual(len(result['calls']), 2)

    def test_opposing_fit_observations_require_review(self):
        for model in [self.small, self.strong]:
            self.answer(model, 'triage', triage())
            self.answer(model, 'fit', fit(multiple=True))
        result = self.engine.classify('tight bust loose waist')
        self.assertEqual(result['status'], 'Review')

    def test_strong_failure_does_not_trust_small_guess(self):
        self.answer(self.small, 'triage', triage(confidence='low'))
        class Denied(Exception):
            status_code = 401
        self.engine.chains[self.strong, 'triage'] = RunnableLambda(lambda _: (_ for _ in ()).throw(Denied('secret-value')))
        result = self.engine.classify('tight')
        self.assertEqual(result['status'], 'Failed')
        self.assertNotIn('secret-value', result['note'])
        self.assertEqual(len(result['calls']), 2)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(ROOT / 'data/sample_returns.csv', dtype=str, keep_default_na=False)

    def test_bad_inputs(self):
        for change in ['missing', 'date', 'dropdown', 'reserved', 'tracking']:
            frame = self.frame.copy()
            if change == 'missing': frame = frame.drop(columns='vendor_id')
            if change == 'tracking': frame['row_id'] = 'source-value'
            if change == 'date': frame.loc[0, 'return_date'] = '2026-02-31'
            if change == 'dropdown': frame.loc[0, 'dropdown_reason'] = 'Surprise'
            if change == 'reserved': frame['status'] = 'Classified'
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_csv(frame.to_csv(index=False).encode())

    def test_ids_keep_leading_zeroes(self):
        self.frame.loc[0, 'return_id'] = '00001'
        parsed = load_csv(self.frame.to_csv(index=False).encode())
        self.assertEqual(parsed.loc[0, 'return_id'], '00001')

    def test_missing_and_repeated_ids_keep_all_rows(self):
        frame = self.frame.head(4).copy()
        frame.loc[0, 'return_id'] = '   '
        frame.loc[1:2, 'return_id'] = '00001'
        parsed = load_csv(frame.to_csv(index=False).encode())
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed.return_id.tolist()[:3], ['', '00001', '00001'])
        self.assertTrue(parsed.row_id.is_unique)
        self.assertIn('Missing return ID', parsed.loc[0, 'data_quality_note'])
        self.assertTrue(parsed.loc[1:2, 'data_quality_note'].str.contains('Repeated return ID').all())
        self.assertEqual(parsed.loc[3, 'data_quality_note'], '')
        self.assertEqual(parsed.row_id.tolist(), load_csv(frame.to_csv(index=False).encode()).row_id.tolist())

    def test_absent_id_column_and_duplicate_rows(self):
        frame = self.frame.head(1).drop(columns='return_id')
        frame = pd.concat([frame, frame], ignore_index=True)
        parsed = load_csv(frame.to_csv(index=False).encode())
        self.assertEqual(len(parsed), 2)
        self.assertTrue(parsed.return_id.eq('').all())
        self.assertTrue(parsed.data_quality_note.str.contains('Possible duplicate row').all())
        result = run_batch(parsed, SimpleNamespace(classify=lambda _: base_result()))
        self.assertTrue(result.row_id.is_unique)
        self.assertEqual(result.data_quality_note.tolist(), parsed.data_quality_note.tolist())
        exported = pd.read_csv(io.BytesIO(export_csv(result)), keep_default_na=False)
        self.assertEqual(exported.row_id.tolist(), parsed.row_id.tolist())

    def test_batch_order_and_row_failure_isolation(self):
        class Stub:
            def classify(self, text):
                if 'waist' in text: raise RuntimeError('failure')
                return base_result()
        result = run_batch(self.frame.head(5), Stub())
        self.assertEqual(result.return_id.tolist(), self.frame.head(5).return_id.tolist())
        self.assertEqual(len(result), 5)
        self.assertEqual(result.iloc[4].status, 'Passed through')
        self.assertIn('Failed', result.status.tolist())

    def test_aggregation_excludes_review_and_counts_unknown(self):
        records = []
        for status, direction, vendor in [('Classified', 'too_small', 'V1'), ('Passed through', '', 'V1'),
                                          ('Review', 'too_small', 'V1'), ('Classified', 'too_small', '')]:
            records.append(dict(vendor_id=vendor, category='Kurti', size_ordered='M', reason='fit',
                                status=status, direction=direction, evidence='tight' if direction else ''))
        groups, missing = aggregate(pd.DataFrame(records), minimum=3)
        self.assertEqual(missing, 1)
        self.assertEqual(groups.iloc[0].fit_complaints, 2)
        self.assertEqual(groups.iloc[0].too_small_share, 0.5)
        self.assertFalse(groups.iloc[0].eligible)

    def test_unknown_cost_is_not_zero(self):
        call = dict(model='m', input_tokens=100, output_tokens=20, outcome='valid')
        _, total = cost_report([{'calls': [call]}], {})
        self.assertIsNone(total)
        _, total = cost_report([{'calls': [call]}], {'m': {'input': .001, 'output': .002}})
        self.assertAlmostEqual(total, .14)
        _, total = cost_report([{'calls': [dict(call, input_tokens=None)]}], {'m': {'input': .001, 'output': .002}})
        self.assertIsNone(total)

    def test_csv_formula_escaping(self):
        content = export_csv(pd.DataFrame({'text': ['=1+1', ' @SUM(A1)', 'normal']})).decode('utf-8-sig')
        self.assertIn("'=1+1", content)
        self.assertIn("' @SUM(A1)", content)

    def test_evaluation_counts_abstentions_and_failures(self):
        gold = pd.DataFrame({'expected_reason': ['fit', 'fit', 'unclear'],
                             'expected_direction': ['too_small', 'too_small', '']})
        predictions = pd.DataFrame({'reason': ['fit', 'fit', 'unclear'],
                                    'direction': ['too_small', 'too_small', ''],
                                    'status': ['Classified', 'Review', 'Unclear']})
        result = score(gold, predictions)
        self.assertAlmostEqual(result['top_level_accuracy'], 2/3)
        self.assertEqual(result['fit_direction_accuracy'], .5)


class TransportTests(unittest.TestCase):
    def test_real_langchain_sdk_serialization_and_parsing_without_network(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            stage = body['response_format']['json_schema']['name']
            parsed = triage() if stage == 'Triage' else fit()
            return httpx.Response(200, json={
                'id': 'offline-test', 'object': 'chat.completion', 'created': 1,
                'model': body['model'], 'system_fingerprint': None,
                'choices': [{'index': 0, 'message': {'role': 'assistant',
                             'content': parsed.model_dump_json()}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
            })

        client = OpenRouter(api_key='offline', retry_config=None,
                            client=httpx.Client(transport=httpx.MockTransport(respond)))
        with patch('classifier.OpenRouter', return_value=client):
            result = Classifier(Settings(api_key='offline')).classify('waist tight')
        self.assertEqual(result['status'], 'Classified')
        self.assertEqual(result['direction'], 'too_small')
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0]['response_format']['json_schema']['strict'])
        self.assertEqual(sum(c['input_tokens'] for c in result['calls']), 200)


class InterfaceTests(unittest.TestCase):
    def test_no_key_sample_and_missing_key_message(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': ''}):
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            self.assertEqual(len(app.exception), 0)
            app.radio[0].set_value('Bundled synthetic sample').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.dataframe), 1)
            button = next(b for b in app.button if b.label == 'Classify returns')
            button.click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('OPENROUTER_API_KEY' in x.value for x in app.error))

    def test_review_can_select_separate_rows_with_same_return_id(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run()
        app.radio[0].set_value('Bundled synthetic sample').run()
        frame = pd.read_csv(ROOT / 'data/sample_returns.csv', dtype=str, keep_default_na=False).head(2)
        frame['return_id'] = 'same-return'
        frame = load_csv(frame.to_csv(index=False).encode())
        result = run_batch(frame, SimpleNamespace(classify=lambda _: base_result(reason='fit', status='Classified')))
        result.loc[0, 'note'] = 'First item'
        result.loc[1, 'note'] = 'Second item'
        app.session_state['batch'] = dict(results=result, prices={}, rate_time=None, elapsed=0.1,
                                          source='Test fixture', total=2)
        app.run()
        self.assertEqual(len(app.exception), 0)
        selector = next(x for x in app.selectbox if x.label == 'Inspect a row')
        self.assertEqual(selector.options, frame.row_id.tolist())
        selector.set_value(frame.row_id.iloc[1]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(x.value == 'Second item' for x in app.markdown))

    def test_completed_results_survive_rerun_without_model_calls(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run()
        app.radio[0].set_value('Bundled synthetic sample').run()
        frame = load_csv((ROOT / 'data/sample_returns.csv').read_bytes()).head(4)
        result = run_batch(frame, SimpleNamespace(classify=lambda _: base_result(reason='fit', status='Classified')))
        app.session_state['batch'] = dict(results=result, prices={}, rate_time=None, elapsed=0.1,
                                          source='Test fixture', total=4)
        with patch('classifier.Classifier.classify', side_effect=AssertionError('Rerun called model')):
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(x.value == '4' for x in app.metric))
            app.run()
            self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
