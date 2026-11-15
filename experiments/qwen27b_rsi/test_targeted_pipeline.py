"""Offline write → QFAD → original lines → reward → training-feedback integration."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from medical_navigation import FIELDS, NavigationGraph, extraction_messages
from run_targeted_writer import paired_result
from train_extractor import evaluate_candidate


class PipelineTests(unittest.TestCase):
    @patch('medical_navigation.rerank_tokens', return_value=1)
    @patch('medical_navigation.base.rerank_request', return_value={})
    @patch('medical_navigation.base.rerank', return_value=[{'index': 0}])
    @patch.object(NavigationGraph, 'embed', lambda self, texts: np.ones((len(texts), 3)) / np.sqrt(3))
    def test_localizes_failed_write_by_original_line(self, *unused):
        original = 'Visit date\nDoctor: stop medication A.\nOther history.\nNext call.\nPatient: I stopped medication A.'
        question = {'query_id': 'q', 'question': 'Was the medication actually stopped?'}
        patient = SimpleNamespace(persona=1, history=[(0, original)], groups=[(None, [question])],
            stores={'chunk': {'ids': ['s'], 'texts': [original], 'vectors': np.ones((1, 3)) / np.sqrt(3)}})
        graph = NavigationGraph(patient)
        sources = [graph.sources['s']]
        condition = {'criterion': 'treatment_status', 'description': 'Reported execution',
                     'anchors': [{'source_id': 's', 'span_id': 4, 'quote': original.splitlines()[4]}]}
        episode = {'source_ids': ['s'], 'queries': [{'query_id': 'q', 'stage': 0,
                   'requirements': {'bundles': [{'conditions': [condition]}]}}]}
        class Reader:
            def answer(self, candidate, q, stage, label):
                retrieval = candidate.retrieve(q['question'], stage)
                return {'score': {'correct': 4 in retrieval['source_windows'][0]['line_ids']},
                        'retrieval': retrieval}
        class Coverage:
            def score(self, question, requirements, retrieval):
                ok = original.splitlines()[4] in retrieval['documents'][0]
                return {'coverage': float(ok), 'complete': ok, 'bundles': [{'conditions': [{'covered': ok}]}]}
        judge = SimpleNamespace(chat=lambda **kwargs: SimpleNamespace(usage={}, content=json.dumps({
            'facts': [{'id': 'f', 'supported': True}], 'relations': []})))
        outputs = []
        with TemporaryDirectory() as directory:
            for line, relation in [(1, 'was advised to stop'), (4, 'reported stopping')]:
                fact = dict.fromkeys(FIELDS, '')
                fact.update(id='f', subject='patient', relation=relation, object='medication A',
                            polarity='affirmed', source_id='s', evidence_ids=[line])
                row = {'finished': True, 'text': json.dumps({'facts': [fact], 'relations': []})}
                result, candidate = evaluate_candidate(row, graph, sources, {}, episode, Reader(), Coverage(),
                                                       judge, Path(directory), str(line), collect_hint=True)
                citation = next(iter(candidate.state['facts'].values()))['citations'][0]
                self.assertEqual(candidate.resolve_citation(citation), (line, original.splitlines()[line]))
                outputs.append(result)
        self.assertIn('lines 4–4', outputs[0]['text_feedback'])
        self.assertIsNone(outputs[1]['text_feedback'])
        self.assertGreater(outputs[1]['feedback']['reward'], outputs[0]['feedback']['reward'])
        self.assertNotIn(question['question'], extraction_messages(sources)[-1]['content'])

    def test_paired_counts_do_not_hide_losses_or_ties(self):
        rows = {'guided': {'a': True, 'b': False}, 'frozen': {'a': False, 'b': False},
                'scalar': {'a': False, 'b': True}, 'raw': {'a': True, 'b': True}}
        scores = paired_result(rows, ['a', 'b'])
        self.assertEqual(scores['guided_vs']['scalar'], {'wins': ['a'], 'losses': ['b']})
        self.assertTrue(scores['rl_gain'])
        self.assertFalse(scores['beats_raw'])
        self.assertFalse(scores['guidance_gain_over_scalar'])
        with self.assertRaises(ValueError):
            paired_result(rows, ['a'])
