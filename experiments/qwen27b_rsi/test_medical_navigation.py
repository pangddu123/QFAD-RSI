"""Regression cases for extraction replay and relation-evidence navigation."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from medical_navigation import NavigationGraph, FIELDS, parse_extraction, admit_extraction


def fact(source, span, relation='takes'):
    values = dict.fromkeys(FIELDS, '')
    values.update(subject='patient', relation=relation, object='drug', polarity='affirmed')
    values['citations'] = [{'source_id': source, 'span_id': span, 'quote': 'Patient: reported event'}]
    return values


def patient():
    texts = ['[2024-01-01]\nPatient: reported event',
             '[2024-01-02]\nPatient: reported event',
             '[2024-01-03]\nDoctor: discussion\n\nOther context\n\nPatient: I stopped the earlier regimen after the new plan.']
    return SimpleNamespace(persona=1, history=list(zip([0, 0, 1], texts)), stores={
        'chunk': {'ids': ['a', 'b', 'c'], 'texts': texts, 'vectors': np.ones((3, 3)) / np.sqrt(3)}})


def embeddings(self, texts):
    return np.ones((len(texts), 3)) / np.sqrt(3)


class NavigationTests(unittest.TestCase):
    def test_model_copied_citations_are_replaced_with_original_spans(self):
        import json
        graph = NavigationGraph(patient())
        values = {k: v for k, v in fact('a', 1).items() if k in FIELDS}
        values.update(id='f0', source_id='a', evidence_ids=[1],
                      citations=[{'source_id': 'invented', 'span_id': 99, 'quote': 'invented text'}])
        parsed = parse_extraction({'text': json.dumps({'facts': [values], 'relations': []}),
                                   'finished': True}, [graph.sources['a']])
        self.assertEqual(parsed['facts'][0]['citations'], [
            {'source_id': 'a', 'span_id': 1, 'quote': graph.sources['a']['text'].splitlines()[1]}])

    def test_local_error_preserves_supported_facts_and_edges(self):
        extraction = {'facts': [{'id': 'f0'}, {'id': 'f1'}, {'id': 'f2'}], 'relations': [
            {'from': 'f0', 'to': 'f1'}, {'from': 'f0', 'to': 'f2'}, {'from': 'f0', 'to': 'old'}]}
        verdict = {'facts': [{'id': f, 'supported': ok, 'reason': 'source judgement'}
                            for f, ok in [('f0', True), ('f1', True), ('f2', False)]],
                   'relations': [{'index': i, 'supported': True, 'reason': 'source judgement'} for i in range(3)]}
        accepted, report = admit_extraction(extraction, verdict, {'old': {}})
        self.assertEqual([f['id'] for f in accepted['facts']], ['f0', 'f1'])
        self.assertEqual([r['to'] for r in accepted['relations']], ['f1', 'old'])
        self.assertEqual(len(report['rejected']), 2)
        self.assertEqual(len(extraction['facts']), 3)

    def test_bad_relation_does_not_remove_its_supported_endpoints(self):
        extraction = {'facts': [{'id': 'f0'}, {'id': 'f1'}],
                      'relations': [{'from': 'f0', 'to': 'f1'}]}
        verdict = {'facts': [{'id': f, 'supported': True, 'reason': 'explicit report'} for f in ('f0', 'f1')],
                   'relations': [{'index': 0, 'supported': False, 'reason': 'causal direction reversed'}]}
        accepted, report = admit_extraction(extraction, verdict)
        self.assertEqual(accepted['facts'], extraction['facts'])
        self.assertEqual(accepted['relations'], [])
        self.assertEqual(report['accepted_facts'], 2)

    def test_replaced_identifier_cannot_silently_become_new_fact(self):
        old = fact('a', 1)
        graph = NavigationGraph(patient(), {'facts': {'a:fact:0': old}, 'relations': []})
        new = {k: old[k] for k in FIELDS}
        new.update(id='f0', source_id='a', evidence_ids=[1], relation='stopped')
        output = {'facts': [new], 'relations': [
            {'from': 'a:fact:0', 'to': 'f0', 'relation': 'precedes', 'source_id': 'a', 'evidence_ids': [1]}]}
        import json
        parsed = parse_extraction({'text': json.dumps(output), 'finished': True},
                                  [graph.sources['a']], graph.state['facts'])
        with self.assertRaises(ValueError):
            graph.extract_into(parsed, ['a'])
        self.assertEqual(graph.state['facts']['a:fact:0']['relation'], 'takes')

    def graph(self):
        return NavigationGraph(patient(), {'facts': {'fa': fact('a', 1), 'fb': fact('b', 1, 'stopped')},
            'relations': [{'from': 'fa', 'to': 'fb', 'relation': 'state_change', 'citations': [
                {'source_id': 'c', 'span_id': 5, 'quote': 'Patient: I stopped the earlier regimen after the new plan.'}]}]})

    def test_reextract_relation_source_replaces_its_old_edges(self):
        graph = self.graph()
        updated = graph.extract_into({'facts': [], 'relations': []}, ['c'])
        self.assertEqual(updated.state['facts'], graph.state['facts'])
        self.assertEqual(updated.state['relations'], [])
        self.assertEqual(len(graph.state['relations']), 1)

    @patch('medical_navigation.rerank_tokens', return_value=1)
    @patch('medical_navigation.base.rerank_request', return_value={})
    @patch('medical_navigation.base.rerank', side_effect=lambda q, docs: [{'index': i} for i in range(len(docs))])
    @patch.object(NavigationGraph, 'embed', embeddings)
    def test_relation_source_is_navigable_and_delivered(self, *unused):
        evidence = self.graph().retrieve('What changed after the new plan?', 1)
        self.assertIn('c', evidence['source_ids'])
        self.assertTrue(any('I stopped the earlier regimen after the new plan.' in d for d in evidence['documents']))

    @patch('medical_navigation.rerank_tokens', return_value=1)
    @patch('medical_navigation.base.rerank_request', return_value={})
    @patch('medical_navigation.base.rerank', side_effect=lambda q, docs: [{'index': i} for i in range(len(docs))])
    @patch.object(NavigationGraph, 'embed', embeddings)
    def test_future_relation_source_is_not_visible(self, *unused):
        evidence = self.graph().retrieve('What changed?', 0)
        self.assertNotIn('c', evidence['source_ids'])
        self.assertNotIn('c', evidence['qfad_source_order'])


if __name__ == '__main__':
    unittest.main()
