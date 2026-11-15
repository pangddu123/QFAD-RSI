"""Failure attribution, clinical prompt scope and the action-token distillation loss."""
from types import SimpleNamespace
import unittest

import torch

from policy_worker import guided_logps
from targeted_guidance import extraction_hint


class GuidanceTests(unittest.TestCase):
    sources = {'s': {'text': 'visit date\nDoctor: stop it.\nPatient: I stopped it.'}}
    queries = [{'requirements': {'bundles': [{'conditions': [{'criterion': 'treatment_status',
        'description': 'actual discontinuation', 'anchors': [
            {'source_id': 's', 'span_id': 2, 'quote': 'Patient: I stopped it.'}]}]}]}}]

    def hint(self, evidence, complete=False, sources=None):
        result = {'parsed': True, 'admission': {'rejected': []}, 'correct': [False], 'coverage': [
            {'complete': complete, 'bundles': [{'conditions': [{'covered': complete}]}]}]}
        return extraction_hint(self.sources if sources is None else sources, self.queries, result, [evidence])

    def test_advice_window_misses_execution(self):
        hint = self.hint({'source_windows': [{'source_id': 's', 'line_ids': [0, 1]}]})
        self.assertIn('window missed', hint)
        self.assertIn('lines 2–2', hint)
        self.assertIn('patient reporting', hint)

    def test_wrong_answer_with_delivered_evidence_is_not_writer_failure(self):
        trace = {'source_windows': [{'source_id': 's', 'line_ids': [0, 1, 2]}]}
        self.assertIsNone(self.hint(trace, complete=True))
        self.assertIsNone(self.hint(trace))
        self.assertIsNone(self.hint({'source_windows': []}, sources={}))

    def test_distinguish_diffusion_from_delivery_failure(self):
        self.assertIn('no reachable', self.hint({'source_windows': []}))
        self.assertIn('reranking budget', self.hint({'source_windows': [], 'qfad_source_order': ['s']}))
        self.assertIn('shortlisted', self.hint({'source_windows': [], 'rerank_source_ids': ['s']}))

    def test_full_vocabulary_kl_only_updates_unhinted_action_positions(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.student = torch.nn.Parameter(torch.zeros(5, 8))
                self.teacher = torch.nn.Parameter(torch.arange(48.).reshape(6, 8) / 3)

            def forward(self, input_ids, logits_to_keep, **kwargs):
                table = self.teacher if input_ids.shape[1] == 6 else self.student
                return SimpleNamespace(logits=table[-logits_to_keep:].unsqueeze(0))

        model = TinyPolicy()
        chosen, kl = guided_logps(model, [1, 2, 3], [4, 5], [1, 2, 3, 7])
        expected = torch.nn.functional.kl_div(model.teacher.detach()[3:5].log_softmax(-1),
            model.student.detach()[2:4].softmax(-1), reduction='none').sum(-1)
        torch.testing.assert_close(kl, expected)
        torch.testing.assert_close(chosen, model.student.detach()[2:4].log_softmax(-1)[[0, 1], [4, 5]])
        kl.sum().backward()
        self.assertIsNone(model.teacher.grad)
        self.assertEqual(model.student.grad[[0, 1, 4]].count_nonzero(), 0)
        self.assertEqual(model.student.grad[2:4].count_nonzero(), 16)


if __name__ == '__main__':
    unittest.main()
