"""Reward ordering and zero-signal group checks."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json

from clinical_evidence import EvidenceScorer, group_advantages, task_reward, extraction_reward, has_task_signal


class RewardTests(unittest.TestCase):
    def test_identical_evidence_has_one_paired_judgement(self):
        class Judge:
            calls = 0
            def responses(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(content=json.dumps({'bundles': [{'conditions': [
                    {'covered': self.calls == 1, 'source_ids': ['s'], 'reason': 'test'}]}]}),
                    usage={'total_tokens': 10})
        judge = Judge()
        scorer = EvidenceScorer(judge)
        requirements = {'bundles': [{'conditions': [{'description': 'Medication was stopped'}]}]}
        evidence = {'documents': ['Patient: I stopped the medication.'], 'source_ids': ['s']}
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: scorer.score('Current status?', requirements, evidence), range(4)))
        self.assertEqual(judge.calls, 1)
        self.assertEqual([r['coverage'] for r in results], [1.] * 4)
        self.assertEqual(sum(not r['cache_hit'] for r in results), 1)

    def reward(self, correct=True, coverage=1.):
        return task_reward([{'score': {'correct': correct}}],
                             [{'coverage': coverage, 'complete': coverage == 1.}])

    def test_correct_answer_without_evidence_receives_no_success_credit(self):
        self.assertEqual(self.reward(coverage=0.), 0.)
        self.assertEqual(self.reward(coverage=.5), .25)
        self.assertEqual(self.reward(), 1.5)
        self.assertEqual(self.reward(correct=False), .5)

    def test_equal_groups_have_no_learning_signal(self):
        self.assertEqual(group_advantages([0., 0., 0., 0.]), [0.] * 4)
        advantages = group_advantages([0., .25, .5, 1.5])
        self.assertLess(advantages[0], 0.)
        self.assertGreater(advantages[-1], 0.)
        self.assertAlmostEqual(sum(advantages), 0.)

    def test_local_errors_do_not_erase_task_success(self):
        answers = [{'score': {'correct': True}}]
        coverage = [{'coverage': 1., 'complete': True}]
        admission = {'proposed_facts': 10, 'accepted_facts': 9,
                     'proposed_relations': 2, 'accepted_relations': 1}
        result = extraction_reward(answers, coverage, admission)
        self.assertEqual(result['task_utility'], 1.5)
        self.assertAlmostEqual(result['source_error_rate'], 2 / 12)
        self.assertGreater(result['reward'], 1.)

    def test_formatting_or_source_quality_alone_is_not_task_signal(self):
        rows = [{'parsed': True, 'feedback': {'task_utility': .5}},
                {'parsed': True, 'feedback': {'task_utility': .5}},
                {'parsed': False, 'feedback': {'task_utility': 0.}}]
        self.assertFalse(has_task_signal(rows))
        rows[1]['feedback']['task_utility'] = 1.5
        self.assertTrue(has_task_signal(rows))


if __name__ == '__main__':
    unittest.main()
