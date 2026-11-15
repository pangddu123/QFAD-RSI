"""Regression checks for the distinction between session labels and original line citations."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_curriculum import patient_episodes
from train_extractor import episode_windows


class CurriculumTests(unittest.TestCase):
    def test_session_label_does_not_create_a_line_anchor(self):
        def session(sid, text, noise=False):
            return SimpleNamespace(session_id=sid, texts=[text], is_noise=noise)

        def question(qid, points):
            return SimpleNamespace(query_id=qid, answers_data=[{'is_correct': True, 'content': 'yes'}],
                source_key_points=[{'session_id': sid, 'content': point} for sid, point in points])

        units = [SimpleNamespace(sessions_to_inject=[session(1, 'day\nA was stopped.'), session(7, 'noise', True)],
                    queries_to_evaluate=[question('paraphrase', [(1, 'Stopped treatment A')]),
                                         question('literal', [(1, 'A was stopped.')]),
                                         question('future', [(1, 'A was stopped.'), (2, 'future note')]),
                                         question('noise', [(7, 'noise')])]),
                 SimpleNamespace(sessions_to_inject=[session(2, 'future note')], queries_to_evaluate=[])]
        dataset = SimpleNamespace(load=lambda: None, get_evaluation_units=lambda: units)
        frame = SimpleNamespace(content=['day\nA was stopped.', 'future note'], hash_id=['s1', 's2'])
        with patch('benchmark_curriculum.MedMemoryBenchDataset', return_value=dataset), \
             patch('benchmark_curriculum.pd.read_parquet', return_value=frame), \
             patch('benchmark_curriculum.base.harness.session_passages', side_effect=lambda s, enc: s.texts):
            episodes, audit = patient_episodes(5, None, None)
        self.assertEqual(audit, {'total': 4, 'excluded': {'missing_or_future_source_session': 2}})
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]['source_ids'], ['s1'])
        weak, literal = [q['requirements']['bundles'][0]['conditions'][0] for q in episodes[0]['queries']]
        self.assertEqual(weak['session_source_ids'], ['s1'])
        self.assertEqual(weak['anchors'], [])
        self.assertEqual(literal['anchors'], [{'source_id': 's1', 'span_id': 1, 'quote': 'A was stopped.'}])

    def test_window_ends_before_a_write_after_the_query_checkpoint(self):
        episodes = [dict(patient=5, source_stage=s, queries=[{'stage': s}]) for s in [0, 0, 1]]
        self.assertEqual([len(w) for w in episode_windows(episodes, 3)], [2, 1])
