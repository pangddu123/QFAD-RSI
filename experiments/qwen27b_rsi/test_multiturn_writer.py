"""Check retrospective credit, branch-specific observations and the worker payload."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from policy_worker import encode_guided_turns
from targeted_guidance import attribute_writes, render_hint
from train_extractor import train_window


def evidence(lines=(), covered=False, skipped=()):
    verdict = {'complete': covered, 'coverage': float(covered), 'bundles': [{'conditions': [
        {'covered': covered, 'source_ids': ['a'] if covered else []}]}]}
    trace = {'source_windows': [{'source_id': 'a', 'line_ids': list(lines)}] if lines else [],
             'context_skipped_source_ids': list(skipped)}
    return verdict, trace


class CreditTests(unittest.TestCase):
    query = {'stage': 0, 'query_id': 'q', 'requirements': {'bundles': [{'conditions': [{
        'criterion': 'baseline_measurement', 'description': 'SECRET ANNOTATION',
        'anchors': [{'source_id': 'a', 'span_id': 1, 'quote': 'initial result'}]}]}]}}
    turns = [{'parsed': True, 'sources': {s: {'text': 'visit\ninitial result'}},
              'admission': {'rejected': []}} for s in ('a', 'b', 'c')]

    def events(self, prefixes):
        return attribute_writes(self.turns, [self.query], [prefixes])

    def test_missed_at_original_write(self):
        events = self.events([evidence(), evidence([0]), evidence([0]), evidence([0])])
        self.assertEqual([len(e) for e in events], [1, 0, 0])
        hint = render_hint(events[0])
        self.assertIn('lines 1–1', hint)
        self.assertNotIn('SECRET ANNOTATION', hint)

    def test_only_last_loss_is_credited_after_recovery(self):
        events = self.events([evidence([0, 1], True), evidence(), evidence([0, 1], True), evidence([0])])
        self.assertEqual([len(e) for e in events], [0, 0, 1])
        self.assertEqual(events[2][0]['lost_lines'], {'a': [1]})

    def test_do_not_blame_writer_for_reader_or_budget(self):
        for final in [evidence([0, 1]), evidence([], skipped=['a']), evidence([1], True)]:
            self.assertEqual(self.events([evidence(), evidence([0, 1], True), final, final]), [[], [], []])

    def test_worker_rejects_hint_for_another_branch(self):
        paths = [[{'group': 't0', 'row': 0}, {'group': 't1', 'row': 0}]]
        msgs = lambda s: [{'role': 'user', 'content': s}]
        encode = lambda m: list(m[-1]['content'])
        groups = {'t0': (list('old'), [], 0), 't1': (list('next'), [], 0)}
        item = {'group': 't0', 'row': 0, 'messages': msgs('old'), 'hint': 'use record date'}
        teachers = encode_guided_turns(paths, [item], groups, encode)
        self.assertEqual(set(teachers), {('t0', 0)})
        with self.assertRaises(ValueError):
            encode_guided_turns(paths, [{**item, 'messages': msgs('next')}], groups, encode)

    def test_controller_preserves_each_branch_until_terminal_update(self):
        class Graph:
            state = {'facts': {}, 'relations': []}
            sources = {s: {'text': 'visit\ninitial result', 'stage': 0, 'source_id': s} for s in ('a', 'b')}
            patient = SimpleNamespace(groups=[(None, [{'query_id': 'q', 'question': 'initial value?'}])])
            def __init__(self, writes=()):
                self.writes = writes
            def visible_facts(self, stage):
                return {s: {'sample': row} for s, row in self.writes}
            def retrieve(self, question, stage, context_cap):
                lines = [0, 1] if ('a', 0) in self.writes else [0] if self.writes else []
                return evidence(lines)[1]
        def apply(row, graph, sources, existing, judge):
            return {'parsed': True, 'admission': {'accepted_facts': 1, 'proposed_facts': 1,
                'accepted_relations': 0, 'proposed_relations': 0, 'rejected': []}}, Graph(
                    graph.writes + ((sources[0]['source_id'], row['index']),))
        evaluator = SimpleNamespace(context_cap=lambda *args: 100,
            answer=lambda g, q, s, name: {'score': {'correct': ('a', 0) in g.writes},
                'retrieval': g.retrieve(q, s, 100)})
        scorer = SimpleNamespace(score=lambda q, req, trace: evidence(covered=any(
            1 in w['line_ids'] for w in trace['source_windows']))[0])
        packets = []
        def request(p):
            packets.append(p)
            if p['operation'] == 'sample':
                start = int(p['id'].rsplit('b', 1)[1]) if '_t0_' in p['id'] else 0
                return {'event': 'samples', 'policy_step': 0, 'rows': [
                    {'index': start+i, 'text': str(start+i)} for i in range(p['count'])]}
            return {'event': 'updated', 'step': 1, 'adapter_delta_l2': 1.,
                    'guided_turns': len(p['turn_guidance']), 'targeted_reverse_kl': .2}
        window = [{'patient': 1, 'id': s, 'source_ids': [s], 'queries': [self.query]} for s in ('a', 'b')]
        for group_size in (4, 9):
            with self.subTest(group_size=group_size), TemporaryDirectory() as tmp, patch(
                    'train_extractor.apply_sample', side_effect=apply):
                packets.clear()
                report, state = train_window(window, Graph(), 'g', 17, request, evaluator, scorer, None,
                                             Path(tmp), .1, group_size=group_size)
                credit = json.loads((Path(tmp) / 'g_b1.credit.json').read_text())
                self.assertEqual(state.writes, (('a', 0), ('b', 0)))
                first = [p for p in packets if '_t0_' in p['id']]
                self.assertEqual([p['count'] for p in first], [4] if group_size == 4 else [4, 4, 1])
                follow = [p for p in packets if '_t1_' in p['id']]
                self.assertEqual(len(follow), group_size)
                self.assertEqual(len({p['seed'] for p in first + follow}), len(first + follow))
                for b, packet in enumerate(follow):
                    observation = json.loads(packet['messages'][-1]['content'])
                    self.assertEqual(observation['existing_facts'], {'a': {'sample': b}})
                    self.assertNotIn('SECRET ANNOTATION', str(packet['messages']))
                update = packets[-1]
                self.assertEqual([p['operation'] for p in packets].count('update'), 1)
                self.assertEqual([len(path) for path in update['trajectories']], [2] * group_size)
                self.assertEqual([(r['group'], r['row']) for r in update['turn_guidance']],
                                 [(f'g_t0_b{4*(b//4)}', b%4) for b in range(1, group_size)])
                self.assertEqual(report['distinct_output_trajectories'], group_size)
                self.assertEqual(report['fully_parsed_trajectories'], group_size)
                self.assertEqual([bool(h) for h in credit['hints_by_turn']], [True, False])
