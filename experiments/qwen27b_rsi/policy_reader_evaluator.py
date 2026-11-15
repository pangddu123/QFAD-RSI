"""Evaluate graph writes using a fixed Reader-policy snapshot and answerer."""
import json
import threading

from graph_environment import QueryBudgetExceeded, base
from reader_environment import ReaderEpisode


class PolicyReaderEvaluator:
    def __init__(self, answerer, request):
        self.answerer = answerer
        self.judge = answerer.judge
        self.request = request
        self.cache = {}
        self.lock = threading.Lock()
        self.raw_usage = base.read(base.OUT / 'raw_usage.json')

    def answer(self, graph, question, stage, label, paired_answer=None):
        key = (graph.patient.persona, question['query_id'], stage, json.dumps(graph.state, sort_keys=True))
        # One policy worker serves the candidate comparisons; identical states share a realization.
        with self.lock:
            if key in self.cache:
                result = {**self.cache[key], 'reader_cache_hit': True}
                base.write_json(self.answerer.directory / f'{label}.json', result)
                return result
            persona, qid = graph.patient.persona, question['query_id']
            raw = next(r for r in base.read(base.RAW / f'persona_{persona:02d}' /
                f'session_{10 * (stage + 1):03d}_answers.json') if r['query_id'] == qid)
            budget = self.raw_usage[str(persona)][qid]['total_tokens']
            cap = min(8000, len(base.ENC.encode(raw['formatted_context'])))
            episode = ReaderEpisode(graph, question, stage, cap, native_budget=budget)
            packets, tokens, failure = [], 0, None
            for index in range(4):
                packet = self.request({'operation': 'infer_reader', 'id': f'{label}_reader{index}',
                    'messages': episode.messages, 'seed': 0, 'max_tokens': 192,
                    'native_budget': budget - tokens})
                packets.append(packet)
                if packet['event'] == 'budget_exhausted':
                    failure = 'Reader decision exceeds remaining budget or input limit'
                    break
                row = packet['rows'][0]
                tokens += packet['prompt_tokens'] + len(row['token_ids'])
                try:
                    if tokens >= budget:
                        raise ValueError('Reader exhausted the native query budget')
                    episode.step(row, policy_tokens=tokens)
                except (ValueError, KeyError, TypeError) as error:
                    failure = str(error)
                    break
                if episode.terminal:
                    break
            evidence = episode.evidence()
            if failure is None:
                try:
                    result = self.answerer.answer(graph, question, stage, label, evidence=evidence,
                                                  native_budget=budget - tokens, paired_answer=paired_answer)
                except QueryBudgetExceeded as error:
                    failure = str(error)
            if failure is not None:
                result = {'query_id': qid, 'answer': None, 'answer_generated': False,
                          'score': {'correct': False, 'reason': failure, 'basis': 'Reader environment failure'},
                          'retrieval': evidence,
                          'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
            result = {**result, 'reader_valid': failure is None, 'reader_cache_hit': False,
                      'reader_policy_tokens': tokens,
                      'online_model_tokens': tokens + result['usage']['total_tokens'],
                      'native_budget': budget, 'reader_packets': packets, 'reader_messages': episode.messages}
            assert result['online_model_tokens'] <= budget
            base.write_json(self.answerer.directory / f'{label}.json', result)
            self.cache[key] = result
            return result
