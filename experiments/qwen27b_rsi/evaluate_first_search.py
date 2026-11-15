"""Measure saved Reader searches with direct evidence delivery and fixed answering."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from graph_environment import GraphEnvironment, QueryBudgetExceeded, base, parse_action_text
from reader_environment import ReaderEpisode
from train_writer import Evaluator


def main():
    root = base.ROOT / 'results/qwen27b_rsi'
    reader = root / 'reader_factual_before_full'
    out = root / 'reader_first_search_full'
    assert not out.exists(), 'Inspect an existing run before resuming it'
    out.mkdir()
    assert base.read(reader / 'complete.json') == {'queries': 197, 'worker_exit': 0}
    state = base.read(reader / 'graph_snapshot.json')
    graphs = {p: GraphEnvironment(base.Patient(p, store_names=('chunk',)), state[str(p)]) for p in (1, 2)}
    evaluator = Evaluator(out / 'answers')
    base.write_json(out / 'graph_snapshot.json', state)
    base.write_json(out / 'configuration.json', {
        'process_id': os.getpid(), 'reader_run': str(reader), 'queries': 197,
        'reader': 'Saved first SEARCH; all returned facts and ranked source previews delivered as evidence',
        'answer_model': 'Qwen3.8-27B-FP8', 'answer_served_name': 'macaq',
        'context_cap': 8000, 'answer_max_tokens': 16384, 'concurrency': 2,
        'budget': 'Per-query RAW-RAG native model tokens, including the first policy decision',
        'scope': 'Intervention on saved search trajectories; no policy parameter changes or newly generated search decisions'})

    jobs = [(p, stage, q) for p, g in graphs.items()
            for stage, (_, qs) in enumerate(g.patient.groups) for q in qs]
    assert len(jobs) == 197

    def evaluate(job):
        persona, stage, question = job
        episode = f'p{persona}_{question["query_id"]}'
        graph = graphs[persona]
        trajectory = base.read(reader / 'trajectories' / f'{episode}.json')
        packet = trajectory['policy_packets'][0]
        row = packet['rows'][0]
        action = parse_action_text(row['text'])
        assert row['finished'] and action['operation'] == 'SEARCH'
        observation = json.loads(trajectory['messages'][3]['content'])['tool_observation']
        retrieval = trajectory['evidence']['retrieval_inputs'][0]
        assert action['query'] == retrieval['query']
        sources = [s['source_id'] for s in observation['sources']]
        fact_ids = list(observation['facts'])
        assert all(graph.sources[s]['stage'] <= stage for s in sources)
        assert all(graph.visible_facts(stage)[fid] == observation['facts'][fid] for fid in fact_ids)
        raw = next(r for r in base.read(base.RAW / f'persona_{persona:02d}' /
            f'session_{10 * (stage + 1):03d}_answers.json') if r['query_id'] == question['query_id'])
        context_cap = min(8000, len(base.ENC.encode(raw['formatted_context'])))
        policy_tokens = packet['prompt_tokens'] + len(row['token_ids'])
        budget = trajectory['native_budget']
        env = ReaderEpisode(graph, question, stage, context_cap=context_cap, native_budget=budget)
        env.selected_sources, env.selected_facts = sources, fact_ids
        env.actions = [action]
        env.rerank_native_tokens = retrieval['native_input_tokens']
        env.retrieval_inputs = [retrieval]
        env.rerank_tokens = sum(len(base.ENC.encode(graph.sources[s]['text']))
                                for s in retrieval['source_ids'])
        evidence = env.evidence()
        paired_path = reader / 'answers' / f'{episode}.json'
        paired = base.read(paired_path) if paired_path.exists() else None
        try:
            answer = evaluator.answer(graph, question, stage, episode, evidence=evidence,
                                      native_budget=budget - policy_tokens, paired_answer=paired)
        except QueryBudgetExceeded as error:
            result = {'valid': False, 'correct': False, 'reason': str(error)}
            answer_tokens = 0
        else:
            answer_tokens = answer['usage']['total_tokens']
            result = {'valid': True, 'correct': answer['score']['correct'],
                      'finish_reason': answer['finish_reason']}
        assert policy_tokens + answer_tokens <= budget
        record = {'episode': episode, 'result': result,
                  'prior_result': trajectory['result'], 'raw_correct': trajectory['raw_correct'],
                  'policy_tokens': policy_tokens, 'answer_tokens': answer_tokens,
                  'total_tokens': policy_tokens + answer_tokens, 'native_budget': budget,
                  'policy_packet': packet, 'evidence': evidence,
                  'source_selection': 'All first-search candidates in their saved order, subject to the common context cap'}
        base.write_json(out / 'trajectories' / f'{episode}.json', record)
        print(episode, result, 'prior', trajectory['result'], flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate, jobs))
    summary = {
        'queries': len(rows), 'correct': sum(r['result']['correct'] for r in rows),
        'invalid': sum(not r['result']['valid'] for r in rows),
        'prior_correct': sum(r['prior_result']['correct'] for r in rows),
        'raw_correct': sum(r['raw_correct'] for r in rows),
        'total_model_tokens': sum(r['total_tokens'] for r in rows),
        'policy_tokens': sum(r['policy_tokens'] for r in rows),
        'answer_tokens': sum(r['answer_tokens'] for r in rows),
        'raw_model_tokens': sum(r['native_budget'] for r in rows),
        'rerank_native_input_tokens': sum(r['evidence']['rerank_native_input_tokens'] for r in rows),
        'gained': [r['episode'] for r in rows if r['result']['correct'] and not r['prior_result']['correct']],
        'lost': [r['episode'] for r in rows if not r['result']['correct'] and r['prior_result']['correct']],
        'recovered_prior_invalid': [r['episode'] for r in rows if r['result']['correct'] and not r['prior_result']['valid']],
        'scope': 'One saved SEARCH followed by fixed answering; graph construction and training costs are separate; not a policy-learning result'}
    base.write_json(out / 'summary.json', summary)
    base.write_json(out / 'complete.json', {'queries': len(rows)})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
