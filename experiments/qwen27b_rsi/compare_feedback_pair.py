"""Summarize the full-dataset frozen-Writer feedback-content comparison."""
import json
from pathlib import Path

from summarize_writer_budget import embedding_usage


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path(__file__).resolve().parents[3] / 'results/qwen27b_rsi'
    seed = read(root / 'seed_graph_cost.json')
    indexing = read(root / 'retrieval_budget.json')['patients']
    raw_budget = read(root / 'writer_factual_before_budget.json')
    generations = {v: root / f'writer_feedback_fp8_{v}' for v in ('original', 'answer')}
    configs = {v: read(p / 'configuration.json') for v, p in generations.items()}
    for key in ('episodes', 'reader_run', 'policy_model', 'decoding', 'max_action_tokens',
                'max_policy_input_tokens', 'source_selection', 'graph_acceptance',
                'feedback_model_tokens', 'feedback_rerank_native_input_tokens'):
        assert configs['original'][key] == configs['answer'][key], key
    assert not configs['original']['include_reader_generated_answer']
    assert configs['answer']['include_reader_generated_answer']
    assert read(generations['original'] / 'initial_graph.json') == read(generations['answer'] / 'initial_graph.json')
    for index in range(len(configs['original']['episodes'])):
        a, b = [read(generations[v] / f'step{index}.result.json') for v in ('original', 'answer')]
        assert a['episode'] == b['episode'] and a['observed_source_ids'] == b['observed_source_ids']
    feedback = Path(configs['original']['reader_run'])
    feedback_judge = sum(read(feedback / 'answers' / f'{e}.json')['judge_usage']['total_tokens']
                         for e in configs['original']['episodes'] if (feedback / 'answers' / f'{e}.json').exists())
    conditions, rows = {}, {}
    evaluation_configs = []
    summaries = {v: read(p / 'summary.json') for v, p in generations.items()}
    paired_writer_tokens = sum(s['writer_tokens'] for s in summaries.values())
    paired_source_tokens = sum(
        d['support'].get('usage', {}).get('total_tokens', 0)
        + d['support'].get('fact_support', {}).get('usage', {}).get('total_tokens', 0)
        for s in summaries.values() for d in s['results'])
    for variant, generation in generations.items():
        evaluation = root / f'writer_feedback_fp8_{variant}_full'
        cfg = configs[variant]
        assert read(generation / 'complete.json') == {'episodes': 21, 'api_generation_complete': True}
        assert read(evaluation / 'summary.json')['queries'] == 197
        assert read(generation / 'graph_state.json') == read(evaluation / 'graph_snapshot.json')
        evaluation_configs.append(read(evaluation / 'configuration.json'))
        data = {p.stem: read(p) for p in (evaluation / 'answers').glob('*.json')}
        assert len(data) == 197
        rows[variant] = data
        summary = summaries[variant]
        online = sum(d['usage']['total_tokens'] for d in data.values())
        qwen = online + paired_writer_tokens + cfg['feedback_model_tokens'] + seed['writer_model_tokens']
        source_judge = seed['source_judge_tokens'] + paired_source_tokens
        embedding_calls = {name: embedding_usage(path, root) for name, path in
                           [('paired_generation', generation), ('final_queries', evaluation), ('feedback_full_run', feedback)]}
        embedding_tokens = sum(p['graph_source_index_native_tokens'] for p in indexing.values()) \
            + sum(x['native_tokens'] for x in embedding_calls.values())
        conditions[variant] = {
            'correct': sum(d['score']['correct'] for d in data.values()),
            'supported_writes': summary['changed'], 'unsupported_or_invalid': summary['unsupported_or_invalid'],
            'answer_tokens': online, 'writer_tokens': summary['writer_tokens'],
            'paired_writer_tokens_charged': paired_writer_tokens,
            'feedback_tokens': cfg['feedback_model_tokens'], 'qwen27b_tokens_including_seed': qwen,
            'sol_source_tokens_including_seed': source_judge, 'sol_feedback_tokens': feedback_judge,
            'all_llm_native_tokens': qwen + source_judge + feedback_judge,
            'embedding_tokens_upper_bound': embedding_tokens, 'embedding_call_records': embedding_calls,
            'rerank_tokens': sum(d['retrieval']['rerank_native_input_tokens'] for d in data.values())
                             + cfg['feedback_rerank_native_input_tokens']}
    for key in ('reader', 'answer_model', 'answer_served_name', 'context_cap', 'answer_max_tokens'):
        assert evaluation_configs[0][key] == evaluation_configs[1][key], key
    a, b = rows['original'], rows['answer']
    assert set(a) == set(b)
    report = {'queries': 197, 'conditions': conditions,
        'raw': {'correct': raw_budget['raw_correct'], 'qwen27b_tokens': raw_budget['raw_qwen27b_tokens'],
                'embedding_tokens': raw_budget['raw_embedding_total_tokens'],
                'rerank_tokens': raw_budget['raw_rerank_total_tokens']},
        'gained': [k for k in sorted(a) if b[k]['score']['correct'] and not a[k]['score']['correct']],
        'lost': [k for k in sorted(a) if a[k]['score']['correct'] and not b[k]['score']['correct']],
        'scope': 'Frozen FP8 Writer; matched initial graph and source pools; Reader-generated answer added only in the answer condition. This is a feedback-content control, not an RL comparison.',
        'budget_scope': 'Final answers, selected feedback acquisition, common seed and source/feedback judging. Both Writer generations and their source judgments are charged to each condition because matched source trimming consults both graph states. Experimental accuracy scoring and training are separate; seed annotation labor is unmeasured.',
        'embedding_upper_bound': 'Both paired generations share one process; all generation embedding calls are charged to each condition. All 197 feedback embedding calls are also charged.',
        'token_interpretation': 'Native token counts are not FLOPs, latency or money.'}
    (root / 'writer_feedback_fp8_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
