"""Account for graph construction, query feedback and final retrieval separately."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def embedding_usage(run, results_root):
    pid = read(run / 'configuration.json')['process_id']
    path = results_root / 'embedding_usage' / f'{pid}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {'process_id': pid, 'requests': len(rows),
            'native_tokens': sum(row['native_usage']['total_tokens'] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    comparison = read(args.comparison)
    assert comparison['queries'] == 197
    results_root = Path(__file__).resolve().parents[3] / 'results/qwen27b_rsi'
    indexing = read(results_root / 'retrieval_budget.json')['patients']
    raw_index = sum(p['raw_final_index_once_native_tokens'] for p in indexing.values())
    graph_index = sum(p['graph_source_index_native_tokens'] for p in indexing.values())
    raw_query = sum(sum(p['raw_query_embedding_native_tokens'].values()) for p in indexing.values())
    raw_rerank = sum(p['raw_rerank_native_input_tokens'] for p in indexing.values())
    conditions = {}
    for label in ('before', 'after'):
        generation = Path(comparison['writer_' + label])
        evaluation = Path(comparison[label])
        feedback = Path(read(generation / 'configuration.json')['reader_run'])
        assert read(generation / 'complete.json')['worker_exit'] == 0
        assert read(evaluation / 'summary.json')['queries'] == 197
        uses = {name: embedding_usage(run, results_root) for name, run in
                [('writer_generation', generation), ('final_queries', evaluation), ('feedback_full_run', feedback)]}
        totals = comparison['totals'][label]
        source_judge = totals['writer_source_judge_tokens'] + comparison['common_seed_cost']['source_judge_tokens']
        feedback_judge = totals['feedback_judge_tokens']
        qwen = totals['qwen_tokens_including_seed']
        rerank = totals['online_rerank_native_input_tokens'] + totals['feedback_rerank_native_input_tokens']
        embeddings = graph_index + sum(u['native_tokens'] for u in uses.values())
        conditions[label] = {
            'correct': totals['correct'],
            'qwen27b_native_tokens': qwen,
            'sol_source_verification_tokens': source_judge,
            'sol_feedback_scoring_tokens': feedback_judge,
            'total_llm_native_token_count': qwen + source_judge + feedback_judge,
            'embedding_index_once_native_tokens': graph_index,
            'embedding_call_records': uses,
            'embedding_total_native_tokens_upper_bound': embeddings,
            'rerank_total_native_input_tokens': rerank,
            'within_raw_qwen_token_count': qwen <= comparison['raw_model_tokens'],
            'within_raw_total_llm_token_count': qwen + source_judge + feedback_judge <= comparison['raw_model_tokens'],
            'within_raw_embedding_token_count': embeddings <= raw_index + raw_query,
            'within_raw_rerank_token_count': rerank <= raw_rerank}
    report = {
        'queries': 197,
        'raw': {'correct': comparison['raw_correct'], 'qwen27b_native_tokens': comparison['raw_model_tokens'],
                'embedding_index_once_native_tokens': raw_index, 'embedding_query_native_tokens': raw_query,
                'embedding_total_native_tokens': raw_index + raw_query,
                'rerank_total_native_input_tokens': raw_rerank},
        'conditions': conditions,
        'common_seed': comparison['common_seed_cost'],
        'scope': 'Final queries plus selected Reader feedback acquisition, greedy Writer generation, '
                 'seed construction, and source/feedback judging. Policy training and experimental accuracy scoring are separate.',
        'embedding_feedback_bound': 'All embedding calls from the complete 197-question feedback run are charged; '
                                    'only the selected feedback episodes are needed for generation.',
        'token_count_interpretation': 'Counts use each running model tokenizer. The aggregate LLM token count '
                                      'is an accounting measure, not a FLOPs, latency, or monetary-cost estimate.',
        'paired_reuse_accounting': 'Identical answers and source judgments share one realization. '
                                   'Original token usage is charged to each condition as a standalone execution estimate.',
        'source_annotation_labor': 'Provided training annotations for the shared seed; labor was not measured.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
