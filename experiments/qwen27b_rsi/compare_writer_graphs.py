"""Compare frozen Writer-generated graphs with fixed QFAD retrieval and answering."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def compare(before, after, writer_before, writer_after):
    configurations = [read(p / 'configuration.json') for p in (before, after)]
    for key in ('reader', 'answer_model', 'answer_served_name', 'context_cap', 'answer_max_tokens'):
        assert configurations[0][key] == configurations[1][key], key
    generators = [read(p / 'configuration.json') for p in (writer_before, writer_after)]
    assert generators[0].get('feedback_protocol') == generators[1].get('feedback_protocol')
    assert generators[0].get('current_reader_feedback') == generators[1].get('current_reader_feedback')
    for key in ('episodes', 'reader_run', 'policy_model', 'decoding', 'max_action_tokens',
                'max_observed_sources', 'max_policy_input_tokens', 'graph_acceptance', 'feedback_origin',
                'feedback_model_tokens', 'feedback_rerank_native_input_tokens'):
        assert generators[0][key] == generators[1][key], key
    assert read(writer_before / 'initial_graph.json') == read(writer_after / 'initial_graph.json')
    seed_cost = read(Path(__file__).resolve().parents[3] / 'results/qwen27b_rsi/seed_graph_cost.json')
    assert read(Path(seed_cost['graph_path'])) == read(writer_before / 'initial_graph.json')
    rows, totals = [], {}
    for label, evaluation, generation in [('before', before, writer_before),
                                           ('after', after, writer_after)]:
        assert read(evaluation / 'summary.json')['queries'] == 197
        assert read(evaluation / 'configuration.json')['queries'] == 197
        generated = read(generation / 'complete.json')
        assert generated['worker_exit'] == 0
        assert generated['episodes'] == len(generators[0]['episodes'])
        graph = read(generation / 'graph_state.json')
        assert graph == read(evaluation / 'graph_snapshot.json')
        data = {p.stem: read(p) for p in (evaluation / 'answers').glob('*.json')}
        assert len(data) == 197
        rows.append(data)
        summary = read(generation / 'summary.json')
        config = read(generation / 'configuration.json')
        feedback_judge_tokens = 0
        for episode in config['episodes']:
            answer_path = Path(config['reader_run']) / 'answers' / f'{episode}.json'
            if answer_path.exists():
                feedback_judge_tokens += read(answer_path)['judge_usage']['total_tokens']
        if config.get('current_reader_feedback'):
            current = [read(generation / 'current_answers' / f'{e}.json') for e in config['episodes']]
            assert config['current_reader_model_tokens'] == sum(r['usage']['total_tokens'] for r in current)
            assert config['current_reader_judge_tokens'] == sum(r['judge_usage']['total_tokens'] for r in current)
            assert config['current_reader_rerank_native_input_tokens'] == sum(
                r['retrieval']['rerank_native_input_tokens'] for r in current)
        feedback_model_tokens = config['feedback_model_tokens'] + config.get('current_reader_model_tokens', 0)
        feedback_judge_tokens += config.get('current_reader_judge_tokens', 0)
        online = sum(r['usage']['total_tokens'] for r in data.values())
        totals[label] = {
            'correct': sum(r['score']['correct'] for r in data.values()),
            'online_model_tokens': online,
            'writer_model_tokens': summary['writer_tokens'],
            'feedback_model_tokens': feedback_model_tokens,
            'saved_reader_feedback_model_tokens': config['feedback_model_tokens'],
            'current_reader_feedback_model_tokens': config.get('current_reader_model_tokens', 0),
            'feedback_judge_tokens': feedback_judge_tokens,
            'writer_source_judge_tokens': sum(
                r['support'].get('usage', {}).get('total_tokens', 0)
                + r['support'].get('fact_support', {}).get('usage', {}).get('total_tokens', 0)
                for r in summary['results']),
            'online_plus_feedback_and_write_tokens': online + summary['writer_tokens']
                                                      + feedback_model_tokens,
            'qwen_tokens_including_seed': online + summary['writer_tokens']
                                          + feedback_model_tokens + seed_cost['writer_model_tokens'],
            'online_rerank_native_input_tokens': sum(
                r['retrieval']['rerank_native_input_tokens'] for r in data.values()),
            'feedback_rerank_native_input_tokens': config['feedback_rerank_native_input_tokens']
                                                   + config.get('current_reader_rerank_native_input_tokens', 0),
            'facts': sum(len(p['facts']) for p in graph.values()),
            'relations': sum(len(p['relations']) for p in graph.values()),
            'supported_writes': summary['changed'],
            'unsupported_or_invalid_writes': summary['unsupported_or_invalid']}
    old, new = rows
    assert set(old) == set(new)
    results_root = Path(__file__).resolve().parents[3] / 'results/qfad_query_writer_rl_20261113'
    raw_usage = read(results_root / 'raw_usage.json')
    pairs = []
    for episode in sorted(old):
        a, b = old[episode], new[episode]
        persona, query_id = episode.split('_', 1)
        raw_correct = read(results_root / 'semantic/evaluation' / persona / f'{query_id}_raw.json')['correct']
        pairs.append({'episode': episode, 'before_correct': a['score']['correct'],
                      'after_correct': b['score']['correct'], 'raw_correct': raw_correct,
                      'raw_model_tokens': raw_usage[persona[1:]][query_id]['total_tokens']})
    return {
        'queries': 197, 'reader': configurations[0]['reader'],
        'before': str(before), 'after': str(after), 'totals': totals,
        'writer_before': str(writer_before), 'writer_after': str(writer_after),
        'common_seed_cost': seed_cost,
        'raw_correct': sum(p['raw_correct'] for p in pairs),
        'raw_model_tokens': sum(p['raw_model_tokens'] for p in pairs),
        'gained': [p['episode'] for p in pairs if p['after_correct'] and not p['before_correct']],
        'lost': [p['episode'] for p in pairs if p['before_correct'] and not p['after_correct']],
        'wins_vs_raw': [p['episode'] for p in pairs if p['after_correct'] and not p['raw_correct']],
        'losses_vs_raw': [p['episode'] for p in pairs if p['raw_correct'] and not p['after_correct']],
        'pairs': pairs,
        'scope': 'Greedy Writer generations with identical initial graph and Reader feedback; '
                 'fixed Reader and answerer for all 197 questions',
        'cost_scope': 'Online decisions and answers, selected feedback acquisition, and Writer generation; '
                      'feedback scoring and source-validation judge tokens are reported separately; '
                      'common seed construction is included in qwen_tokens_including_seed; '
                      'embedding costs and source-annotation labor are separate; '
                      'training and experimental evaluation costs are separate'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('before', 'after', 'writer-before', 'writer-after', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after, args.writer_before, args.writer_after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
