"""Compare all frozen Writer samples under the same Reader and input states."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    return {
        'samples': len(rows),
        'supported': sum(r['supported'] for r in rows),
        'invalid_graph_operations': sum(r['action'] is None for r in rows),
        'source_unsupported_operations': sum(r['action'] is not None and not r['supported'] for r in rows),
        'noop': sum(r['action'] is not None and r['action']['operation'] == 'NOOP' for r in rows),
        'qa_improving_writes': sum(r['supported'] and sum(r['query_deltas']) > 0 for r in rows),
        'qa_regressing_writes': sum(r['supported'] and sum(r['query_deltas']) < 0 for r in rows),
        'writes_with_any_query_regression': sum(r['supported'] and any(d < 0 for d in r['query_deltas']) for r in rows),
        'correctness_preserving_cost_savings': sum(r['supported'] and all(d == 0 for d in r['query_deltas'])
                                                  and r['cost_term'] > 0 for r in rows),
        'mean_net_correct': sum(sum(r['query_deltas']) for r in rows) / len(rows),
        'mean_reward': sum(r['paired_reward'] for r in rows) / len(rows),
        'mean_writer_tokens': sum(r['writer_tokens'] for r in rows) / len(rows),
        'mean_online_tokens_including_marginal_write': sum(r['reader_costs']['after_including_marginal_write'] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    configs = [read(p / 'configuration.json') for p in (args.before, args.after)]
    for key in ('training_run', 'episodes', 'samples_per_state', 'seed', 'reader_checkpoint',
                'reader_protocol', 'answer_model', 'max_action_tokens'):
        assert configs[0][key] == configs[1][key], f'Unpaired configuration: {key}'
    summaries = []
    for path, config in zip((args.before, args.after), configs):
        assert read(path / 'complete.json')['worker_exit'] == 0
        summary = read(path / 'summary.json')
        assert len(summary['results']) == len(config['episodes']) * config['samples_per_state']
        summaries.append(summary)
    comparisons = []
    for episode in configs[0]['episodes']:
        before, after = [sorted((r for r in s['results'] if r['episode'] == episode),
                                key=lambda r: r['sample_index']) for s in summaries]
        assert [r['sample_index'] for r in before] == [r['sample_index'] for r in after]
        a, b = metrics(before), metrics(after)
        comparisons.append({'episode': episode, 'before': a, 'after': b,
                            'mean_reward_change': b['mean_reward'] - a['mean_reward'],
                            'mean_net_correct_change': b['mean_net_correct'] - a['mean_net_correct'],
                            'changed_actions': sum(x['action'] != y['action'] for x, y in zip(before, after))})
    result = {'before': str(args.before), 'after': str(args.after),
              'samples_per_state': configs[0]['samples_per_state'], 'states': comparisons,
              'mean_net_correct_change_per_state_set': sum(r['mean_net_correct_change'] for r in comparisons),
              'scope': 'Descriptive paired sample comparison; all samples included; no best-of-N selection'}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
