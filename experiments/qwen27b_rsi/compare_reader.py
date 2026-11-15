"""Compare complete frozen Reader evaluations on the same graph and questions."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before_config = read(args.before / 'configuration.json')
    after_config = read(args.after / 'configuration.json')
    for field in ('policy_model', 'answer_model', 'decoding', 'max_actions',
                  'max_action_tokens', 'source_token_budget', 'reader_protocol',
                  'rerank_document_token_budget', 'native_budget'):
        assert before_config[field] == after_config[field], field
    assert read(args.before / 'graph_snapshot.json') == read(args.after / 'graph_snapshot.json')
    pairs = []
    summaries = {}
    trajectories = []
    for name, directory in [('before', args.before), ('after', args.after)]:
        completion = read(directory / 'complete.json')
        assert completion['worker_exit'] == 0
        rows = {p.stem: read(p) for p in (directory / 'trajectories').glob('*.json')}
        assert len(rows) == completion['queries'] == read(directory / 'configuration.json')['queries']
        trajectories.append(rows)
        summaries[name] = {
            'queries': len(rows),
            'correct': sum(row['result']['correct'] for row in rows.values()),
            'valid': sum(row['result']['valid'] for row in rows.values()),
            'model_tokens': sum(row['total_tokens'] for row in rows.values()),
            'rerank_native_input_tokens': sum(row['evidence']['rerank_native_input_tokens'] for row in rows.values()),
            'invalid_reasons': dict(Counter(row['result']['reason'] for row in rows.values()
                                            if not row['result']['valid']))}
    before, after = trajectories
    assert set(before) == set(after)
    for episode in sorted(before):
        old, new = before[episode], after[episode]
        assert old['raw_correct'] == new['raw_correct']
        assert old['native_budget'] == new['native_budget']
        pairs.append({
            'episode': episode,
            'before_correct': old['result']['correct'],
            'after_correct': new['result']['correct'],
            'raw_correct': new['raw_correct'],
            'actions_changed': old['evidence']['actions'] != new['evidence']['actions'],
            'model_token_change': new['total_tokens'] - old['total_tokens']})
    result = {
        'before': str(args.before), 'after': str(args.after), 'summaries': summaries,
        'gained': [p['episode'] for p in pairs if p['after_correct'] and not p['before_correct']],
        'lost': [p['episode'] for p in pairs if p['before_correct'] and not p['after_correct']],
        'wins_vs_raw': [p['episode'] for p in pairs if p['after_correct'] and not p['raw_correct']],
        'losses_vs_raw': [p['episode'] for p in pairs if p['raw_correct'] and not p['after_correct']],
        'net_correct_change': summaries['after']['correct'] - summaries['before']['correct'],
        'raw_correct': sum(p['raw_correct'] for p in pairs),
        'raw_model_tokens': sum(row['native_budget'] for row in after.values()),
        'over_budget': [episode for episode, row in after.items()
                        if row['total_tokens'] > row['native_budget']],
        'pairs': pairs,
        'scope': 'One greedy trajectory per question on an identical fixed graph; Writer and training costs separate'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'pairs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
