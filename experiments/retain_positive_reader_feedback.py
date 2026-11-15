"""Retain successful graph evidence selections from the adaptation feedback pool."""
import argparse
import re

from qfad_fact_graph import base
from rsi_reader_policy import ReaderPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-round', type=base.Path, required=True)
    parser.add_argument('--evaluation', type=base.Path, required=True)
    args = parser.parse_args()
    complete = base.read(args.reader_round / 'complete.json')
    assert base.read(args.evaluation / 'summary.json')['n'] == 197
    policy = ReaderPolicy()
    policy.load(complete['policy'])
    feedback = []
    for path in sorted(args.evaluation.glob('p*/*.json')):
        if path.name.endswith('.answer.json'):
            continue
        row = base.read(path)
        feedback.append((f'p{row["persona"]}_{row["query_id"]}', path, row))
    for path in sorted(args.reader_round.glob('*/old_graph/baseline.json')):
        feedback.append((path.parent.parent.name, path, base.read(path)))
    for path in sorted(args.reader_round.parent.glob('reader_length_probe/*/updated.json')):
        feedback.append((path.parent.name, path, base.read(path)))
    retained = []
    for episode, path, row in feedback:
        if not row['semantic_score']['correct']:
            continue
        context = row['context']
        assert context.startswith('Memory 1:\n')
        documents = re.split(r'\n\nMemory \d+:\n', context[len('Memory 1:\n'):])
        assert context == '\n\n'.join(f'Memory {i+1}:\n{doc}' for i, doc in enumerate(documents))
        count = len(row['retrieved_facts']) if 'retrieved_facts' in row else len(row['fact_indices'])
        assert len(documents) == count <= 15
        likelihood = row.get('query_likelihood', {}).get('token_probability', 0.)
        policy.remember(episode, documents, 1. + .2 * likelihood)
        retained.append({'episode_id': episode, 'feedback_file': str(path), 'documents': count})
    output = args.reader_round / 'policy_with_feedback.pt'
    policy.save(output)
    base.write_json(args.reader_round / 'retained_feedback.json', {'positive_records': retained,
        'paths': len(policy.paths), 'answer_text_stored_in_policy': False,
        'evaluation_role': 'same-patient adaptation feedback before fresh frozen-state evaluation'})
    base.write_json(args.reader_round / 'ready.json', {**complete, 'policy': str(output),
        'retained_paths': len(policy.paths)})
    print('Retained graph evidence paths', len(policy.paths), flush=True)


if __name__ == '__main__':
    main()
