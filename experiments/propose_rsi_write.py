"""Materialize supported facts from a rewarded Writer action for Reader evaluation."""
import argparse
from pathlib import Path

from qfad_fact_graph import base
from fact_writer_protocol import parse_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rollout = base.read(args.run / f'rollouts_{args.step}.json')
    rewards = base.read(args.run / f'rewards_{args.step}.json')
    request = next(r for r in base.read(args.run / 'writer_inputs.json') if r['episode_id'] == rollout['episode_id'])
    eligible = [i for i, row in enumerate(rewards['samples']) if row['valid']]
    assert eligible, 'No structurally valid Writer action in this group'
    selected = max(eligible, key=lambda i: rewards['rewards'][i])
    directory = args.run / f'pilot/step{args.step}_sample{selected}'
    prediction = base.read(directory / 'writer_output.json')
    writes = parse_output(prediction['text'], request['sources'])
    supported = base.read(directory / 'source_support.json')['supported']
    assert sum(len(row['facts']) for row in writes) == len(supported)
    offset = 0
    proposed = []
    for row in writes:
        flags = supported[offset:offset + len(row['facts'])]
        offset += len(row['facts'])
        facts = [fact for fact, keep in zip(row['facts'], flags) if keep]
        if facts:
            proposed.append({'source_id': row['source_id'], 'facts': facts})
    assert proposed, 'No source-supported facts remain in the selected Writer action'
    base.write_json(args.output / 'proposed_writes.json', proposed)
    base.write_json(args.output / 'proposal.json', {'episode_id': rollout['episode_id'],
        'step': args.step, 'sample': selected, 'status': 'awaiting query evaluation',
        'candidate_reward': rewards['rewards'][selected],
        'candidate_query_quality_change': rewards['query_reward_components'][selected],
        'source_support_file': str(directory / 'source_support.json'),
        'candidate_fact_count': len(supported), 'proposed_fact_count': sum(len(r['facts']) for r in proposed),
        'generation': 'sampled Writer action', 'unsupported_facts_excluded': True})
    print('Proposed supported facts', sum(len(r['facts']) for r in proposed), flush=True)


if __name__ == '__main__':
    main()
