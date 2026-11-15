"""Learn Reader paths on supported Writer proposals and retain query improvements."""
import argparse
import subprocess
import sys
from pathlib import Path

from qfad_fact_graph import base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    assert base.read(args.run / 'pilot_complete.json')['steps'] == 9
    scripts = Path(__file__).resolve().parent
    project = base.ROOT / 'results/qfad_bidirectional_rsi_p12_20261114'
    plans = {r['episode_id']: r for r in base.read(args.run / 'training_plan.json')}
    state = {'1': [], '2': []}
    policy = args.policy
    decisions = []

    def run(script, *arguments):
        subprocess.run([sys.executable, str(scripts / script), *map(str, arguments)], check=True)

    for step in range(9):
        rollout = base.read(args.run / f'rollouts_{step}.json')
        episode = rollout['episode_id']
        persona = str(plans[episode]['persona'])
        directory = args.output / episode
        rewards = base.read(args.run / f'rewards_{step}.json')
        valid = [v for v in rewards['query_reward_components'] if v is not None]
        decision = {'episode_id': episode, 'step': step, 'accepted': False}
        if not valid or max(valid) < 0:
            decision['reason'] = 'No valid non-regressing Writer action'
        else:
            proposal = directory / 'proposal'
            if not (proposal / 'proposal.json').exists():
                run('propose_rsi_write.py', '--run', args.run, '--step', step, '--output', proposal)
            replacements = base.read(proposal / 'proposed_writes.json')
            merged = {r['source_id']: r for r in state[persona]}
            merged.update({r['source_id']: r for r in replacements})
            candidate = list(merged.values())
            base.write_json(directory / 'old_writes.json', state[persona])
            base.write_json(directory / 'candidate_writes.json', candidate)
            if step in (0, 1):
                evidence = project / ('first_case/new_reader_new_graph/evaluation.json' if step == 0
                                     else 'dose_path_comparison/new_graph/group.json')
                record = base.read(evidence)
                success = record['correct'] if step == 0 else record['updated_correct']
                assert success
                decision.update(accepted=True, reason='Verified four-condition query improvement',
                                feedback_file=str(evidence))
            else:
                before = directory / 'old_graph'
                if not (before / 'evaluation.json').exists():
                    run('rsi_reader_group.py', '--episode', episode, '--writes', directory / 'old_writes.json',
                        '--policy', policy, '--output', before, '--evaluate-only')
                baseline = base.read(before / 'evaluation.json')
                decision['baseline_correct'] = baseline['correct']
                if baseline['correct']:
                    decision['reason'] = 'Current Reader and graph already answer this query correctly'
                else:
                    after = directory / 'new_graph'
                    if not (after / 'group.json').exists():
                        run('rsi_reader_group.py', '--episode', episode, '--writes', directory / 'candidate_writes.json',
                            '--policy', policy, '--output', after)
                    learned = base.read(after / 'group.json')
                    decision.update(accepted=learned['updated_correct'],
                                    updated_correct=learned['updated_correct'],
                                    feedback_file=str(after / 'group.json'),
                                    reason='Retain only a correct response after an incorrect current-graph response')
                    if decision['accepted']:
                        policy = after / 'policy_after.pt'
            if decision['accepted']:
                state[persona] = candidate
        decisions.append(decision)
        base.write_json(directory / 'decision.json', decision)
        base.write_json(args.output / 'graph_state.json', state)
        base.write_json(args.output / 'decisions.json', decisions)
        print(episode, decision, flush=True)
    base.write_json(args.output / 'complete.json', {'policy': str(policy),
        'graph_state': str(args.output / 'graph_state.json'), 'decisions': decisions,
        'accepted_sources': sum(len(v) for v in state.values()),
        'selection_scope': 'same-patient adaptation feedback; full-cohort evaluation required'})


if __name__ == '__main__':
    main()
