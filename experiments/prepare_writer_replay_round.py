"""Continue Writer optimization against the learned Reader and accepted graph."""
import argparse
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--reader-round', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    complete = read(args.reader_round / 'ready.json')
    assert read(args.previous / 'pilot_complete.json')['steps'] == 9
    state = read(complete['graph_state'])
    project = Path(__file__).resolve().parents[2] / 'results/qfad_bidirectional_rsi_p12_20261114'
    assert read(args.evaluation / 'summary.json')['n'] == 197
    previous_plans = read(args.previous / 'training_plan.json')
    previous_inputs = {r['episode_id']: r for r in read(args.previous / 'writer_inputs.json')}
    source_inputs = {r['episode_id']: r for r in read(project / 'writer_source_inputs.json')}
    pool = read(project / 'feedback_pool.json')
    seen = set(previous_inputs)
    failures = []
    for item in pool:
        episode = item['episode_id']
        if episode in seen or episode not in source_inputs or len(source_inputs[episode]['sources']) != 1:
            continue
        result = read(args.evaluation / f'p{item["persona"]}/{item["query_id"]}.json')
        if item['verified_source_reader_correct'] and not result['semantic_score']['correct']:
            failures.append(item)
    failures.sort(key=lambda r: (not r['raw_semantic_correct'], r['checkpoint'], r['persona'], r['query_id']))
    plans = previous_plans[:2]
    inputs = [previous_inputs[r['episode_id']] for r in plans]
    for item in failures[:7]:
        plans.append({'episode_id': item['episode_id'], 'persona': item['persona'],
            'stage': item['stage'], 'checkpoint': item['checkpoint'], 'query_ids': [item['query_id']],
            'source_selection': {'adaptation_feedback_used': True,
                'verified_source_file': item['feedback_source_file'], 'writer_receives_query_or_reference': False},
            'raw_semantic_correct': item['raw_semantic_correct']})
        inputs.append(source_inputs[item['episode_id']])
    for plan in previous_plans[2:]:
        if len(plans) == 9:
            break
        plans.append(plan)
        inputs.append(previous_inputs[plan['episode_id']])
    assert len(plans) == len(inputs) == 9
    personas = {r['episode_id']: str(r['persona']) for r in plans}
    for item in inputs:
        patches = {r['source_id']: r['facts'] for r in state[personas[item['episode_id']]]}
        for source in item['sources']:
            if source['source_id'] in patches:
                source['old_facts'] = [{k: v for k, v in fact.items() if k != 'evidence'}
                                       for fact in patches[source['source_id']]]
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('writer_inputs.json', inputs), ('training_plan.json', plans), ('graph_state.json', state)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    shutil.copyfile(complete['policy'], args.output / 'reader_policy.pt')
    scripts = Path(__file__).resolve().parent
    names = ['fact_writer_worker.py', 'fact_writer_protocol.py', 'train_fact_writer.py', 'query_reward.py',
             'qfad_fact_graph.py', 'rsi_reader_policy.py', 'rsi_reader_group.py', 'semantic_query_writer_rl.py']
    for name in names:
        shutil.copyfile(scripts / name, args.output / name)
    remote_root = '/home/administrator/du/memos/fact_writer_rl_20261114'
    remote = f'{remote_root}/{args.output.name}'
    previous = f'{remote_root}/{args.previous.name}'
    script = f'''set -e
if pgrep -f '[f]act_writer_worker.py|[i]nfer_fact_writer.py' >/dev/null; then
    echo 'Writer GPU work is already active'; exit 1
fi
free_mb=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
test "$free_mb" -ge 18000
test ! -e {shlex.quote(remote)}
mkdir -p {shlex.quote(remote + '/pilot')}
cp {shlex.quote(previous + '/pilot/checkpoint.pt')} {shlex.quote(remote + '/pilot/checkpoint.pt')}
'''
    subprocess.run(['ssh', '-T', 'win', 'ssh 2A100 bash -s'], input=script.encode(), check=True)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for name in names + ['writer_inputs.json', 'training_plan.json']:
            tar.add(args.output / name, arcname=name)
    subprocess.run(['ssh', '-T', 'win', f'ssh 2A100 tar -C {shlex.quote(remote)} -xf -'],
                   input=archive.getvalue(), check=True)
    metadata = {'initialization': 'previous Writer checkpoint with optimizer state',
                'previous_run': str(args.previous), 'start_step': 9, 'end_step_exclusive': 18,
                'reader_policy': str(args.output / 'reader_policy.pt'),
                'graph_state': str(args.output / 'graph_state.json'),
                'writer_input': 'visible sources and current graph facts; no questions or reference answers',
                'reward_metric': 'fixed_semantic_rubric', 'answer_model_frozen': 'Qwen3.8-27B',
                'training_patients': [1, 2], 'remote': remote,
                'curriculum': 'Reader-improved episodes and source-verified failures from the full evaluation',
                'new_failure_episodes': min(7, len(failures)),
                'episode_ids': [r['episode_id'] for r in plans]}
    (args.output / 'configuration.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metadata, flush=True)


if __name__ == '__main__':
    main()
