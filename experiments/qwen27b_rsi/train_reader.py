"""Train Qwen27B retrieval actions on a fixed source-grounded graph."""
import argparse
import json
from pathlib import Path
import subprocess

from graph_environment import GraphEnvironment, QueryBudgetExceeded, base
from reader_environment import ReaderEpisode, READER_PROTOCOL
from train_writer import Evaluator, REMOTE


def reader_rewards(results, raw_correct, raw_tokens):
    rewards = []
    for result in results:
        if not result['valid']:
            rewards.append(-2.)
            continue
        delta = int(result['correct']) - int(raw_correct)
        saving = (1. - result['total_tokens'] / raw_tokens) if result['correct'] else 0.
        rewards.append(float(delta) + .01 * saving)
    return rewards


def reader_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = (sum((reward - mean) ** 2 for reward in rewards) / len(rewards)) ** .5
    return [(reward - mean) / max(std, 1.) for reward in rewards]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--resume', required=True, help='Remote shared-policy checkpoint')
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = base.read(args.state)
    graphs = {p: GraphEnvironment(base.Patient(p, store_names=('chunk',)), state.get(str(p))) for p in
              {int(episode.split('_')[0][1:]) for episode in args.episodes}}
    raw_usage = base.read(base.OUT / 'raw_usage.json')
    if args.start_index:
        assert base.read(args.out / 'graph_snapshot.json') == state
    else:
        base.write_json(args.out / 'graph_snapshot.json', state)
    configuration = {
        'episodes': args.episodes, 'base_model': 'Qwen3.8-27B', 'resume': args.resume,
        'answer_model_frozen': True, 'graph_frozen': True, 'max_actions': 4,
        'reader_protocol': READER_PROTOCOL,
        'max_action_tokens': 192, 'candidates': 4, 'source_token_budget': 8000,
        'update_epochs': args.update_epochs,
        'advantage_baseline': 'sampled trajectory mean; RAW correctness is a per-question constant',
        'advantage_scale': 'max(group reward standard deviation, 1); preserve small cost-only updates',
        'rerank_document_token_budget': 32000, 'answer_plus_policy_budget': 'native RAW-RAG tokens per query',
        'reward': 'correctness delta versus the frozen RAW-RAG outcome; correct answers use bounded native-token savings versus RAW as tiebreak; invalid and over-budget trajectories rejected',
        'attribution_control': 'Fixed Reader on the same graph, recorded separately from the reward reference',
        'credit': 'trajectory advantage on action tokens; first-exposure provenance recorded',
        'evidence_coverage_auxiliary': False}
    assert 0 <= args.start_index <= len(args.episodes)
    if args.start_index:
        original = base.read(args.out / 'configuration.json')
        for key in configuration.keys() - {'resume'}:
            assert original[key] == configuration[key], key
        for index in range(args.start_index):
            assert base.read(args.out / f'step{index}.update.json')['event'] == 'updated'
        base.write_json(args.out / 'resume_state.json', {
            'checkpoint': args.resume, 'next_episode_index': args.start_index})
    else:
        base.write_json(args.out / 'configuration.json', configuration)
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {REMOTE}/{args.out.name} --resume {args.resume} --update-epochs {args.update_epochs}']
    evaluator = Evaluator(args.out / 'answers')
    writer_feedback = base.read(args.out / 'reader_feedback.json') if args.start_index else {}
    with (args.out / 'worker.log').open('a' if args.start_index else 'w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Reader policy worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready', ready
            if args.start_index:
                assert ready['step'] == base.read(args.out / f'step{args.start_index - 1}.update.json')['step']
            base.write_json(args.out / ('resume_runtime.json' if args.start_index else 'worker_runtime.json'), ready)
            for index, episode_id in enumerate(args.episodes):
                if index < args.start_index:
                    continue
                persona = int(episode_id.split('_')[0][1:])
                qid = episode_id.split('_', 1)[1]
                stage = int(qid.split('_')[1]) // 10 - 1
                graph = graphs[persona]
                question = next(q for q in graph.patient.groups[stage][1] if q['query_id'] == qid)
                baseline = evaluator.answer(graph, question, stage, f'step{index}_fixed')
                raw_answer = next(r for r in base.read(base.RAW / f'persona_{persona:02d}' /
                    f'session_{10 * (stage + 1):03d}_answers.json') if r['query_id'] == qid)
                context_cap = min(8000, len(base.ENC.encode(raw_answer['formatted_context'])))
                budget = raw_usage[str(persona)][qid]['total_tokens']
                raw_correct = base.read(base.ROOT / 'results/qfad_query_writer_rl_20261113/semantic/evaluation' /
                                        f'p{persona}' / f'{qid}_raw.json')['correct']
                episodes = [ReaderEpisode(graph, question, stage, context_cap, native_budget=budget) for _ in range(4)]
                traces, results, costs = [[] for _ in range(4)], [None] * 4, [0] * 4
                first_group = f'step{index}_first'
                first = request({'operation': 'sample', 'id': first_group, 'messages': episodes[0].messages,
                    'count': 4, 'max_tokens': 192, 'seed': 1000 + index})
                base.write_json(args.out / f'{first_group}.samples.json', first)
                for branch, episode in enumerate(episodes):
                    packet, row, group = first, first['rows'][branch], first_group
                    for action_index in range(4):
                        traces[branch].append({'group': group, 'row': branch if action_index == 0 else 0})
                        costs[branch] += packet['prompt_tokens'] + len(row['token_ids'])
                        try:
                            if costs[branch] >= budget:
                                raise ValueError('Policy exceeded native RAW-RAG token budget')
                            episode.step(row, policy_tokens=costs[branch])
                        except (ValueError, KeyError, TypeError) as error:
                            results[branch] = {'valid': False, 'reason': str(error)}
                            break
                        if episode.terminal:
                            break
                        group = f'step{index}_branch{branch}_action{action_index + 1}'
                        packet = request({'operation': 'sample', 'id': group, 'messages': episode.messages,
                            'count': 1, 'max_tokens': 192, 'seed': 2000 + index * 100 + branch * 4 + action_index})
                        base.write_json(args.out / f'{group}.samples.json', packet)
                        row = packet['rows'][0]
                    evidence = episode.evidence()
                    if results[branch] is None:
                        try:
                            answer = evaluator.answer(graph, question, stage, f'step{index}_branch{branch}',
                                evidence=evidence, native_budget=budget - costs[branch])
                            total_tokens = costs[branch] + answer['usage']['total_tokens']
                            if total_tokens > budget:
                                raise QueryBudgetExceeded('Measured native token usage exceeds RAW-RAG budget')
                            results[branch] = {'valid': True, 'correct': answer['score']['correct'],
                                'delta': int(answer['score']['correct']) - int(baseline['score']['correct']),
                                'delta_vs_raw': int(answer['score']['correct']) - int(raw_correct),
                                'total_tokens': total_tokens, 'native_budget': budget}
                        except QueryBudgetExceeded as error:
                            results[branch] = {'valid': False, 'reason': str(error)}
                    base.write_json(args.out / f'step{index}_branch{branch}.trajectory.json', {
                        'evidence': evidence, 'messages': episode.messages, 'policy_tokens': costs[branch],
                        'policy_actions': traces[branch], 'result': results[branch]})
                rewards = reader_rewards(results, raw_correct, budget)
                eligible = [i for i, result in enumerate(results) if result['valid']]
                if eligible:
                    best = max(eligible, key=lambda i: (results[i]['correct'], -results[i]['total_tokens']))
                    evidence = episodes[best].evidence()
                    source_ids = list(evidence['source_ids'])
                    for fid in evidence['fact_ids']:
                        source_ids.extend(c['source_id'] for c in graph.state['facts'][fid]['citations'])
                    writer_feedback[episode_id] = {'source_ids': list(dict.fromkeys(source_ids)),
                        'correct': results[best]['correct'], 'selected_branch': best,
                        'origin': 'Evidence actually obtained and selected by the Qwen27B Reader policy'}
                    base.write_json(args.out / 'reader_feedback.json', writer_feedback)
                advantages = reader_advantages(rewards)
                base.write_json(args.out / f'step{index}.rewards.json', {
                    'results': results, 'rewards': rewards, 'advantages': advantages,
                    'raw_correct': raw_correct, 'raw_tokens': budget,
                    'fixed_reader_correct': baseline['score']['correct']})
                update = request({'operation': 'update', 'id': f'step{index}', 'trajectories': traces,
                    'rewards': rewards, 'advantages': advantages})
                base.write_json(args.out / f'step{index}.update.json', update)
                print(episode_id, rewards, 'adapter_delta', update['adapter_delta_l2'], flush=True)
            request({'operation': 'close'})
            worker.wait()
            base.write_json(args.out / 'complete.json', {'steps': len(args.episodes), 'worker_exit': worker.returncode})
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
