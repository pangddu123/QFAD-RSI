"""Train source-only batch extraction with fixed QFAD and clinical query feedback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from clinical_evidence import EvidenceScorer, extraction_reward, group_advantages, has_task_signal, validate_requirements
from medical_navigation import NavigationGraph, admit_extraction, base, extraction_messages, parse_extraction, verify_extraction
from medmemory_scoring import JudgeClient
from train_writer import Evaluator
from targeted_guidance import attribute_writes, extraction_hint, render_hint


REMOTE = '/home/administrator/du/memos/qwen27b_rsi'


def apply_sample(row, graph, sources, existing, judge):
    extraction, admission, support, accepted = None, None, None, None
    candidate = graph
    try:
        extraction = parse_extraction(row, sources, existing)
    except (ValueError, KeyError, TypeError) as error:
        support = {'parse_error': str(error)}
    else:
        support = verify_extraction(extraction, sources, existing, judge)
        accepted, admission = admit_extraction(extraction, support, existing)
        candidate = graph.extract_into(accepted, [s['source_id'] for s in sources])
    return {'parsed': extraction is not None, 'extraction': extraction, 'support': support,
            'admission': admission, 'admitted': accepted}, candidate


def evaluate_candidate(row, graph, sources, existing, episode, evaluator, evidence_scorer, judge, directory, label,
                       collect_hint=False):
    result, candidate = apply_sample(row, graph, sources, existing, judge)
    answers, coverage = [], []
    for item in episode['queries']:
        question = next(q for q in graph.patient.groups[item['stage']][1] if q['query_id'] == item['query_id'])
        answer = evaluator.answer(candidate, question, item['stage'], label + '_' + item['query_id'])
        covers = evidence_scorer.score(question['question'], item['requirements'], answer['retrieval'])
        answers.append(answer)
        coverage.append(covers)
    result.update({'coverage': coverage,
              'query_ids': [q['query_id'] for q in episode['queries']],
              'correct': [a['score']['correct'] for a in answers],
              'feedback': extraction_reward(answers, coverage, result['admission'])})
    if collect_hint:
        result['text_feedback'] = extraction_hint({s['source_id']: s for s in sources},
            episode['queries'], result, [answer['retrieval'] for answer in answers])
    base.write_json(directory / f'{label}.feedback.json', result)
    base.write_json(directory / f'{label}.graph.json', candidate.state)
    return result, candidate


def episode_windows(episodes, size):
    if size < 1:
        raise ValueError('A trajectory requires at least one write')
    window = []
    for episode in episodes:
        past_query = any(q['stage'] < episode.get('source_stage', -1) for e in window for q in e['queries'])
        if window and (episode['patient'] != window[0]['patient'] or len(window) == size or past_query):
            yield window
            window = []
        window.append(episode)
    if window:
        yield window


def train_window(window, initial, group, seed, request, evaluator, scorer, judge, out, weight, group_size=4):
    """Independent on-policy trajectories, followed by one group update."""
    branches = [{'graphs': [initial], 'turns': [], 'path': [], 'outputs': []} for _ in range(group_size)]
    base.write_json(out / f'{group}.before_graph.json', initial.state)
    queries = {}
    sampled_step = None
    for t, episode in enumerate(window):
        for item in episode['queries']:
            previous = queries.setdefault(item['query_id'], item)
            if previous != item:
                raise ValueError('Conflicting annotations for a repeated query')
        # The first observation is shared. Later observations depend on each branch's own writes.
        batch_size = 1 if t else 4
        for b in range(0, group_size, batch_size):
            branch = branches[b]
            graph = branch['graphs'][-1]
            sources = [graph.sources[sid] for sid in episode['source_ids']]
            stage = min(s['stage'] for s in sources)
            existing = dict(list(graph.visible_facts(stage).items())[-8:])
            messages = extraction_messages(sources, existing)
            key = f'{group}_t{t}_b{b}'
            count = min(batch_size, group_size - b)
            samples = request({'operation': 'sample', 'id': key, 'messages': messages, 'seed': seed + group_size*t + b,
                               'max_tokens': 3072, 'count': count})
            if sampled_step is None:
                sampled_step = samples['policy_step']
            if samples['event'] != 'samples' or len(samples['rows']) != count or samples['policy_step'] != sampled_step:
                raise ValueError('Incomplete or off-policy multi-write trajectory')
            base.write_json(out / f'{key}.observation.json', messages)
            base.write_json(out / f'{key}.samples.json', samples)
            for row_index, row in enumerate(samples['rows']):
                dest = branch if t else branches[b + row_index]
                result, next_graph = apply_sample(row, graph, sources, existing, judge)
                dest['turns'].append({**result, 'sources': {s['source_id']: s for s in sources}, 'messages': messages})
                dest['graphs'].append(next_graph)
                dest['outputs'].append(row['text'])
                dest['path'].append({'group': key, 'row': row_index})

    items = list(queries.values())
    if not items:
        raise ValueError('No terminal query in this trajectory')

    def score_branch(pair):
        b, branch = pair
        final = branch['graphs'][-1]
        histories, answers, verdicts = [], [], []
        label = f'{group}_b{b}'
        for item in items:
            stage, qid = item['stage'], item['query_id']
            q = next(q for q in final.patient.groups[stage][1] if q['query_id'] == qid)
            answer = evaluator.answer(final, q, stage, f'{label}_{qid}')
            verdict = scorer.score(q['question'], item['requirements'], answer['retrieval'])
            terminal = (verdict, answer['retrieval'])
            answers.append(answer)
            verdicts.append(verdict)
            # Only unresolved terminal conditions need before/after attribution.
            if verdict['complete']:
                histories.append([None] * len(window) + [terminal])
                continue
            cap = evaluator.context_cap(final, q, stage)
            history = []
            for prefix in branch['graphs'][:-1]:
                trace = prefix.retrieve(q['question'], stage, context_cap=cap)
                coverage = scorer.score(q['question'], item['requirements'], trace)
                history.append((coverage, trace))
            histories.append(history + [terminal])
        events = attribute_writes(branch['turns'], items, histories)
        per_turn = [extraction_reward(answers, verdicts, turn['admission']) for turn in branch['turns']]
        reward = sum(x['reward'] for x in per_turn) / len(per_turn)
        hints = [render_hint(turn_events) for turn_events in events]
        base.write_json(out / f'{label}.credit.json', {'patient': window[0]['patient'], 'policy_step': sampled_step,
            'episodes': [e['id'] for e in window], 'queries': list(queries), 'path': branch['path'],
            'source_ids_by_turn': [list(t['sources']) for t in branch['turns']],
            'writes': [{'parsed': t['parsed'], 'accepted': t.get('admitted'), 'admission': t['admission'],
                        'source_verdict': t.get('support')} for t in branch['turns']],
            'prefix_evidence': histories, 'events_by_turn': events, 'hints_by_turn': hints,
            'source_error_by_turn': [x['source_error_rate'] for x in per_turn], 'reward': reward})
        return reward, hints

    with ThreadPoolExecutor(max_workers=4) as pool:
        scored = list(pool.map(score_branch, enumerate(branches)))
    rewards = [score for score, _ in scored]
    advantages = group_advantages(rewards)
    paths = [b['path'] for b in branches]
    guidance = [{**action, 'messages': turn['messages'], 'hint': hint}
                for branch, (_, hints) in zip(branches, scored)
                for action, turn, hint in zip(branch['path'], branch['turns'], hints, strict=True) if hint]
    report = {'episodes': [e['id'] for e in window], 'policy_step': sampled_step, 'rewards': rewards,
              'advantages': advantages, 'trajectories': paths, 'sampled_trajectories': len(paths),
              'distinct_output_trajectories': len({tuple(b['outputs']) for b in branches})}
    if any(advantages) or (weight and guidance):
        extras = {'turn_guidance': guidance, 'guidance_weight': weight} if weight else {}
        update = request({'operation': 'update', 'id': group, 'trajectories': paths,
                          'advantages': advantages, 'rewards': rewards, **extras})
        if (update['event'] != 'updated' or update['step'] != sampled_step + 1 or
                update['adapter_delta_l2'] <= 0 or (weight and update['guided_turns'] != len(guidance))):
            raise RuntimeError('Unexpected multi-turn update result')
        base.write_json(out / f'{group}.update.json', update)
        report.update(updated_step=update['step'], guided_turns=update.get('guided_turns', 0),
                      targeted_reverse_kl=update.get('targeted_reverse_kl', 0.))
    else:
        keys = sorted({action['group'] for path in paths for action in path})
        if request({'operation': 'discard_samples', 'groups': keys})['event'] != 'discarded':
            raise RuntimeError('Worker did not release the skipped trajectory samples')
        report['skipped'] = 'No group advantage or enabled hint'
    base.write_json(out / f'{group}.rewards.json', report)
    return report, branches[0]['graphs'][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', help='Remote extraction-policy checkpoint')
    parser.add_argument('--worker-script', default=REMOTE + '/policy_worker.py')
    parser.add_argument('--guidance-weight', type=float, help='Failed-turn reverse KL weight. '
                        'Set to 0 for a matched scalar-only ablation. Omit for the legacy task-variation gate.')
    args = parser.parse_args()
    if args.guidance_weight is not None and (not math.isfinite(args.guidance_weight) or args.guidance_weight < 0):
        parser.error('Guidance weight must be finite and nonnegative')
    calibration = base.read(base.ROOT / 'results/medical_sage/clinical_calibration/summary.json')
    assert calibration['cases'] == 14 and not calibration['failures'], 'Clinical calibration must pass before training'
    plan = base.read(args.plan)
    turns_per_rollout = plan.get('turns_per_rollout', 3) if args.guidance_weight is not None else 1
    group_size = plan.get('group_size', 4)
    if type(group_size) is not int or group_size < 2 or (args.guidance_weight is None and group_size != 4):
        raise ValueError('Targeted group size must be >= 2; legacy single-write training uses 4')
    allowed = set(plan.get('patient_split', {}).get('train', [1, 2]))
    if not plan['episodes'] or {e['patient'] for e in plan['episodes']} - allowed:
        raise ValueError('Episode outside the configured training patients')
    assert not args.out.exists(), 'Use a new output directory for a new training run'
    graphs = {pid: NavigationGraph(base.Patient(pid, store_names=('chunk',)))
              for pid in sorted({e['patient'] for e in plan['episodes']})}
    source_order = {pid: {text: i for i, (_, text) in reversed(list(enumerate(g.patient.history)))}
                    for pid, g in graphs.items()}
    previous_order = {pid: -1 for pid in graphs}
    for episode in plan['episodes']:
        graph = graphs[episode['patient']]
        positions = [source_order[episode['patient']][graph.sources[s]['text']] for s in episode['source_ids']]
        assert positions and min(positions) > previous_order[episode['patient']], 'Writer episodes must follow source arrival order'
        previous_order[episode['patient']] = max(positions)
        source_stage = max(graph.sources[s]['stage'] for s in episode['source_ids'])
        for query in episode['queries']:
            assert query['stage'] >= source_stage
            validate_requirements(query['requirements'], graph, query['stage'])
    args.out.mkdir(parents=True)
    base.write_json(args.out / 'configuration.json', {
        'process_id': os.getpid(), 'plan': plan, 'resume': args.resume,
        'model': 'Qwen3.8-27B', 'policy_quantization': 'NF4', 'sample_count': group_size,
        'max_output_tokens': 3072, 'source_only_writer': True,
        'reader': 'Fixed QFAD with original-source windows', 'answerer': 'Fixed Qwen3.8-27B-FP8',
        'reward': 'Supported answer count + 0.5 mean clinical coverage - 0.25 rejected-atom fraction',
        'update_condition': ('Task-utility variation among parsed candidates' if args.guidance_weight is None
                             else 'Nonzero group advantages or a nonzero-weight failed-turn hint'),
        'guidance_weight': args.guidance_weight, 'worker_script': args.worker_script,
        'turns_per_rollout': turns_per_rollout,
        'state_transition': 'First sampled candidate, independent of reward; this is a training rollout state',
        'update_epochs': 1, 'scope': 'Extraction training execution; not a complete-patient improvement estimate',
    })
    remote_out = REMOTE + '/' + args.out.name
    remote_args = ['env', 'CUDA_VISIBLE_DEVICES=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        '/home/administrator/du/memskill-determinacy-20261014/venv/bin/python', '-u',
        args.worker_script, '--model', '/home/administrator/du/model/Qwen/Qwen3.8-27B',
        '--out', remote_out, '--update-epochs', '1']
    if args.resume:
        remote_args += ['--resume', args.resume]
    command = ['ssh', '-T', 'win', 'ssh 2A100 ' + shlex.join(remote_args)]
    judge = JudgeClient('gpt-6.1-sol')
    evidence_scorer = EvidenceScorer(judge)
    evaluator = Evaluator(args.out / 'answers')
    results = []
    with (args.out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Policy worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            runtime = receive()
            assert runtime['event'] == 'ready'
            base.write_json(args.out / 'worker_runtime.json', runtime)
            for index, window in enumerate(episode_windows(plan['episodes'], turns_per_rollout)):
                episode = window[0]
                graph = graphs[episode['patient']]
                if args.guidance_weight is not None:
                    result, graphs[episode['patient']] = train_window(window, graph, f'group{index}',
                        17 + index * group_size * turns_per_rollout, request, evaluator, evidence_scorer, judge,
                        args.out, args.guidance_weight, group_size)
                    results.append(result)
                    base.write_json(args.out / 'groups.json', results)
                    base.write_json(args.out / 'rollout_state.json', {str(pid): g.state for pid, g in graphs.items()})
                    print(json.dumps(result), flush=True)
                    continue
                sources = [graph.sources[sid] for sid in episode['source_ids']]
                stage = min(s['stage'] for s in sources)
                existing = dict([(fid, f) for fid, f in graph.visible_facts(stage).items()
                    if not any(c['source_id'] in episode['source_ids'] for c in f['citations'])][-8:])
                messages = extraction_messages(sources, existing)
                group = f'group{index}'
                base.write_json(args.out / f'{group}.observation.json', messages)
                base.write_json(args.out / f'{group}.before_graph.json', graph.state)
                samples = request({'operation': 'sample', 'id': group, 'messages': messages,
                                   'seed': 17 + index, 'max_tokens': 3072, 'count': 4})
                assert samples['event'] == 'samples' and len(samples['rows']) == 4
                base.write_json(args.out / f'{group}.samples.json', samples)
                def evaluate(item):
                    i, row = item
                    return evaluate_candidate(row, graph, sources, existing, episode, evaluator, evidence_scorer, judge,
                                              args.out, f'{group}_candidate{i}')
                with ThreadPoolExecutor(max_workers=4) as pool:
                    evaluated = list(pool.map(evaluate, enumerate(samples['rows'])))
                candidates = [result for result, _ in evaluated]
                rewards = [r['feedback']['reward'] for r in candidates]
                advantages = group_advantages(rewards)
                signal = has_task_signal(candidates)
                result = {'episode': episode['id'], 'policy_step': samples['policy_step'],
                          'rewards': rewards, 'advantages': advantages, 'task_signal': signal,
                          'task_utilities': [r['feedback']['task_utility'] for r in candidates]}
                base.write_json(args.out / f'{group}.rewards.json', result)
                if signal and any(advantages):
                    update = request({'operation': 'update', 'id': group,
                                      'advantages': advantages, 'rewards': rewards})
                    assert update['event'] == 'updated' and update['adapter_delta_l2'] > 0
                    assert update['step'] == samples['policy_step'] + 1
                    base.write_json(args.out / f'{group}.update.json', update)
                    result['updated_step'] = update['step']
                else:
                    result['skipped'] = 'No signal under the configured update condition'
                # Training transitions use a fixed sampled row, never a reward-selected graph.
                graphs[episode['patient']] = evaluated[0][1]
                base.write_json(args.out / 'rollout_state.json', {str(pid): g.state for pid, g in graphs.items()})
                results.append(result)
                base.write_json(args.out / 'groups.json', results)
                print(json.dumps(result), flush=True)
            closed = request({'operation': 'close'})
            worker.wait()
            assert closed['event'] == 'closed' and worker.returncode == 0
            summary = {'groups': len(results), 'updates': sum('updated_step' in r for r in results),
                       'guided_turns': sum(r.get('guided_turns', 0) for r in results),
                       'groups_with_positive_guidance_kl': sum(r.get('targeted_reverse_kl', 0.) > 0 for r in results),
                       'initial_step': runtime['step'], 'final_step': closed['step'],
                       'remote_checkpoint': remote_out + '/checkpoint.pt' if closed['step'] > runtime['step'] else None,
                       'scope': 'Batch-extraction training execution, not a full-patient accuracy result'}
            if args.guidance_weight is not None:
                summary['sampled_trajectories'] = sum(r['sampled_trajectories'] for r in results)
                summary['windows_with_reward_variation'] = sum(len(set(r['rewards'])) > 1 for r in results)
            base.write_json(args.out / 'summary.json', summary)
            base.write_json(args.out / 'complete.json', {'worker_exit': worker.returncode, 'final_step': closed['step']})
            print(json.dumps(summary), flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
