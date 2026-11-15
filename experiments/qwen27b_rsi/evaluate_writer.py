"""Regenerate Writer actions from identical saved observations with a frozen policy."""
import argparse
import json
from pathlib import Path
import subprocess

from graph_environment import GraphEnvironment, base
from train_writer import Evaluator, REMOTE, reader_cost_term
from reader_environment import READER_PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--resume', help='Remote checkpoint; omit for the initial base policy')
    parser.add_argument('--paired-with', type=Path)
    parser.add_argument('--reader-checkpoint', help='Remote frozen Reader snapshot used for every graph comparison')
    parser.add_argument('--samples', type=int, default=1, help='One is greedy; larger values evaluate all sampled actions')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    assert args.samples > 0
    args.out.mkdir(parents=True, exist_ok=True)
    training = base.read(args.training_run / 'configuration.json')
    episodes = training['episodes']
    if args.paired_with:
        prior_config = base.read(args.paired_with / 'configuration.json')
        assert prior_config['training_run'] == str(args.training_run.resolve())
        assert prior_config.get('reader_checkpoint') == args.reader_checkpoint
        assert prior_config.get('reader_protocol') == (READER_PROTOCOL if args.reader_checkpoint else None)
    base.write_json(args.out / 'configuration.json', {
        'training_run': str(args.training_run.resolve()), 'episodes': episodes,
        'policy_model': 'Qwen3.8-27B', 'resume': args.resume,
        'policy_frozen': True, 'decoding': 'greedy' if args.samples == 1 else 'sample',
        'samples_per_state': args.samples, 'seed': args.seed, 'max_action_tokens': 768,
        'answer_model': 'Qwen3.8-27B-FP8', 'answer_model_frozen': True,
        'reader_checkpoint': args.reader_checkpoint,
        'reader_protocol': READER_PROTOCOL if args.reader_checkpoint else None,
        'paired_with': str(args.paired_with) if args.paired_with else None,
        'scope': 'Independent saved input states; every generated action evaluated, no QA-based selection'})
    patients = {p: base.Patient(p, store_names=('chunk',)) for p in {int(e.split('_')[0][1:]) for e in episodes}}
    evaluator = Evaluator(args.out / 'answers')
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {REMOTE}/{args.out.name}']
    if args.resume:
        command[-1] += ' --resume ' + args.resume
    results = []
    with (args.out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Writer evaluation worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready', ready
            base.write_json(args.out / 'worker_runtime.json', ready)
            if args.reader_checkpoint:
                from policy_reader_evaluator import PolicyReaderEvaluator
                frozen = request({'operation': 'freeze_reader', 'checkpoint': args.reader_checkpoint})
                assert frozen['event'] == 'reader_frozen', frozen
                base.write_json(args.out / 'reader_snapshot.json', frozen)
                evaluator = PolicyReaderEvaluator(evaluator, request)
            for index, episode in enumerate(episodes):
                persona = int(episode.split('_')[0][1:])
                qid = episode.split('_', 1)[1]
                stage = int(qid.split('_')[1]) // 10 - 1
                graph = GraphEnvironment(patients[persona], base.read(args.training_run / f'step{index}.before.json'))
                observation = base.read(args.training_run / f'step{index}.observation.json')
                base.write_json(args.out / f'step{index}.observation.json', observation)
                sources = [s['source_id'] for s in json.loads(observation[1]['content'])['sources']]
                query_ids = [f.stem.split('_noop_', 1)[1] for f in
                             sorted((args.training_run / 'answers').glob(f'step{index}_noop_*.json'))]
                assert qid in query_ids
                queries = [next(q for q in graph.patient.groups[stage][1] if q['query_id'] == other)
                           for other in query_ids]
                baseline = [evaluator.answer(graph, q, stage, f'step{index}_noop_{q["query_id"]}',
                            paired_answer=base.read((args.paired_with or args.training_run) / 'answers' /
                                                   f'step{index}_noop_{q["query_id"]}.json')) for q in queries]
                packet = request({'operation': 'infer' if args.samples == 1 else 'sample',
                                  'id': f'step{index}', 'messages': observation, 'count': args.samples,
                                  'seed': args.seed + index, 'max_tokens': 768, 'native_budget': 12768})
                base.write_json(args.out / f'step{index}.samples.json', packet)
                for branch in range(args.samples):
                    prefix = f'step{index}' if args.samples == 1 else f'step{index}_sample{branch}'
                    action = None
                    try:
                        if packet['event'] == 'budget_exhausted':
                            raise ValueError('Writer observation exceeds policy budget')
                        action = graph.parse(packet['rows'][branch], sources, stage)
                    except (ValueError, KeyError, TypeError) as error:
                        verdict = {'supported': False, 'reason': str(error)}
                        candidate = graph
                    else:
                        verdict = graph.verify(action, evaluator.judge)
                        candidate = graph.apply(action, verdict) if verdict['supported'] else graph
                    base.write_json(args.out / f'{prefix}.support.json', verdict)
                    base.write_json(args.out / f'{prefix}.graph.json', candidate.state)
                    answers = []
                    for q, original in zip(queries, baseline):
                        label = f'{prefix}_candidate_{q["query_id"]}'
                        if candidate.state == graph.state:
                            answer = {**original, 'unchanged_graph_reuse': True}
                            base.write_json(args.out / 'answers' / f'{label}.json', answer)
                        else:
                            paired_path = args.paired_with / 'answers' / f'{label}.json' if args.paired_with else None
                            paired = base.read(paired_path) if paired_path and paired_path.exists() else None
                            answer = evaluator.answer(candidate, q, stage, label, paired_answer=paired)
                        answers.append(answer)
                    record = {'episode': episode, 'sample_index': branch, 'action': action, 'supported': verdict['supported'],
                              'query_ids': query_ids, 'before': [a['score']['correct'] for a in baseline],
                              'after': [a['score']['correct'] for a in answers],
                              'query_deltas': [int(a['score']['correct']) - int(b['score']['correct'])
                                               for a, b in zip(answers, baseline)],
                              'writer_tokens': packet['prompt_tokens'] + len(packet['rows'][branch]['token_ids'])
                                               if 'rows' in packet else 0}
                    if args.reader_checkpoint:
                        marginal_tokens = (len(packet['rows'][branch]['token_ids'])
                                           if action and action['operation'] != 'NOOP' else 0)
                        cost_term, costs = reader_cost_term(baseline, answers, marginal_tokens)
                        record.update(cost_term=cost_term, reader_costs=costs,
                                      paired_reward=sum(record['query_deltas']) + .01 * cost_term
                                      if verdict['supported'] else -len(queries) - 1)
                    base.write_json(args.out / f'{prefix}.result.json', record)
                    results.append(record)
                    print(episode, branch, action['operation'] if action else None,
                          record['supported'], record['query_deltas'], flush=True)
            request({'operation': 'close'})
            worker.wait()
            base.write_json(args.out / 'summary.json', {
                'states': len(episodes), 'samples_per_state': args.samples, 'results': results,
                'net_correct': sum(sum(r['query_deltas']) for r in results),
                'mean_net_correct_per_state_set': sum(sum(r['query_deltas']) for r in results) / args.samples,
                'writer_tokens': sum(r['writer_tokens'] for r in results),
                'scope': 'All regenerated operations on the same saved observations and pre-write graphs; no best-of-N selection'})
            base.write_json(args.out / 'complete.json', {'states': len(episodes), 'samples_per_state': args.samples,
                                                       'worker_exit': worker.returncode})
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
