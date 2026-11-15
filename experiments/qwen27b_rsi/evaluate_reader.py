"""Evaluate a frozen Qwen27B Reader through actual budgeted graph actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading

from graph_environment import GraphEnvironment, QueryBudgetExceeded, base
from reader_environment import ReaderEpisode, READER_PROTOCOL
from train_writer import Evaluator, REMOTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--resume', help='Remote checkpoint; omit to evaluate the initial policy')
    parser.add_argument('--episodes', nargs='+', help='Defaults to all 197 questions')
    parser.add_argument('--paired-with', type=Path, help='Reuse answers for identical prompts and completion limits')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--first-search-only', action='store_true',
                        help='Deliver the first SEARCH results directly to the frozen answerer')
    parser.add_argument('--patients', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--raw-usage', type=Path, default=base.OUT / 'raw_usage.json')
    parser.add_argument('--raw-scores', type=Path, default=base.OUT / 'semantic/evaluation')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.paired_with:
        paired_config = base.read(args.paired_with / 'configuration.json')
        assert paired_config['answer_model'] == 'Qwen3.8-27B-FP8'
    state = base.read(args.state)
    graphs = {p: GraphEnvironment(base.Patient(p, store_names=('chunk',)), state.get(str(p))) for p in args.patients}
    jobs = [(p, stage, q) for p, graph in graphs.items()
            for stage, (_, queries) in enumerate(graph.patient.groups) for q in queries]
    if args.episodes:
        jobs = [job for job in jobs if f'p{job[0]}_{job[2]["query_id"]}' in args.episodes]
        assert len(jobs) == len(set(args.episodes)), 'Unknown episode requested'
    elif args.patients == [1, 2]:
        assert len(jobs) == 197
    base.write_json(args.out / 'graph_snapshot.json', state)
    base.write_json(args.out / 'configuration.json', {
        'policy_model': 'Qwen3.8-27B', 'resume': args.resume, 'decoding': 'greedy', 'process_id': os.getpid(),
        'paired_with': str(args.paired_with) if args.paired_with else None,
        'policy_frozen': True, 'graph_frozen': True, 'answer_model_frozen': True,
        'answer_model': 'Qwen3.8-27B-FP8', 'queries': len(jobs),
        'concurrency': args.concurrency,
        'patients': args.patients, 'raw_usage': str(args.raw_usage), 'raw_scores': str(args.raw_scores),
        'max_actions': 1 if args.first_search_only else 4,
        'first_search_only': args.first_search_only,
        'max_action_tokens': 192, 'source_token_budget': 8000,
        'reader_protocol': READER_PROTOCOL,
        'rerank_document_token_budget': 32000,
        'native_budget': 'RAW-RAG answer tokens per question, including all Reader decisions and answer',
        'selection': 'One greedy trajectory per question, without answer-based path selection'})
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {REMOTE}/{args.out.name}']
    if args.resume:
        command[-1] += ' --resume ' + args.resume
    raw_usage = base.read(args.raw_usage)
    evaluator = Evaluator(args.out / 'answers')
    rows = []
    with (args.out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Reader worker ended; see worker.log')
            return json.loads(line)
        request_lock = threading.Lock()
        def request(packet):
            with request_lock:
                worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
                worker.stdin.flush()
                return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready', ready
            base.write_json(args.out / 'worker_runtime.json', ready)
            def evaluate(job):
                persona, stage, question = job
                name = f'p{persona}_{question["query_id"]}'
                path = args.out / 'trajectories' / f'{name}.json'
                if path.exists():
                    return base.read(path)
                graph = graphs[persona]
                raw = next(r for r in base.read(base.RAW / f'persona_{persona:02d}' /
                    f'session_{10 * (stage + 1):03d}_answers.json') if r['query_id'] == question['query_id'])
                budget = raw_usage[str(persona)][question['query_id']]['total_tokens']
                cap = min(8000, len(base.ENC.encode(raw['formatted_context'])))
                episode = ReaderEpisode(graph, question, stage, cap, native_budget=budget)
                policy_tokens, answer_tokens, result, packets = 0, 0, None, []
                for action_index in range(1 if args.first_search_only else 4):
                    packet = request({'operation': 'infer', 'id': f'{name}_action{action_index}',
                        'messages': episode.messages, 'seed': 0, 'max_tokens': 192,
                        'native_budget': budget - policy_tokens})
                    packets.append(packet)
                    if packet['event'] == 'budget_exhausted':
                        result = {'valid': False, 'correct': False,
                                  'reason': 'Next decision exceeds remaining budget or policy input limit'}
                        break
                    row = packet['rows'][0]
                    policy_tokens += packet['prompt_tokens'] + len(row['token_ids'])
                    try:
                        if policy_tokens >= budget:
                            raise ValueError('Policy exceeded native RAW-RAG token budget')
                        observation = episode.step(row, policy_tokens=policy_tokens)
                        if args.first_search_only:
                            episode.finish_first_search(observation)
                    except (ValueError, KeyError, TypeError) as error:
                        result = {'valid': False, 'correct': False, 'reason': str(error)}
                        break
                    if episode.terminal:
                        break
                evidence = episode.evidence()
                if result is None:
                    try:
                        prior_path = args.paired_with / 'answers' / f'{name}.json' if args.paired_with else None
                        paired_answer = base.read(prior_path) if prior_path and prior_path.exists() else None
                        answer = evaluator.answer(graph, question, stage, name, evidence=evidence,
                                                  native_budget=budget - policy_tokens, paired_answer=paired_answer)
                        answer_tokens = answer['usage']['total_tokens']
                        if policy_tokens + answer_tokens > budget:
                            raise QueryBudgetExceeded('Measured native usage exceeded RAW-RAG budget')
                        result = {'valid': True, 'correct': answer['score']['correct']}
                    except QueryBudgetExceeded as error:
                        result = {'valid': False, 'correct': False, 'reason': str(error)}
                raw_score = base.read(args.raw_scores / f'p{persona}' / f'{question["query_id"]}_raw.json')['correct']
                row = {'episode_id': name, 'result': result, 'raw_correct': raw_score,
                       'policy_tokens': policy_tokens, 'answer_tokens': answer_tokens,
                       'total_tokens': policy_tokens + answer_tokens, 'native_budget': budget,
                       'evidence': evidence, 'messages': episode.messages, 'policy_packets': packets}
                base.write_json(path, row)
                print(name, result, 'tokens', row['total_tokens'], '/', budget, flush=True)
                return row
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rows = list(pool.map(evaluate, jobs))
            request({'operation': 'close'})
            worker.wait()
            feedback = {}
            for row in rows:
                if not row['result']['valid']:
                    continue
                persona = int(row['episode_id'].split('_', 1)[0][1:])
                source_ids = list(row['evidence']['source_ids'])
                for fid in row['evidence']['fact_ids']:
                    source_ids.extend(c['source_id'] for c in graphs[persona].state['facts'][fid]['citations'])
                feedback[row['episode_id']] = {
                    'source_ids': list(dict.fromkeys(source_ids)),
                    'correct': row['result']['correct'],
                    'origin': 'Evidence selected by one frozen greedy Qwen27B Reader trajectory'}
            base.write_json(args.out / 'reader_feedback.json', feedback)
            base.write_json(args.out / 'summary.json', {
                'queries': len(rows), 'correct': sum(r['result']['correct'] for r in rows),
                'raw_correct': sum(r['raw_correct'] for r in rows),
                'invalid': sum(not r['result']['valid'] for r in rows),
                'model_tokens': sum(r['total_tokens'] for r in rows),
                'raw_model_tokens': sum(r['native_budget'] for r in rows),
                'rerank_document_tokens': sum(r['evidence']['rerank_document_tokens'] for r in rows),
                'rerank_native_input_tokens': sum(r['evidence']['rerank_native_input_tokens'] for r in rows)
                                              if all('rerank_native_input_tokens' in r['evidence'] for r in rows) else None,
                'over_budget': [r['episode_id'] for r in rows if r['total_tokens'] > r['native_budget']],
                'scope': 'Frozen policy, one actual trajectory per question; graph construction and training costs separate'})
            base.write_json(args.out / 'complete.json', {'queries': len(rows), 'worker_exit': worker.returncode})
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
