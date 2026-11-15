"""Build a graph sequentially with one frozen Writer action per observed Reader feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from graph_environment import GraphEnvironment, base
from medmemory_scoring import JudgeClient
from train_writer import Evaluator, REMOTE
from writer_feedback import from_trajectory, bounded_observation, observed_feedback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reader-run', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--resume', help='Remote checkpoint; omit for the initial Qwen27B policy')
    parser.add_argument('--include-reader-answer', action='store_true')
    parser.add_argument('--current-reader-feedback', action='store_true')
    parser.add_argument('--hide-answer-labels', action='store_true',
                        help='Withhold correctness labels from frozen Writer transfer evaluation')
    args = parser.parse_args()
    if args.current_reader_feedback:
        assert args.include_reader_answer
        assert not args.hide_answer_labels
    args.out.mkdir(parents=True, exist_ok=True)
    state = base.read(args.state)
    feedback_graph = base.read(args.reader_run / 'graph_snapshot.json')
    assert state == feedback_graph, 'Reader feedback must come from the specified initial graph'
    assert base.read(args.reader_run / 'complete.json')['worker_exit'] == 0
    episodes = list(dict.fromkeys(args.episodes))
    episodes.sort(key=lambda e: (int(e.split('_')[0][1:]), int(e.split('_')[2]), e))
    graphs = {p: GraphEnvironment(base.Patient(p, store_names=('chunk',)), state.get(str(p)))
              for p in {int(e.split('_')[0][1:]) for e in episodes}}
    feedback = {}
    for episode in episodes:
        trajectory = base.read(args.reader_run / 'trajectories' / f'{episode}.json')
        persona = int(episode.split('_', 1)[0][1:])
        sources = from_trajectory(trajectory, state[str(persona)]['facts'])
        feedback[episode] = {
            **sources,
            'reader_answer_correct': trajectory['result']['correct'],
            'reader_valid': trajectory['result']['valid'],
            'model_tokens': trajectory['total_tokens'],
            'rerank_native_input_tokens': trajectory['evidence']['rerank_native_input_tokens']}
        if args.include_reader_answer:
            answer_path = args.reader_run / 'answers' / f'{episode}.json'
            feedback[episode]['generated_answer'] = base.read(answer_path)['answer'] if answer_path.exists() else None
        if args.hide_answer_labels:
            feedback[episode]['correct'] = None
            feedback[episode]['reader_answer_correct'] = None
    base.write_json(args.out / 'initial_graph.json', state)
    base.write_json(args.out / 'configuration.json', {
        'process_id': os.getpid(),
        'episodes': episodes, 'reader_run': str(args.reader_run.resolve()),
        'state': str(args.state.resolve()), 'resume': args.resume,
        'policy_model': 'Qwen3.8-27B', 'policy_frozen': True, 'decoding': 'greedy',
        'max_action_tokens': 768, 'max_observed_sources': 6, 'max_policy_input_tokens': 12000,
        'graph_acceptance': 'Source support only; one generated action per feedback; no downstream QA selection',
        'feedback_origin': 'Single frozen greedy Reader trajectories on the initial graph',
        'feedback_model_tokens': sum(row['model_tokens'] for row in feedback.values()),
        'feedback_rerank_native_input_tokens': sum(row['rerank_native_input_tokens'] for row in feedback.values())})
    if args.include_reader_answer:
        config = base.read(args.out / 'configuration.json')
        config['feedback_protocol'] = 'reader-generated-answer'
        base.write_json(args.out / 'configuration.json', config)
    if args.hide_answer_labels:
        config = base.read(args.out / 'configuration.json')
        config['correctness_feedback'] = 'withheld; no new-patient answer labels enter Writer observations'
        base.write_json(args.out / 'configuration.json', config)
    if args.current_reader_feedback:
        config = base.read(args.out / 'configuration.json')
        config['current_reader_feedback'] = True
        base.write_json(args.out / 'configuration.json', config)
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {REMOTE}/{args.out.name}']
    if args.resume:
        command[-1] += ' --resume ' + args.resume
    judge = JudgeClient('gpt-6.1-sol')
    current_evaluator = Evaluator(args.out / 'current_answers') if args.current_reader_feedback else None
    results = []
    with (args.out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Writer generation worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready', ready
            base.write_json(args.out / 'worker_runtime.json', ready)
            for index, episode in enumerate(episodes):
                persona, qid = int(episode.split('_')[0][1:]), episode.split('_', 1)[1]
                stage = int(qid.split('_')[1]) // 10 - 1
                graph = graphs[persona]
                question = next(q for q in graph.patient.groups[stage][1] if q['query_id'] == qid)
                row = feedback[episode]
                current = current_evaluator.answer(graph, question, stage, episode) if current_evaluator else None
                source_ids, observation = bounded_observation(graph, question['question'], row['source_ids'], stage,
                                                              observed_feedback(row, current))
                base.write_json(args.out / f'step{index}.before.json', graph.state)
                base.write_json(args.out / f'step{index}.observation.json', observation)
                packet = request({'operation': 'infer', 'id': f'step{index}', 'messages': observation,
                                  'seed': 0, 'max_tokens': 768, 'native_budget': 12768})
                base.write_json(args.out / f'step{index}.samples.json', packet)
                action = None
                writer_tokens = 0
                if packet['event'] == 'budget_exhausted':
                    verdict = {'supported': False, 'reason': 'Writer observation exceeds policy input budget'}
                else:
                    writer_tokens = packet['prompt_tokens'] + len(packet['rows'][0]['token_ids'])
                    try:
                        action = graph.parse(packet['rows'][0], source_ids, stage)
                    except (ValueError, KeyError, TypeError) as error:
                        verdict = {'supported': False, 'reason': str(error)}
                    else:
                        verdict = graph.verify(action, judge)
                        if verdict['supported']:
                            graphs[persona] = graph.apply(action, verdict)
                result = {'episode': episode, 'action': action, 'support': verdict,
                          'writer_tokens': writer_tokens,
                          'graph_changed': graphs[persona].state != graph.state,
                          'feedback': row, 'observed_source_ids': source_ids}
                if current is not None:
                    result['current_reader_cost'] = {
                        'model_tokens': current['usage']['total_tokens'],
                        'judge_tokens': current['judge_usage']['total_tokens'],
                        'rerank_native_input_tokens': current['retrieval']['rerank_native_input_tokens']}
                results.append(result)
                base.write_json(args.out / f'step{index}.result.json', result)
                base.write_json(args.out / 'graph_state.json',
                                {**state, **{str(p): g.state for p, g in graphs.items()}})
                print(episode, action['operation'] if action else None, verdict['supported'], flush=True)
            request({'operation': 'close'})
            worker.wait()
            if args.current_reader_feedback:
                config = base.read(args.out / 'configuration.json')
                for field in ('model_tokens', 'judge_tokens', 'rerank_native_input_tokens'):
                    config['current_reader_' + field] = sum(r['current_reader_cost'][field] for r in results)
                base.write_json(args.out / 'configuration.json', config)
            base.write_json(args.out / 'summary.json', {
                'episodes': len(results), 'changed': sum(r['graph_changed'] for r in results),
                'unsupported_or_invalid': sum(not r['support']['supported'] for r in results),
                'writer_tokens': sum(r['writer_tokens'] for r in results),
                'feedback_model_tokens': sum(row['model_tokens'] for row in feedback.values()),
                'results': results,
                'scope': 'Sequential frozen Writer generation from actual Reader feedback; source-gated acceptance'})
            base.write_json(args.out / 'complete.json', {'episodes': len(results), 'worker_exit': worker.returncode})
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
