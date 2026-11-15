"""Test current-graph query feedback with the frozen trained NF4 Writer."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import subprocess

import requests

from graph_environment import GraphEnvironment, base
from train_writer import Evaluator, REMOTE


def main():
    root = base.ROOT / 'results/qwen27b_rsi'
    out = root / 'writer_current_feedback_control'
    assert not out.exists()
    out.mkdir()
    generation = root / 'writer_aligned_before'
    assert base.read(generation / 'complete.json') == {'episodes': 21, 'worker_exit': 0}
    config = base.read(generation / 'configuration.json')
    episode = 'p1_session_50_tla_2'
    index = config['episodes'].index(episode)
    state = base.read(generation / f'step{index}.before.json')
    observation = base.read(generation / f'step{index}.observation.json')
    payload = json.loads(observation[1]['content'])
    source_ids = [s['source_id'] for s in payload['sources']]
    graph = GraphEnvironment(base.Patient(1, store_names=('chunk',)), state)
    stage = 4
    query_ids = base.read(root / 'writer_factual/step3_sample1.reward.json')['query_ids']
    questions = [next(q for q in graph.patient.groups[stage][1] if q['query_id'] == qid) for qid in query_ids]
    evaluator = Evaluator(out / 'answers')
    def baseline_answer(q):
        saved = base.read(root / 'writer_factual/answers' / f'step3_noop_{q["query_id"]}.json')
        return evaluator.answer(graph, q, stage, f'baseline_{q["query_id"]}', paired_answer=saved)
    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline = list(pool.map(baseline_answer, questions))
    current = baseline[0]
    treatment = deepcopy(observation)
    fields = json.loads(treatment[1]['content'])
    fields['feedback']['current_graph_reader'] = {
        'reader': 'fixed QFAD source retrieval plus graph facts',
        'answer_correct': current['score']['correct'], 'generated_answer': current['answer'],
        'selected_source_ids': current['retrieval']['source_ids'],
        'selected_fact_ids': current['retrieval']['fact_ids']}
    treatment[1]['content'] = json.dumps(fields, ensure_ascii=False)
    session = requests.Session()
    session.trust_env = False
    for messages in (observation, treatment):
        response = session.post('http://127.0.0.1:19109/tokenize', json={
            'model': 'macaq', 'messages': messages, 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
        response.raise_for_status()
        assert response.json()['count'] <= 12000
    session.close()
    base.write_json(out / 'configuration.json', {
        'episode': episode, 'query_ids': query_ids, 'checkpoint': config['resume'],
        'writer_model': 'Qwen3.8-27B NF4', 'answer_model': 'Qwen3.8-27B-FP8',
        'initial_graph': str(generation / f'step{index}.before.json'),
        'source_ids': source_ids, 'decoding': 'greedy, seed 0, no thinking',
        'scope': 'Same trained policy and source evidence; only current-graph Reader feedback is added'})
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {REMOTE}/{out.name} --resume {config["resume"]}']
    samples = []
    with (out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Writer probe worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready' and ready['step'] == 26
            base.write_json(out / 'worker_runtime.json', ready)
            for name, messages in [('saved_reader_feedback', observation), ('current_graph_reader_feedback', treatment)]:
                base.write_json(out / f'{name}.observation.json', messages)
                packet = request({'operation': 'infer', 'id': name, 'messages': messages,
                                  'seed': 0, 'max_tokens': 768, 'native_budget': 12768})
                assert packet['event'] == 'samples'
                base.write_json(out / f'{name}.samples.json', packet)
                samples.append((name, packet))
            request({'operation': 'close'})
            worker.wait()
            assert worker.returncode == 0
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()
    results = []
    for name, packet in samples:
        action = None
        try:
            action = graph.parse(packet['rows'][0], source_ids, stage)
        except (ValueError, KeyError, TypeError) as error:
            support = {'supported': False, 'reason': str(error)}
        else:
            support = graph.verify(action, evaluator.judge)
        if name == 'saved_reader_feedback':
            assert action == base.read(generation / f'step{index}.result.json')['action'], 'Frozen control changed'
        result = {'condition': name, 'action': action, 'support': support,
                  'writer_tokens': packet['prompt_tokens'] + len(packet['rows'][0]['token_ids']),
                  'net_correct': None}
        if support['supported']:
            candidate = graph.apply(action, support)
            def answer(pair):
                q, old = pair
                return evaluator.answer(candidate, q, stage, f'{name}_{q["query_id"]}', paired_answer=old)
            with ThreadPoolExecutor(max_workers=2) as pool:
                answers = list(pool.map(answer, zip(questions, baseline)))
            changes = [int(a['score']['correct']) - int(b['score']['correct']) for a, b in zip(answers, baseline)]
            result.update(query_deltas=changes, net_correct=sum(changes))
        results.append(result)
        base.write_json(out / f'{name}.result.json', result)
        print(name, support['supported'], result['net_correct'], flush=True)
    base.write_json(out / 'summary.json', {
        'episode': episode, 'query_ids': query_ids,
        'baseline_correct': [r['score']['correct'] for r in baseline],
        'current_graph_reader_answer': current['answer'],
        'current_feedback_model_tokens': current['usage']['total_tokens'],
        'current_feedback_judge_tokens': current['judge_usage']['total_tokens'],
        'current_feedback_rerank_native_input_tokens': current['retrieval']['rerank_native_input_tokens'],
        'results': results,
        'scope': 'One frozen NF4 policy, identical original evidence and graph. No parameter updates. '
                 'Current feedback uses the actual model answer and correctness, never reference-answer content.'})


if __name__ == '__main__':
    main()
