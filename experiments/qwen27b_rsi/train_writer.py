"""Train atomic Qwen27B writes against paired, fixed-Reader query outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import requests

from openai import OpenAI

from graph_environment import GraphEnvironment, QueryBudgetExceeded, base
from medmemory_scoring import JudgeClient
from semantic_query_writer_rl import RUBRIC
from writer_feedback import bounded_observation, observed_feedback

ROOT = base.ROOT
REMOTE = '/home/administrator/du/memos/qwen27b_rsi'


class Evaluator:
    def __init__(self, directory):
        self.directory = directory
        self.client = OpenAI(api_key='local', base_url='http://127.0.0.1:19109/v1', timeout=900)
        self.judge = JudgeClient('gpt-6.1-sol')
        self.cache = {}
        self.lock = threading.Lock()

    @staticmethod
    def context_cap(graph, question, stage):
        checkpoint = 10 * (stage + 1)
        raw = next(r for r in base.read(base.RAW / f'persona_{graph.patient.persona:02d}' /
            f'session_{checkpoint:03d}_answers.json') if r['query_id'] == question['query_id'])
        return min(8000, len(base.ENC.encode(raw['formatted_context'])))

    def answer(self, graph, question, stage, label, evidence=None, native_budget=None, paired_answer=None):
        retrieved = evidence if evidence is not None else graph.retrieve(
            question['question'], stage, context_cap=self.context_cap(graph, question, stage))
        context = '\n\n'.join(retrieved['documents'])
        prompt = base.ANSWER_PROMPT.format(context=context,
            question=base.PROMPT.format_query(question['question'], question['query_type']))
        max_tokens = 16384
        if native_budget is not None:
            session = requests.Session()
            session.trust_env = False
            tokenized = session.post('http://127.0.0.1:19109/tokenize', json={
                'model': 'macaq', 'messages': [{'role': 'user', 'content': prompt}],
                'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
            tokenized.raise_for_status()
            max_tokens = min(max_tokens, native_budget - tokenized.json()['count'])
            if max_tokens <= 0:
                raise QueryBudgetExceeded('Policy and evidence exhausted the native answer-token budget')
        if (paired_answer is not None and paired_answer['prompt'] == prompt
                and paired_answer.get('max_completion_tokens', 16384) == max_tokens):
            result = {**paired_answer, 'retrieval': retrieved, 'answer_cache_hit': True, 'paired_reuse': True}
            base.write_json(self.directory / f'{label}.json', result)
            return result
        key = (graph.patient.persona, question['query_id'], prompt, max_tokens)
        # Exact prompts share the same realization in this paired experiment.
        with self.lock:
            pending = self.cache.get(key)
            if pending is None:
                from concurrent.futures import Future
                pending = Future()
                self.cache[key] = pending
                owner = True
            else:
                owner = False
        if owner:
            try:
                response = self.client.chat.completions.create(model='macaq',
                    messages=[{'role': 'user', 'content': prompt}], temperature=0, seed=0,
                    max_tokens=max_tokens, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                text = response.choices[0].message.content or ''
                matches = list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*', text, re.I))
                answer = text[matches[-1].end():].strip() if matches else text.strip()
                payload = {'question': question['question'], 'question_type': question['query_type'],
                    'reference_answers': [a['content'] for a in question['answers'] if a.get('is_correct')],
                    'response': answer}
                verdict_response = self.judge.chat(messages=[{'role': 'system', 'content': RUBRIC},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
                verdict = json.loads(verdict_response.content)
                assert type(verdict['correct']) is bool
                pending.set_result({'answer': answer, 'response': text, 'score': verdict,
                    'usage': response.usage.model_dump(), 'judge_usage': verdict_response.usage,
                    'finish_reason': response.choices[0].finish_reason})
            except BaseException as error:
                pending.set_exception(error)
                raise
        result = {'query_id': question['query_id'], 'prompt': prompt, 'retrieval': retrieved,
                  'max_completion_tokens': max_tokens,
                  **pending.result(), 'answer_cache_hit': not owner}
        base.write_json(self.directory / f'{label}.json', result)
        return result


def group_advantages(results, question_count, rejected_cost_terms=None):
    rewards = []
    for index, result in enumerate(results):
        if not result['supported']:
            rewards.append(float(-question_count - 1) if rejected_cost_terms is None
                           else .01 * rejected_cost_terms[index])
        else:
            rewards.append(float(result['net_correct']) + .01 * result['cost_term'])
    # The explicit no-write control is a baseline, never a fabricated policy sample.
    anchor = rewards + [0.]
    mean = sum(anchor) / len(anchor)
    std = (sum((r - mean) ** 2 for r in anchor) / len(anchor)) ** .5
    advantages = [r / max(std, 1.) for r in rewards]
    assert all(a <= 0 for a, result in zip(advantages, results) if not result['supported'])
    assert all(a <= 0 for a, result in zip(advantages, results)
               if result['supported'] and result['net_correct'] < 0)
    return rewards, advantages


def reader_cost_term(baseline, answers, write_tokens):
    old_cost = sum(a['online_model_tokens'] for a in baseline)
    new_cost = sum(a['online_model_tokens'] for a in answers) + write_tokens
    savings = -write_tokens
    for before, after in zip(baseline, answers):
        difference = before['online_model_tokens'] - after['online_model_tokens']
        retained_correct = before['score']['correct'] and after['score']['correct']
        savings += difference if retained_correct else min(0, difference)
    term = savings / max(1, old_cost, new_cost)
    if any(b['score']['correct'] and not a['score']['correct'] for b, a in zip(baseline, answers)) \
            or any(not a['reader_valid'] for a in answers):
        term = min(0., term)
    return term, {'before': old_cost, 'after_including_marginal_write': new_cost,
                  'credited_savings': savings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--resume', help='Remote policy checkpoint path')
    parser.add_argument('--evidence-feedback', type=Path, help='Verified dialogue or observed Reader evidence for training')
    parser.add_argument('--replay-run', type=Path, help='Restore each episode from its saved pre-write graph')
    parser.add_argument('--reader-checkpoint', help='Remote Qwen27B Reader snapshot, fixed throughout this Writer phase')
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--rejected-action-reward', choices=('penalty', 'no-change'), default='penalty')
    parser.add_argument('--aligned-reader-feedback', action='store_true')
    parser.add_argument('--current-reader-feedback', action='store_true')
    parser.add_argument('--retention-run', type=Path, help='Complete fixed-Reader evaluation supplying previously correct query IDs')
    args = parser.parse_args()
    if args.current_reader_feedback:
        assert args.aligned_reader_feedback and args.replay_run and not args.reader_checkpoint
        assert base.read(args.replay_run / 'configuration.json')['current_reader_feedback'] is True
    args.out.mkdir(parents=True, exist_ok=True)
    state = base.read(args.state) if args.state else {}
    assert 0 <= args.start_index < len(args.episodes)
    if args.start_index:
        assert args.resume
        state = base.read(args.out / 'graph_state.json')
    evidence_feedback = base.read(args.evidence_feedback) if args.evidence_feedback else {}
    if args.aligned_reader_feedback:
        assert all('generated_answer' in evidence_feedback[e] for e in args.episodes)
    retention = {1: set(), 2: set()}
    if args.retention_run:
        assert base.read(args.retention_run / 'summary.json')['queries'] == 197
        assert base.read(args.retention_run / 'graph_snapshot.json') == (base.read(args.state) if args.state else {})
        for path in (args.retention_run / 'answers').glob('*.json'):
            persona, qid = path.stem.split('_', 1)
            if base.read(path)['score']['correct']:
                retention[int(persona[1:])].add(qid)
    replay_indices = {}
    if args.replay_run:
        for index, episode in enumerate(base.read(args.replay_run / 'configuration.json')['episodes']):
            replay_indices.setdefault(episode, index)
        for episode in args.episodes:
            if episode not in replay_indices:
                raise ValueError(f'No pre-write snapshot for {episode}')
    graphs = {}
    for episode in args.episodes:
        persona = int(episode.split('_')[0][1:])
        if persona not in graphs:
            graphs[persona] = GraphEnvironment(base.Patient(persona, store_names=('chunk',)), state.get(str(persona)))
    config = {'episodes': args.episodes, 'model': 'Qwen3.8-27B',
              'answer_weights': '/home/administrator/du/model/Qwen/Qwen3.8-27B-FP8',
              'answer_endpoint': '9009/macaq', 'answer_max_context': 32768,
              'policy_quantization': 'NF4', 'reader': 'fixed QFAD source retrieval plus graph facts',
              'initial_graph': 'raw source nodes; no generated facts from other models',
              'reward': 'source-gated paired semantic QA delta; small write-cost tiebreak',
              'advantage_baseline': 'explicit NOOP reward; group standard deviation sets scale only',
              'advantage_scale': 'max(group reward standard deviation, 1); preserve small cost-only updates',
              'evidence_coverage_auxiliary': False, 'semantic_rubric': RUBRIC,
              'scope': 'Writer phase with fixed Reader and answerer during comparisons',
              'graph_input': str(args.state) if args.state else None,
              'sample_count': 4, 'max_action_tokens': 768, 'related_queries': 2,
              'max_observed_sources': 6, 'max_policy_input_tokens': 12000,
              'update_epochs': args.update_epochs,
              'context_cap': 8000, 'answer_max_tokens': 16384, 'seed': 17}
    config['evidence_feedback'] = str(args.evidence_feedback) if args.evidence_feedback else None
    config['replay_run'] = str(args.replay_run) if args.replay_run else None
    config['state_sampling'] = 'saved pre-write snapshot on each occurrence' if args.replay_run else 'current graph'
    config['feedback_origins'] = sorted({r['origin'] for r in evidence_feedback.values()})
    config['reader_checkpoint'] = args.reader_checkpoint
    if args.rejected_action_reward == 'no-change':
        config['rejected_action_reward'] = 'Rejected actions leave the graph unchanged; zero QA delta minus normalized action-token cost'
    if args.aligned_reader_feedback:
        config['feedback_protocol'] = 'reader-generated-answer'
    if args.current_reader_feedback:
        config['current_reader_feedback'] = True
    if args.retention_run:
        config['retention_run'] = str(args.retention_run)
        config['retention_queries'] = 2
        config['retention_selection'] = 'Two most similar previously correct queries at or before the trigger stage, excluding trigger and related queries'
    if args.reader_checkpoint:
        from reader_environment import READER_PROTOCOL
        config['reader_protocol'] = READER_PROTOCOL
        config['reader'] = 'Frozen Qwen27B retrieval-policy snapshot with QFAD tools'
        config['reward'] = 'Source-gated paired QA delta; bounded Reader model-token savings including marginal write tokens'
    if args.start_index:
        assert base.read(args.out / 'configuration.json') == config
        for index in range(args.start_index):
            assert base.read(args.out / f'step{index}.update.json')['event'] == 'updated'
        base.write_json(args.out / 'resume_state.json', {
            'checkpoint': args.resume, 'next_episode_index': args.start_index})
    else:
        base.write_json(args.out / 'configuration.json', config)
    remote_out = f'{REMOTE}/{args.out.name}'
    command = ['ssh', '-T', 'win',
        f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '
        f'/home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{REMOTE}/policy_worker.py --model /home/administrator/du/model/Qwen/Qwen3.8-27B --out {remote_out} '
        f'--update-epochs {args.update_epochs}']
    if args.resume:
        command[-1] += ' --resume ' + args.resume
    with (args.out / 'worker.log').open('a' if args.start_index else 'w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(f'Policy worker ended: {worker.poll()}; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        ready = receive()
        assert ready['event'] == 'ready', ready
        if args.start_index:
            assert ready['step'] == base.read(args.out / f'step{args.start_index - 1}.update.json')['step']
        base.write_json(args.out / ('resume_runtime.json' if args.start_index else 'worker_runtime.json'), ready)
        print('policy ready', ready, flush=True)
        evaluator = Evaluator(args.out / 'answers')
        if args.reader_checkpoint:
            from policy_reader_evaluator import PolicyReaderEvaluator
            frozen = request({'operation': 'freeze_reader', 'checkpoint': args.reader_checkpoint})
            assert frozen['event'] == 'reader_frozen', frozen
            base.write_json(args.out / 'reader_snapshot.json', frozen)
            evaluator = PolicyReaderEvaluator(evaluator, request)
        try:
            for index, episode in enumerate(args.episodes):
                if index < args.start_index:
                    continue
                persona, qid = int(episode.split('_')[0][1:]), episode.split('_', 1)[1]
                graph = graphs[persona]
                if args.replay_run:
                    snapshot = base.read(args.replay_run / f'step{replay_indices[episode]}.before.json')
                    graph = GraphEnvironment(graph.patient, snapshot, graph.embeddings)
                stage = int(qid.split('_')[1]) // 10 - 1
                prefix = f'step{index}'
                pending_rewards = args.out / f'{prefix}.rewards.json'
                if args.start_index and index == args.start_index and pending_rewards.exists():
                    assert not (args.out / f'{prefix}.update.json').exists()
                    assert graph.state == base.read(args.out / f'{prefix}.before.json')
                    observation = base.read(args.out / f'{prefix}.observation.json')
                    samples = base.read(args.out / f'{prefix}.samples.json')
                    saved = base.read(pending_rewards)
                    results, rewards, advantages = saved['results'], saved['rewards'], saved['advantages']
                    assert len(results) == len(rewards) == len(advantages) == len(samples['rows'])
                    restored = request({'operation': 'restore_samples', 'id': prefix,
                                        'messages': observation, 'samples': samples})
                    assert restored['event'] == 'samples_restored'
                    evaluated = [(result, graph.apply(result['action'],
                        base.read(args.out / f'{prefix}_sample{i}.support.json'))
                        if result['supported'] else None) for i, result in enumerate(results)]
                    print('restored sampled actions and rewards', episode, flush=True)
                else:
                    questions = graph.patient.groups[stage][1]
                    trigger = next(q for q in questions if q['query_id'] == qid)
                    others = [q for q in questions if q['query_id'] != qid]
                    vectors = graph.embed([trigger['question']] + [q['question'] for q in others])
                    order = sorted(range(len(others)), key=lambda i: float(vectors[i + 1] @ vectors[0]), reverse=True)
                    queries = [trigger] + [others[i] for i in order[:2]]
                    if args.retention_run:
                        included = {q['query_id'] for q in queries}
                        anchors = [q for t, (_, qs) in enumerate(graph.patient.groups) if t <= stage
                                   for q in qs if q['query_id'] in retention[persona] and q['query_id'] not in included]
                        if anchors:
                            vectors = graph.embed([trigger['question']] + [q['question'] for q in anchors])
                            indices = sorted(range(len(anchors)), key=lambda i: float(vectors[i + 1] @ vectors[0]), reverse=True)
                            queries.extend(anchors[i] for i in indices[:2])
                    query_stage = lambda q: int(q['query_id'].split('_')[1]) // 10 - 1
                    prefix = f'step{index}'
                    def baseline_answer(q):
                        paired = None
                        if args.current_reader_feedback and q['query_id'] == qid:
                            paired = base.read(args.replay_run / 'current_answers' / f'{episode}.json')
                        answer = evaluator.answer(graph, q, query_stage(q),
                            f'{prefix}_noop_{q["query_id"]}', paired_answer=paired)
                        if paired is not None:
                            assert answer.get('paired_reuse'), 'Current Reader feedback must match its saved pre-write state'
                        return answer
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        baseline = list(pool.map(baseline_answer, queries))
                    if args.retention_run:
                        base.write_json(args.out / f'{prefix}.query_group.json', [
                            {'query_id': q['query_id'], 'stage': query_stage(q),
                             'role': 'trigger' if i == 0 else 'related' if i < 3 else 'retention',
                             'current_graph_correct': answer['score']['correct']}
                            for i, (q, answer) in enumerate(zip(queries, baseline))])
                    source_ids = baseline[0]['retrieval'].get('candidate_source_ids',
                                                            baseline[0]['retrieval']['source_ids'])[:2]
                    if not source_ids:
                        source_ids = list(dict.fromkeys(c['source_id']
                            for fid in baseline[0]['retrieval'].get('fact_ids', [])
                            for c in graph.state['facts'][fid]['citations']))[:2]
                    feedback = {'current_graph_reader_correct': baseline[0]['score']['correct']}
                    if episode in evidence_feedback and evidence_feedback[episode]['source_ids']:
                        source_ids = evidence_feedback[episode]['source_ids'][:6]
                        feedback['evidence_control_answer_correct'] = evidence_feedback[episode]['correct']
                        if 'selected_source_ids' in evidence_feedback[episode]:
                            feedback['reader_selected_source_ids'] = evidence_feedback[episode]['selected_source_ids']
                            feedback['reader_valid'] = evidence_feedback[episode]['valid']
                    if args.aligned_reader_feedback:
                        feedback = observed_feedback(evidence_feedback[episode],
                                                     baseline[0] if args.current_reader_feedback else None)
                    source_ids, observation = bounded_observation(graph, trigger['question'], source_ids, stage, feedback)
                    base.write_json(args.out / f'{prefix}.observation.json', observation)
                    base.write_json(args.out / f'{prefix}.before.json', graph.state)
                    print('sampling', episode, 'baseline', [r['score']['correct'] for r in baseline], flush=True)
                    samples = request({'operation': 'sample', 'id': prefix, 'messages': observation,
                        'seed': 17 + index, 'count': 4, 'max_tokens': 768})
                    assert samples['event'] == 'samples', samples
                    base.write_json(args.out / f'{prefix}.samples.json', samples)
                    def evaluate(pair):
                        i, row = pair
                        name = f'{prefix}_sample{i}'
                        try:
                            action = graph.parse(row, source_ids, stage)
                        except (ValueError, KeyError, TypeError) as error:
                            result = {'supported': False, 'reason': str(error), 'stage': 'parse'}
                            base.write_json(args.out / f'{name}.reward.json', result)
                            return result, None
                        verdict = graph.verify(action, evaluator.judge)
                        base.write_json(args.out / f'{name}.support.json', verdict)
                        if not verdict['supported']:
                            result = {'supported': False, 'reason': verdict.get('reason'), 'stage': 'support', 'action': action}
                            base.write_json(args.out / f'{name}.reward.json', result)
                            return result, None
                        candidate = graph.apply(action, verdict)
                        answers = [evaluator.answer(candidate, q, query_stage(q), f'{name}_{q["query_id"]}') for q in queries]
                        changes = [int(a['score']['correct']) - int(b['score']['correct']) for a, b in zip(answers, baseline)]
                        write_tokens = len(row['token_ids']) if action['operation'] != 'NOOP' else 0
                        cost_term = -min(write_tokens / 768, 1.)
                        reader_costs = None
                        if args.reader_checkpoint:
                            cost_term, reader_costs = reader_cost_term(baseline, answers, write_tokens)
                        result = {'supported': True, 'action': action, 'net_correct': sum(changes),
                            'query_deltas': changes, 'query_ids': [q['query_id'] for q in queries],
                            'write_tokens': write_tokens, 'cost_term': cost_term, 'reader_costs': reader_costs}
                        base.write_json(args.out / f'{name}.reward.json', result)
                        return result, candidate
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        evaluated = list(pool.map(evaluate, enumerate(samples['rows'])))
                    results = [r for r, _ in evaluated]
                    rejected_cost_terms = [-min(len(row['token_ids']) / 768, 1.) for row in samples['rows']] \
                        if args.rejected_action_reward == 'no-change' else None
                    rewards, advantages = group_advantages(results, len(queries), rejected_cost_terms)
                    base.write_json(args.out / f'{prefix}.rewards.json', {
                        'results': results, 'rewards': rewards, 'advantages': advantages,
                        **({'rejected_cost_terms': rejected_cost_terms} if rejected_cost_terms is not None else {})})
                print('updating', episode, rewards, flush=True)
                update = request({'operation': 'update', 'id': prefix, 'advantages': advantages, 'rewards': rewards})
                assert update['event'] == 'updated', update
                base.write_json(args.out / f'{prefix}.update.json', update)
                eligible = [(i, r) for i, r in enumerate(results) if r['supported'] and rewards[i] > 0
                            and (r['net_correct'] > 0 or all(delta == 0 for delta in r['query_deltas']))]
                accepted = max(eligible, key=lambda pair: rewards[pair[0]])[0] if eligible else None
                if accepted is not None:
                    graphs[persona] = evaluated[accepted][1]
                base.write_json(args.out / f'{prefix}.selection.json', {'accepted_sample': accepted,
                    'criterion': 'source support and positive paired QA gain, or unchanged correctness with lower Reader model-token cost'})
                base.write_json(args.out / 'graph_state.json', {str(p): g.state for p, g in graphs.items()})
                print('updated', episode, 'delta', update['adapter_delta_l2'], 'accepted', accepted, flush=True)
            request({'operation': 'close'})
            worker.wait()
            base.write_json(args.out / 'complete.json', {'steps': len(args.episodes), 'worker_exit': worker.returncode})
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
