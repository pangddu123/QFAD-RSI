"""Generate paired frozen FP8 Writer graphs from matched Reader source pools."""
import json
import os

from openai import OpenAI
import requests

from graph_environment import GraphEnvironment, base
from medmemory_scoring import JudgeClient
from writer_feedback import from_trajectory


def main():
    root = base.ROOT / 'results/qwen27b_rsi'
    reader_run = root / 'reader_factual_before_full'
    state = base.read(reader_run / 'graph_snapshot.json')
    assert state == base.read(root / 'writer_greedy_after_selection/graph_state.json')
    assert base.read(reader_run / 'complete.json')['queries'] == 197
    episodes = base.read(root / 'writer_factual_before/configuration.json')['episodes']
    variants = ('original', 'answer')
    directories = {v: root / f'writer_feedback_fp8_{v}' for v in variants}
    for path in directories.values():
        assert not path.exists(), path
        path.mkdir()
    patients = {p: base.Patient(p, store_names=('chunk',)) for p in (1, 2)}
    graphs = {v: {p: GraphEnvironment(patient, state[str(p)]) for p, patient in patients.items()}
              for v in variants}
    feedback = {}
    for episode in episodes:
        trajectory = base.read(reader_run / 'trajectories' / f'{episode}.json')
        persona = episode.split('_', 1)[0][1:]
        answer_path = reader_run / 'answers' / f'{episode}.json'
        answer = base.read(answer_path)['answer'] if answer_path.exists() else None
        feedback[episode] = {**from_trajectory(trajectory, state[persona]['facts']),
            'generated_answer': answer, 'model_tokens': trajectory['total_tokens'],
            'rerank_tokens': trajectory['evidence']['rerank_native_input_tokens']}
    for variant, path in directories.items():
        base.write_json(path / 'initial_graph.json', state)
        base.write_json(path / 'configuration.json', {
            'process_id': os.getpid(), 'episodes': episodes, 'reader_run': str(reader_run),
            'policy_model': 'Qwen3.8-27B-FP8', 'policy_frozen': True,
            'decoding': 'temperature 0, seed 0, no thinking', 'max_action_tokens': 768,
            'max_policy_input_tokens': 12000, 'include_reader_generated_answer': variant == 'answer',
            'source_selection': 'Same source IDs in both conditions at each episode; trim trailing sources until both inputs fit',
            'feedback_model_tokens': sum(row['model_tokens'] for row in feedback.values()),
            'feedback_rerank_native_input_tokens': sum(row['rerank_tokens'] for row in feedback.values()),
            'graph_acceptance': 'One greedy action per feedback; source support only; no QA-based selection'})
    client = OpenAI(api_key='local', base_url='http://127.0.0.1:19109/v1', timeout=900)
    judge = JudgeClient('gpt-6.1-sol')
    session = requests.Session()
    session.trust_env = False
    results = {v: [] for v in variants}
    for index, episode in enumerate(episodes):
        persona, qid = int(episode.split('_')[0][1:]), episode.split('_', 1)[1]
        stage = int(qid.split('_')[1]) // 10 - 1
        question = next(q for q in patients[persona].groups[stage][1] if q['query_id'] == qid)
        row = feedback[episode]
        sources = list(row['source_ids'])
        while True:
            observations, counts = {}, {}
            for variant in variants:
                flags = {'reader_answer_correct': row['correct'], 'reader_valid': row['valid'],
                         'reader_selected_source_ids': row['selected_source_ids'],
                         'feedback_graph': 'The fixed initial graph before this sequence of writes'}
                if variant == 'answer' and row['generated_answer'] is not None:
                    flags['reader_generated_answer'] = row['generated_answer']
                observations[variant] = graphs[variant][persona].observation(
                    question['question'], sources, stage, flags)
                response = session.post('http://127.0.0.1:19109/tokenize', json={
                    'model': 'macaq', 'messages': observations[variant], 'add_generation_prompt': True,
                    'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
                response.raise_for_status()
                counts[variant] = response.json()['count']
            if max(counts.values()) <= 12000:
                break
            if not sources:
                raise ValueError(f'Feedback exceeds the policy input budget: {episode}')
            sources.pop()
        for variant in variants:
            path = directories[variant]
            graph = graphs[variant][persona]
            base.write_json(path / f'step{index}.before.json', graph.state)
            base.write_json(path / f'step{index}.observation.json', observations[variant])
            response = client.chat.completions.create(model='macaq', messages=observations[variant],
                temperature=0, seed=0, max_tokens=768,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            base.write_json(path / f'step{index}.response.json', response.model_dump())
            choice = response.choices[0]
            sample = {'text': choice.message.content or '', 'finished': choice.finish_reason == 'stop'}
            action = None
            try:
                action = graph.parse(sample, sources, stage)
            except (ValueError, KeyError, TypeError) as error:
                support = {'supported': False, 'reason': str(error)}
            else:
                support = graph.verify(action, judge)
                if support['supported']:
                    graphs[variant][persona] = graph.apply(action, support)
            result = {'episode': episode, 'action': action, 'support': support,
                      'writer_tokens': response.usage.total_tokens,
                      'graph_changed': graphs[variant][persona].state != graph.state,
                      'observed_source_ids': sources, 'prompt_tokens': counts[variant]}
            results[variant].append(result)
            base.write_json(path / f'step{index}.result.json', result)
            base.write_json(path / 'graph_state.json',
                            {str(p): g.state for p, g in graphs[variant].items()})
            print(variant, episode, action['operation'] if action else None, support['supported'], flush=True)
    session.close()
    for variant, path in directories.items():
        rows = results[variant]
        base.write_json(path / 'summary.json', {
            'episodes': len(rows), 'changed': sum(r['graph_changed'] for r in rows),
            'unsupported_or_invalid': sum(not r['support']['supported'] for r in rows),
            'writer_tokens': sum(r['writer_tokens'] for r in rows), 'results': rows})
        base.write_json(path / 'complete.json', {'episodes': len(rows), 'api_generation_complete': True})


if __name__ == '__main__':
    main()
