"""Compare Writer proposals with and without a saved Reader-generated answer."""
from copy import deepcopy
import json
from pathlib import Path

import requests

from graph_environment import GraphEnvironment, base
from train_writer import Evaluator


def main():
    root = base.ROOT / 'results/qwen27b_rsi'
    out = root / 'writer_answer_feedback_control'
    out.mkdir(parents=True, exist_ok=True)
    episode = 'p1_session_50_tla_2'
    generation = root / 'writer_factual_before'
    index = base.read(generation / 'configuration.json')['episodes'].index(episode)
    state = base.read(generation / f'step{index}.before.json')
    observation = base.read(generation / f'step{index}.observation.json')
    assert state == base.read(root / 'writer_factual/step3.before.json')
    reader_path = root / 'reader_factual_before_full/answers' / f'{episode}.json'
    reader_answer = base.read(reader_path)['answer']
    source_ids = [s['source_id'] for s in json.loads(observation[1]['content'])['sources']]
    patient = base.Patient(1, store_names=('chunk',))
    graph = GraphEnvironment(patient, state)
    stage = 4
    query_ids = base.read(root / 'writer_factual/step3_sample1.reward.json')['query_ids']
    questions = [next(q for q in patient.groups[stage][1] if q['query_id'] == qid)
                 for qid in query_ids]
    evaluator = Evaluator(out / 'answers')
    baseline = []
    for q in questions:
        saved = base.read(root / 'writer_factual/answers' / f'step3_noop_{q["query_id"]}.json')
        row = evaluator.answer(graph, q, stage, f'baseline_{q["query_id"]}', paired_answer=saved)
        assert row.get('paired_reuse'), 'The saved three-question control must match exactly'
        baseline.append(row)
    base.write_json(out / 'configuration.json', {
        'episode': episode, 'query_ids': query_ids, 'graph': str(generation / f'step{index}.before.json'),
        'observation': str(generation / f'step{index}.observation.json'),
        'reader_answer_origin': str(reader_path), 'reader_answer_field': 'answer',
        'writer_model': 'Qwen3.8-27B-FP8', 'answer_model': 'Qwen3.8-27B-FP8',
        'writer_decoding': 'temperature 0, seed 0, no thinking', 'writer_max_tokens': 768,
        'judge': 'gpt-6.1-sol high',
        'scope': 'Feedback-content control with a frozen FP8 Writer; no policy training or deployed graph change'})
    session = requests.Session()
    session.trust_env = False
    results = []
    for name in ('original_feedback', 'reader_generated_answer'):
        messages = deepcopy(observation)
        if name == 'reader_generated_answer':
            payload = json.loads(messages[1]['content'])
            payload['feedback']['reader_generated_answer'] = reader_answer
            messages[1]['content'] = json.dumps(payload, ensure_ascii=False)
        count = session.post('http://127.0.0.1:19109/tokenize', json={
            'model': 'macaq', 'messages': messages, 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
        count.raise_for_status()
        assert count.json()['count'] <= 12000
        base.write_json(out / f'{name}.observation.json', messages)
        response = evaluator.client.chat.completions.create(model='macaq', messages=messages,
            temperature=0, seed=0, max_tokens=768,
            extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        base.write_json(out / f'{name}.response.json', response.model_dump())
        choice = response.choices[0]
        sample = {'text': choice.message.content or '', 'finished': choice.finish_reason == 'stop'}
        action = None
        try:
            action = graph.parse(sample, source_ids, stage)
        except (ValueError, KeyError, TypeError) as error:
            support = {'supported': False, 'reason': str(error)}
        else:
            support = graph.verify(action, evaluator.judge)
        result = {'condition': name, 'action': action, 'support': support,
                  'writer_usage': response.usage.model_dump(), 'net_correct': None}
        if support['supported']:
            candidate = graph.apply(action, support)
            answers = [evaluator.answer(candidate, q, stage, f'{name}_{q["query_id"]}',
                                       paired_answer=before) for q, before in zip(questions, baseline)]
            changes = [int(a['score']['correct']) - int(b['score']['correct'])
                       for a, b in zip(answers, baseline)]
            result.update(query_deltas=changes, net_correct=sum(changes))
        results.append(result)
        base.write_json(out / f'{name}.result.json', result)
        print(name, support['supported'], result['net_correct'], flush=True)
    session.close()
    base.write_json(out / 'summary.json', {
        'episode': episode, 'query_ids': query_ids, 'reader_generated_answer': reader_answer,
        'baseline_correct': [r['score']['correct'] for r in baseline], 'results': results,
        'scope': 'One greedy proposal per condition; both use the same FP8 Writer and original sources. '
                 'The added answer is an existing model output, not a reference answer. '
                 'This control does not establish an RL gain or a full-dataset improvement.'})


if __name__ == '__main__':
    main()
