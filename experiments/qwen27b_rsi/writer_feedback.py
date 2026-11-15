"""Expose selected and unselected Reader candidates to source-grounded writes."""
import argparse
import json
from pathlib import Path

import requests


def from_trajectory(trajectory, facts):
    selected = list(trajectory['evidence']['source_ids'])
    for fid in trajectory['evidence']['fact_ids']:
        selected.extend(c['source_id'] for c in facts[fid]['citations'])
    selected = list(dict.fromkeys(selected))
    observed = []
    for message in reversed(trajectory['messages'][2:]):
        if message['role'] != 'user':
            continue
        observation = json.loads(message['content'])['tool_observation']
        observed.extend(s['source_id'] for s in observation.get('sources', []))
        observed.extend(observation.get('opened_sources', {}))
        for fact in observation.get('facts', {}).values():
            observed.extend(c['source_id'] for c in fact['citations'])
    observed = list(dict.fromkeys(observed))
    return {'source_ids': list(dict.fromkeys(selected + observed))[:6],
            'selected_source_ids': selected, 'observed_source_ids': observed,
            'correct': trajectory['result']['correct'], 'valid': trajectory['result']['valid'],
            'origin': 'Selected evidence and observed alternatives from one frozen greedy Reader trajectory'}


def bounded_observation(graph, question, source_ids, stage, feedback):
    source_ids = list(source_ids)
    session = requests.Session()
    session.trust_env = False
    try:
        while True:
            observation = graph.observation(question, source_ids, stage, feedback)
            response = session.post('http://127.0.0.1:19109/tokenize', json={
                'model': 'macaq', 'messages': observation, 'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
            response.raise_for_status()
            if response.json()['count'] <= 12000:
                return source_ids, observation
            if not source_ids:
                raise ValueError('Writer question and existing facts exceed the policy input budget')
            source_ids.pop()
    finally:
        session.close()


def observed_feedback(row, current_reader=None):
    fields = {'reader_answer_correct': row['correct'], 'reader_valid': row['valid'],
              'reader_selected_source_ids': row['selected_source_ids'],
              'feedback_graph': 'The fixed initial graph before this sequence of writes'}
    if row.get('generated_answer') is not None:
        fields['reader_generated_answer'] = row['generated_answer']
    if current_reader is not None:
        fields['current_graph_reader'] = {
            'reader': 'fixed QFAD source retrieval plus graph facts',
            'answer_correct': current_reader['score']['correct'],
            'generated_answer': current_reader['answer'],
            'selected_source_ids': current_reader['retrieval']['source_ids'],
            'selected_fact_ids': current_reader['retrieval']['fact_ids']}
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-generated-answer', action='store_true')
    args = parser.parse_args()
    assert json.loads((args.reader_run / 'complete.json').read_text())['worker_exit'] == 0
    state = json.loads((args.reader_run / 'graph_snapshot.json').read_text())
    feedback = {}
    for path in sorted((args.reader_run / 'trajectories').glob('*.json')):
        trajectory = json.loads(path.read_text())
        persona = path.stem.split('_', 1)[0][1:]
        feedback[path.stem] = from_trajectory(trajectory, state[persona]['facts'])
        if args.include_generated_answer:
            answer_path = args.reader_run / 'answers' / path.name
            feedback[path.stem]['generated_answer'] = json.loads(answer_path.read_text())['answer'] \
                if answer_path.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(feedback, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
