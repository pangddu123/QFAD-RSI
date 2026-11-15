"""Prepare source-only writer episodes from Reader-solvable training queries."""
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import re
from qfad_fact_graph import base
from localize_feedback_sources import PROMPT
from medmemory_scoring import JudgeClient

ROOT = base.ROOT/'results/qfad_fact_writer_rl_20261114'
OUT = ROOT/'curriculum'


def main():
    used = {row['episode_id'] for row in base.read(ROOT/'stable_feedback/training_plan.json')}
    selected = []
    for persona in (3, 4, 5):
        directory = base.ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{persona}/eval'
        queries = {q['query_id']: q for q in base.read(directory/'generated_queries.json')['queries']}
        groups = defaultdict(deque)
        for checkpoint in (20, 40, 60, 80):
            for score in base.read(base.RAW/f'persona_{persona:02d}/session_{checkpoint:03d}_scores.json'):
                key = f'p{persona}_{score["query_id"]}'
                if score['is_correct'] and key not in used:
                    groups[score['query_type']].append({'persona': persona, 'checkpoint': checkpoint,
                        'episode_id': key, 'query': queries[score['query_id']], 'raw_correct': True})
        chosen = []
        while len(chosen) < 8:
            available = [kind for kind in sorted(groups) if groups[kind]]
            assert available, f'Insufficient unused Reader-solvable queries for patient {persona}'
            for kind in available:
                chosen.append(groups[kind].popleft())
                if len(chosen) == 8:
                    break
        selected.extend(chosen)
    base.write_json(OUT/'selected_queries.json', selected)
    judge = JudgeClient('gpt-6.1-sol')

    def localize(item):
        q = item['query'];checkpoint = item['checkpoint']
        path = OUT/f'localization/{item["episode_id"]}.json'
        if path.exists():
            return base.read(path)
        points = [p for p in q['source_key_points'] if 0 < int(p.get('session_id', 0)) <= checkpoint]
        assert points
        if len(points) == 1:
            result = {'point_index': 0, 'reason': 'The question has one annotated source point.'}
        else:
            payload = {'question': q['question'],
                'references': [a['content'] for a in q['answers'] if a.get('is_correct')],
                'points': [{'index': i, **p} for i, p in enumerate(points)]}
            result = json.loads(judge.chat(messages=[{'role': 'system', 'content': PROMPT},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'}).content)
        index = result['point_index']
        assert type(index) is int and 0 <= index < len(points)
        row = {'persona': item['persona'], 'query_id': q['query_id'],
               'session_id': int(points[index]['session_id']), 'selected_annotation': points[index], **result}
        base.write_json(path, row)
        return row

    with ThreadPoolExecutor(max_workers=10) as pool:
        localized = list(pool.map(localize, selected))
    inputs = [];plans = [];excluded = []
    for persona in (3, 4, 5):
        patient = base.Patient(persona);chunks = patient.stores['chunk']
        directory = base.ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{persona}/eval'
        sessions = {int(s['session_id']): s for s in base.read(directory/'generated_dialogues.json')['sessions']}
        stages = {checkpoint: (stage, {q['query_id']: q for q in queries})
                  for stage, (checkpoint, queries) in enumerate(patient.groups)}
        for item, location in zip(selected, localized):
            if item['persona'] != persona:
                continue
            checkpoint = item['checkpoint'];stage, queries = stages[checkpoint]
            q = queries[item['query']['query_id']]
            date = sessions[location['session_id']]['event_info']['date']
            dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', q['question'])
            if q['query_type'] == 'temporal_localization' and dates and date not in dates:
                excluded.append({'episode_id': item['episode_id'], 'reason': 'Question date differs from original source session date',
                                 'question_dates': dates, 'source_date': date})
                continue
            visible = {text for s, text in patient.history if s <= stage}
            candidates = [i for i, text in enumerate(chunks['texts']) if text in visible and
                          text.partition('\n')[0] == f'[Session time: {date}]']
            assert candidates
            feedback = q['question']+'\nTraining reference target: '+'; '.join(q['correct_answers'])
            ranked = base.rerank(feedback, [chunks['texts'][i] for i in candidates])
            index = candidates[ranked[0]['index']];text = chunks['texts'][index];source_id = chunks['ids'][index]
            inputs.append({'episode_id': item['episode_id'], 'sources': [{'source_id': source_id, 'text': text,
                'old_facts': patient.openie[text].get('extracted_triples', [])}]})
            plans.append({'episode_id': item['episode_id'], 'persona': persona, 'stage': stage,
                'checkpoint': checkpoint, 'query_ids': [q['query_id']], 'curriculum_raw_correct': True,
                'source_selection': {'training_annotation_used': True, 'session_id': location['session_id'],
                    'record_date': date, 'source_id': source_id, 'writer_receives_query_or_reference': False}})
            base.write_json(OUT/'writer_inputs.json', inputs)
            base.write_json(OUT/'training_plan.json', plans)
            print(item['episode_id'], q['query_type'], date, flush=True)
    base.write_json(OUT/'excluded_queries.json', excluded)
    base.write_json(OUT/'prepared.json', {'selected': len(selected), 'prepared': len(inputs), 'excluded': len(excluded),
                                       'patients': [3, 4, 5], 'uses_target_patients': False})
    print('Prepared', len(inputs), 'Reader-solvable source-only writer episodes', flush=True)


if __name__ == '__main__':
    main()
