"""Prepare frozen-policy transfer cohorts and score their saved RAW-RAG answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from graph_environment import base
from medmemory_scoring import JudgeClient
from semantic_query_writer_rl import RUBRIC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--patients', type=int, nargs='+', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patients = {p: base.Patient(p, store_names=('chunk',)) for p in args.patients}
    jobs, episodes, all_ids = [], [], []
    for persona, patient in patients.items():
        for _, questions in patient.groups:
            temporal = [q for q in questions if q['query_type'] == 'temporal_localization']
            entities = [q for q in questions if q['query_type'] == 'entity_exact_match']
            selected = sorted(temporal or entities or questions, key=lambda q: q['query_id'])[0]
            episodes.append(f'p{persona}_{selected["query_id"]}')
            all_ids.extend(f'p{persona}_{q["query_id"]}' for q in questions)
    # Fix the cohort and feedback triggers before inspecting any answer judgments.
    base.write_json(args.out / 'configuration.json', {
        'patients': args.patients, 'queries': len(all_ids), 'query_ids': all_ids,
        'episodes': episodes,
        'trigger_rule': 'First temporal query by ID at every checkpoint; otherwise first entity query, otherwise first query',
        'initial_graph': 'Empty generated facts and relations for each new patient',
        'correctness_feedback': 'Withheld from Writer; references used only for evaluation',
        'policy_update': False,
        'writer_before': '/home/administrator/du/memos/qwen27b_rsi/reader_factual/checkpoint.pt',
        'writer_after': '/home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint_step26.pt',
        'feedback_reader': '/home/administrator/du/memos/qwen27b_rsi/joint_writer_selection/checkpoint.pt',
        'scope': 'Frozen transfer on additional patients; no test-label-guided trigger selection, graph initialization, or parameter updates'})
    base.write_json(args.out / 'initial_graph.json', {
        str(p): {'facts': {}, 'relations': []} for p in args.patients})
    usages = {}
    for persona, patient in patients.items():
        questions = {q['query_id']: q for _, rows in patient.groups for q in rows}
        raw_answers = {row['query_id']: row
                       for path in (base.RAW / f'persona_{persona:02d}').glob('session_*_answers.json')
                       for row in base.read(path)}
        assert set(raw_answers) == set(questions)
        usage = {}
        with (base.RAW / f'persona_{persona:02d}/logs/rag_requests.jsonl').open() as stream:
            for line in stream:
                if '"call_type": "answer"' not in line[:600]:
                    continue
                record = json.loads(line)
                if not record.get('error') and record.get('response'):
                    usage[record['question_id']] = record['response']['usage']
        assert set(usage) == set(questions)
        usages[str(persona)] = usage
        jobs.extend((persona, question, raw_answers[qid], usage[qid]) for qid, question in questions.items())
    base.write_json(args.out / 'raw_usage.json', usages)
    judge = JudgeClient('gpt-6.1-sol')

    def score(job):
        persona, question, raw, usage = job
        path = args.out / 'raw_scores' / f'p{persona}' / f'{question["query_id"]}_raw.json'
        if path.exists():
            row = base.read(path)
            assert row['response'] == raw['answer'] and row['rubric'] == RUBRIC
            return row
        payload = {'question': question['question'], 'question_type': question['query_type'],
                   'reference_answers': [a['content'] for a in question['answers'] if a.get('is_correct')],
                   'response': raw['answer']}
        response = judge.chat(messages=[{'role': 'system', 'content': RUBRIC},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
        verdict = json.loads(response.content)
        assert type(verdict['correct']) is bool
        row = {'query_id': question['query_id'], 'persona': persona,
               'response': raw['answer'], 'rubric': RUBRIC, 'judge': 'gpt-6.1-sol',
               'reasoning_effort': 'high', 'usage': usage, 'judge_usage': response.usage, **verdict}
        base.write_json(path, row)
        print(f'p{persona}_{question["query_id"]}', verdict['correct'], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=15) as pool:
        rows = list(pool.map(score, jobs))
    base.write_json(args.out / 'raw_summary.json', {
        'queries': len(rows), 'correct': sum(r['correct'] for r in rows),
        'qwen_tokens': sum(r['usage']['total_tokens'] for r in rows),
        'patients': {str(p): {'queries': sum(r['persona'] == p for r in rows),
                            'correct': sum(r['correct'] for r in rows if r['persona'] == p)}
                     for p in args.patients},
        'scope': 'Previously generated RAW-RAG answers, scored under the fixed Sol semantic rubric'})


if __name__ == '__main__':
    main()
