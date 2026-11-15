"""Score all frozen-writer answers with the established semantic rubric."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from semantic_query_writer_rl import ROOT, RAW, RUBRIC, read
from medmemory_scoring import JudgeClient, write_json

OUT = ROOT/'results/qfad_fact_writer_rl_20261114'
RAW_SCORES = ROOT/'results/qfad_query_writer_rl_20261113/semantic'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='')
    parser.add_argument('--split', choices=['validation', 'evaluation'], required=True)
    parser.add_argument('--conditions', nargs='+', default=['original', 'initial', 'trained'])
    parser.add_argument('--query-method', choices=['global', 'focused'], default='focused')
    parser.add_argument('--facts', type=int, default=15)
    args = parser.parse_args()
    root = OUT/args.run/args.split
    suffix = '_focused' if args.query_method == 'focused' else ''
    plan = read(root/'query_plan.json')
    assert len(plan) == len({(r['persona'], r['query_id']) for r in plan}) == (41 if args.split == 'validation' else 197)
    qmaps = {p: {q['query_id']: q for q in read(ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{p}/eval/generated_queries.json')['queries']}
             for p in {r['persona'] for r in plan}}
    raw_answers = {}
    for p in qmaps:
        raw_answers[p] = {r['query_id']: r for path in (RAW/f'persona_{p:02d}').glob('session_*_answers.json') for r in read(path)}
    known = {};groups = {};targets = {}
    for item in plan:
        p = item['persona'];qid = item['query_id'];q = qmaps[p][qid]
        rows = {'raw': raw_answers[p][qid]}
        rows.update({condition: read(root/f'{condition}{suffix}_k{args.facts}/p{p}/{qid}.json') for condition in args.conditions})
        baseline = read(RAW_SCORES/args.split/f'p{p}/{qid}_raw.json')
        assert baseline['rubric'] == RUBRIC and baseline['response'] == rows['raw']['answer']
        assert baseline['judge'] == 'gpt-6.1-sol' and baseline['reasoning_effort'] == 'high'
        known[(p, qid, rows['raw']['answer'])] = baseline
        for condition, row in rows.items():
            payload = {'question': q['question'], 'question_type': q['query_type'],
                       'reference_answers': [a['content'] for a in q['answers'] if a.get('is_correct')],
                       'response': row['answer']}
            key = (p, qid, row['answer'])
            target = root/f'semantic/{condition}{suffix}_k{args.facts}/p{p}/{qid}.json'
            targets[condition, p, qid] = target
            if target.exists():
                prior = read(target)
                assert prior['response'] == row['answer'] and prior['rubric'] == RUBRIC
                if key in known:
                    assert known[key]['correct'] == prior['correct']
                known[key] = prior
            groups.setdefault(key, []).append((target, payload))
    judge = JudgeClient('gpt-6.1-sol')

    def score(job):
        key, entries = job
        payload = entries[0][1]
        if key in known:
            verdict = known[key]
        else:
            response = judge.chat(messages=[{'role': 'system', 'content': RUBRIC},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
            verdict = json.loads(response.content)
            assert type(verdict.get('correct')) is bool
        result = {'query_id': key[1], **payload, 'judge': 'gpt-6.1-sol',
                  'reasoning_effort': 'high', 'rubric': RUBRIC,
                  'correct': verdict['correct'], 'reason': verdict['reason']}
        for path, _ in entries:
            if not path.exists():
                write_json(path, result)
        print(key[0], key[1], [p.parent.parent.name for p, _ in entries], result['correct'], flush=True)

    with ThreadPoolExecutor(max_workers=15) as pool:
        list(pool.map(score, groups.items()))
    conditions = ['raw', *args.conditions]
    scores = {(c, p, qid): read(path) for (c, p, qid), path in targets.items()}
    summary = {'split': args.split, 'query_method': args.query_method, 'n': len(plan),
               'judge': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'rubric': RUBRIC,
               'conditions': {}, 'comparisons': {}}
    for condition in conditions:
        summary['conditions'][condition] = {'correct': sum(scores[condition, r['persona'], r['query_id']]['correct'] for r in plan)}
    for condition in args.conditions:
        for control in ['raw'] + (['initial'] if 'initial' in args.conditions and condition not in ('initial', 'original') else []):
            wins = [];losses = []
            for item in plan:
                p = item['persona'];qid = item['query_id']
                a = scores[condition, p, qid]['correct'];b = scores[control, p, qid]['correct']
                if a > b:
                    wins.append({'persona': p, 'query_id': qid})
                elif a < b:
                    losses.append({'persona': p, 'query_id': qid})
            summary['comparisons'][f'{condition}_vs_{control}'] = {'wins': wins, 'losses': losses,
                'correct_delta': len(wins)-len(losses), 'delta_pp': 100*(len(wins)-len(losses))/len(plan)}
    write_json(root/f'semantic_summary{suffix}_k{args.facts}.json', summary)
    print(summary['conditions'], flush=True)


if __name__ == '__main__':
    main()
