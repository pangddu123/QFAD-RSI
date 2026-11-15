"""Assess whether selected validation sources contain the requested factual evidence."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

from semantic_query_writer_rl import ROOT, read
from medmemory_scoring import JudgeClient, write_json

PROMPT = '''Assess evidence coverage for a factual medical-memory benchmark question. This is an offline retrieval diagnosis. Use only the supplied source text; the reference describes what the question asks for, not evidence that the source contains it. Label sufficient only if this source contains all decisive requested facts, including the required time or change when applicable. Label partial when it contains some but not all decisive facts, and absent when it contains none. A related topic is not sufficient evidence. Do not supply clinical knowledge or infer an unsupported diagnosis. Return JSON with coverage (sufficient, partial, or absent), missing_evidence (a brief string), and supporting_quotes (a list of exact excerpts from the source).'''


def main():
    validation = ROOT / 'results/qfad_fact_writer_rl_20261114/validation'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=validation)
    args = parser.parse_args()
    root = args.source_root
    requests = {r['episode_id']: r for r in read(root / 'writer_inputs.json')}
    queries = {q['query_id']: q for q in read(ROOT / 'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_6/eval/generated_queries.json')['queries']}
    plan = [r for r in read(root / 'query_plan.json')
            if queries[r['query_id']]['query_type'] in ('entity_exact_match', 'temporal_localization', 'state_update')]
    destination = root / 'source_coverage'
    judge = JudgeClient('gpt-6.1-sol')

    def assess(item):
        q = queries[item['query_id']]
        keys = item['writer_request_ids'] if 'writer_request_ids' in item else [item['writer_request_id']]
        sources = [requests[key]['sources'][0] for key in keys]
        source = {'source_id': ','.join(s['source_id'] for s in sources),
                  'text': '\n\n'.join(s['text'] for s in sources)}
        payload = {'question': q['question'], 'question_type': q['query_type'],
                   'reference_answers': [a['content'] for a in q['answers'] if a.get('is_correct')],
                   'source_id': source['source_id'], 'source_text': source['text']}
        path = destination / f'{q["query_id"]}.json'
        if path.exists():
            saved = read(path)
            assert saved['payload'] == payload and saved['rubric'] == PROMPT
            return saved
        response = judge.chat(messages=[{'role': 'system', 'content': PROMPT},
                              {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                              max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
        verdict = json.loads(response.content)
        assert verdict['coverage'] in ('sufficient', 'partial', 'absent')
        assert isinstance(verdict['supporting_quotes'], list)
        saved = {'query_id': q['query_id'], 'payload': payload, 'rubric': PROMPT,
                 'verdict': verdict,
                 'quotes_found_in_source': all(isinstance(t, str) and t in source['text'] for t in verdict['supporting_quotes']),
                 'judge': 'gpt-6.1-sol', 'reasoning_effort': 'high',
                 'semantic_correct': {c: read(validation / f'semantic/{c}_focused_k15/p6/{q["query_id"]}.json')['correct']
                                      for c in ('raw', 'initial', 'trained')}}
        write_json(path, saved)
        print(q['query_id'], verdict['coverage'], flush=True)
        return saved

    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(assess, plan))
    summary = {'n': len(rows), 'coverage': dict(Counter(r['verdict']['coverage'] for r in rows)),
               'unmatched_quote_queries': [r['query_id'] for r in rows if not r['quotes_found_in_source']],
               'queries': [{'query_id': r['query_id'], 'coverage': r['verdict']['coverage'],
                            'semantic_correct': r['semantic_correct']} for r in rows]}
    write_json(destination / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
