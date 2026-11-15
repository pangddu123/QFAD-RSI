"""Compare source support for frozen writers without exposing evaluation answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from qfad_fact_graph import base
from fact_writer_protocol import parse_output
from medmemory_scoring import JudgeClient
from train_fact_writer import SUPPORT_PROMPT

ROOT = base.ROOT / 'results/qfad_fact_writer_rl_20261114'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='')
    parser.add_argument('--split', choices=['validation', 'evaluation'], required=True)
    parser.add_argument('--conditions', nargs='+', default=['initial', 'trained'])
    args = parser.parse_args()
    root = ROOT / args.run / args.split
    requests = base.read(root / 'writer_inputs.json')
    destination = root / 'source_support'
    judge = JudgeClient('gpt-6.1-sol')

    def audit(request):
        key = request['episode_id']
        path = destination / f'{key}.json'
        previous = base.read(path) if path.exists() else None
        facts = list(previous['payload']['facts']) if previous else []
        judged_count = len(facts)
        indices = {}
        invalid = {}
        for condition in args.conditions:
            prediction = base.read(root / f'writer_outputs/{condition}/{key}.json')
            try:
                if not prediction['finished']:
                    raise ValueError('Writer output truncated')
                parsed = parse_output(prediction['text'], request['sources'])
                indices[condition] = []
                for source in parsed:
                    for fact in source['facts']:
                        item = {'source_id': source['source_id'], **fact}
                        if item not in facts:
                            facts.append(item)
                        indices[condition].append(facts.index(item))
            except (ValueError, KeyError, TypeError) as error:
                invalid[condition] = str(error)
        payload = {'sources': [{'source_id': s['source_id'], 'text': s['text']}
                               for s in request['sources']], 'facts': facts}
        if previous:
            assert previous['payload']['sources'] == payload['sources']
        verdict = {'supported': list(previous['verdict']['supported']) if previous else []}
        assessments = list(previous.get('assessments', [])) if previous else []
        if len(facts) > judged_count:
            pending = {'sources': payload['sources'], 'facts': facts[judged_count:]}
            response = judge.chat(messages=[{'role': 'system', 'content': SUPPORT_PROMPT},
                {'role': 'user', 'content': json.dumps(pending, ensure_ascii=False)}],
                max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
            new_verdict = json.loads(response.content)
            assert len(new_verdict['supported']) == len(pending['facts'])
            assert all(type(v) is bool for v in new_verdict['supported'])
            verdict['supported'].extend(new_verdict['supported'])
            assessments.append({'fact_indices': list(range(judged_count, len(facts))),
                                'reason': new_verdict.get('reason', '')})
        saved = {'request_id': key, 'payload': payload, 'indices': indices,
                 'invalid': invalid, 'verdict': verdict,
                 'assessments': assessments,
                 'judge': 'gpt-6.1-sol', 'reasoning_effort': 'high'}
        base.write_json(path, saved)
        print(key, {c: sum(verdict['supported'][i] for i in ids)
                    for c, ids in indices.items()}, flush=True)
        return saved

    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(audit, requests))
    summary = {}
    for condition in args.conditions:
        valid = [r for r in rows if condition in r['indices']]
        flags = [r['verdict']['supported'][i] for r in valid for i in r['indices'][condition]]
        summary[condition] = {'accepted_sources': len(valid), 'invalid_sources': len(rows)-len(valid),
                              'facts': len(flags), 'supported_facts': sum(flags),
                              'supported_fraction': sum(flags)/len(flags) if flags else None}
    base.write_json(destination / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
