"""Build complete patient navigation memory using a served Qwen27B policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from openai import OpenAI
import requests

from medical_navigation import NavigationGraph, admit_extraction, base, extraction_messages, parse_extraction, verify_extraction
from medmemory_scoring import JudgeClient


def build_patient(pid, items, args):
    graph = NavigationGraph(base.Patient(pid, store_names=('chunk',)))
    positions = {}
    for i, (_, text) in enumerate(graph.patient.history):
        positions.setdefault(text, i)
    order = [positions[graph.sources[item['source_id']]['text']] for item in items]
    assert order == sorted(set(order)), 'Sources must be unique and follow arrival order'
    directory = args.out / f'p{pid}'
    client = OpenAI(api_key='local', base_url=args.api_base, timeout=900)
    session = requests.Session()
    session.trust_env = False
    judge = JudgeClient('gpt-6.1-sol')
    counts = {'blocks': 0, 'facts': 0, 'relations': 0, 'writer_tokens': 0,
              'judge_tokens': 0, 'parse_failures': 0}
    for index, item in enumerate(items):
        source = graph.sources[item['source_id']]
        existing = dict(list(graph.state['facts'].items())[-8:])
        messages = extraction_messages([source], existing)
        while True:
            tokenized = session.post(args.api_base.removesuffix('/v1') + '/tokenize', json={
                'model': args.model, 'messages': messages, 'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)
            tokenized.raise_for_status()
            if tokenized.json()['count'] <= 12000:
                break
            if not existing:
                raise ValueError('Original source alone exceeds the Writer input budget')
            existing.pop(next(iter(existing)))
            messages = extraction_messages([source], existing)
        prefix = directory / f'block{index:05d}'
        observation_path = prefix.with_suffix('.observation.json')
        if observation_path.exists():
            assert base.read(observation_path) == messages, 'Saved extraction has a different observation'
        else:
            base.write_json(observation_path, messages)
        sample_path = prefix.with_suffix('.sample.json')
        if sample_path.exists():
            sample = base.read(sample_path)
        else:
            response = client.chat.completions.create(model=args.model, messages=messages,
                temperature=0, seed=0, max_tokens=3072,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            sample = {'text': response.choices[0].message.content or '',
                      'finished': response.choices[0].finish_reason == 'stop',
                      'usage': response.usage.model_dump()}
            base.write_json(sample_path, sample)
        result_path = prefix.with_suffix('.result.json')
        if result_path.exists():
            result = base.read(result_path)
        else:
            extraction, support, admission, accepted = None, None, None, None
            try:
                extraction = parse_extraction(sample, [source], existing)
            except (ValueError, KeyError, TypeError) as error:
                support = {'parse_error': str(error)}
            else:
                support = verify_extraction(extraction, [source], existing, judge)
                accepted, admission = admit_extraction(extraction, support, existing)
            result = {**item, 'extraction': extraction, 'support': support,
                      'admission': admission, 'accepted': accepted}
            base.write_json(result_path, result)
        if result['accepted'] is not None:
            graph = graph.extract_into(result['accepted'], [item['source_id']])
        else:
            counts['parse_failures'] += 1
        counts['blocks'] += 1
        counts['writer_tokens'] += sample['usage']['total_tokens']
        counts['judge_tokens'] += result['support'].get('usage', {}).get('total_tokens', 0)
        counts['facts'] = len(graph.state['facts'])
        counts['relations'] = len(graph.state['relations'])
        base.write_json(directory / 'progress.json', counts)
        print(json.dumps({'patient': pid, **counts}), flush=True)
    base.write_json(directory / 'graph_state.json', graph.state)
    base.write_json(directory / 'complete.json', counts)
    return pid, graph.state, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--api-base', default='http://127.0.0.1:19109/v1')
    parser.add_argument('--model', default='macaq')
    args = parser.parse_args()
    plan = base.read(args.plan)
    assert plan and {item['patient'] for item in plan} <= {1, 2}
    config = {'plan': plan, 'model': args.model, 'api_base': args.api_base,
              'weights_family': 'Qwen3.8-27B', 'inference_quantization': 'FP8',
              'max_input_tokens': 12000, 'max_output_tokens': 3072,
              'source_only_writer': True, 'temperature': 0, 'seed': 0,
              'patient_concurrency': 2, 'source_order': 'Sequential within patient',
              'scope': 'Source-only memory construction; no query or reference-answer input'}
    if (args.out / 'configuration.json').exists():
        assert base.read(args.out / 'configuration.json') == config
    base.write_json(args.out / 'configuration.json', config)
    base.write_json(args.out / 'runtime.json', {'process_id': os.getpid()})
    patients = sorted({item['patient'] for item in plan})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda p: build_patient(p, [r for r in plan if r['patient'] == p], args), patients))
    base.write_json(args.out / 'graph_state.json', {str(pid): state for pid, state, _ in results})
    base.write_json(args.out / 'summary.json', {str(pid): counts for pid, _, counts in results})
    base.write_json(args.out / 'complete.json', {'patients': patients, 'blocks': len(plan)})


if __name__ == '__main__':
    main()
