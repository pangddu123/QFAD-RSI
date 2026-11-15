"""Run source-only Qwen27B extraction and save the resulting navigation graph."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shlex

from medical_navigation import NavigationGraph, base, extraction_messages, parse_extraction, verify_extraction, admit_extraction
from medmemory_scoring import JudgeClient

REMOTE = '/home/administrator/du/memos/qwen27b_rsi'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--worker-script', default=REMOTE + '/policy_worker.py')
    args = parser.parse_args()
    plan = base.read(args.source_plan)
    args.out.mkdir(parents=True, exist_ok=True)
    graphs = {pid: NavigationGraph(base.Patient(pid, store_names=('chunk',)))
              for pid in sorted({r['patient'] for r in plan})}
    base.write_json(args.out / 'configuration.json', {
        'process_id': os.getpid(),
        'sources': plan, 'model': 'Qwen3.8-27B', 'quantization': 'NF4',
        'resume': args.resume, 'parameter_updates': False, 'source_only_writer': True,
        'max_output_tokens': 3072, 'max_input_tokens': 12000,
        'acceptance': 'Individual source-supported facts and relations with admitted endpoints; no downstream answer selection',
    })
    command = ['ssh', '-T', 'win', f'ssh 2A100 env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 '
        f'OPENBLAS_NUM_THREADS=4 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u '
        f'{shlex.quote(args.worker_script)} --model /home/administrator/du/model/Qwen/Qwen3.8-27B '
        f'--out {shlex.quote(REMOTE + "/" + args.out.name)}']
    if args.resume:
        command[-1] += ' --resume ' + shlex.quote(args.resume)
    judge = JudgeClient('gpt-6.1-sol')
    results = []
    with (args.out / 'worker.log').open('w') as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=errors, text=True, bufsize=1)
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Extraction worker ended; see worker.log')
            return json.loads(line)
        def request(packet):
            worker.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
            worker.stdin.flush()
            return receive()
        try:
            ready = receive()
            assert ready['event'] == 'ready'
            base.write_json(args.out / 'worker_runtime.json', ready)
            for index, item in enumerate(plan):
                graph = graphs[item['patient']]
                source = graph.sources[item['source_id']]
                # Earlier facts are source-grounded observations, not query feedback.
                existing = dict([(fid, f) for fid, f in graph.visible_facts(source['stage']).items()
                                 if not any(c['source_id'] == source['source_id'] for c in f['citations'])][-8:])
                messages = extraction_messages([source], existing)
                base.write_json(args.out / f'block{index}.observation.json', messages)
                packet = request({'operation': 'infer', 'id': f'block{index}', 'messages': messages,
                                  'seed': 0, 'max_tokens': 3072, 'native_budget': 15072})
                base.write_json(args.out / f'block{index}.samples.json', packet)
                extraction = None
                admission = None
                if packet['event'] == 'budget_exhausted':
                    verdict = {'supported': False, 'reason': 'Input budget exhausted'}
                else:
                    try:
                        extraction = parse_extraction(packet['rows'][0], [source], existing)
                    except (ValueError, KeyError, TypeError) as error:
                        verdict = {'supported': False, 'reason': str(error)}
                    else:
                        verdict = verify_extraction(extraction, [source], existing, judge)
                        accepted, admission = admit_extraction(extraction, verdict, existing)
                        graphs[item['patient']] = graph.extract_into(accepted, [source['source_id']])
                result = {**item, 'extraction': extraction, 'support': verdict, 'admission': admission,
                          'writer_tokens': packet['prompt_tokens'] + sum(len(r['token_ids']) for r in packet.get('rows', []))}
                results.append(result)
                base.write_json(args.out / f'block{index}.result.json', result)
                base.write_json(args.out / 'graph_state.json', {str(pid): g.state for pid, g in graphs.items()})
                print(index, item['patient'], len(extraction['facts']) if extraction else 0,
                      len(extraction['relations']) if extraction else 0, admission, flush=True)
            request({'operation': 'close'})
            worker.wait()
            summary = {'blocks': len(results), 'fully_supported_blocks': sum(r['support']['supported'] for r in results),
                'blocks_with_admitted_content': sum(bool(r['admission']) and
                    r['admission']['accepted_facts'] + r['admission']['accepted_relations'] > 0 for r in results),
                'facts': sum(len(g.state['facts']) for g in graphs.values()),
                'relations': sum(len(g.state['relations']) for g in graphs.values()),
                'writer_tokens': sum(r['writer_tokens'] for r in results),
                'scope': 'Frozen source-only extraction; no optimizer updates or QA-performance claim'}
            base.write_json(args.out / 'summary.json', summary)
            base.write_json(args.out / 'complete.json', {'blocks': len(results), 'worker_exit': worker.returncode})
            print(summary, flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.close()
                worker.wait()


if __name__ == '__main__':
    main()
