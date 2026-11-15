"""Exercise QFAD-to-original-evidence answering on an extracted graph."""
import argparse
import os
from pathlib import Path

from medical_navigation import NavigationGraph, base
from train_writer import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extraction-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    complete = base.read(args.extraction_run / 'complete.json')
    assert complete['worker_exit'] == 0
    state = base.read(args.extraction_run / 'graph_state.json')
    queries = [(1, 'session_10_sua_1'), (1, 'session_30_tla_2'), (2, 'session_10_tla_2')]
    graphs = {p: NavigationGraph(base.Patient(p, store_names=('chunk',)), state[str(p)]) for p in (1, 2)}
    args.out.mkdir(parents=True, exist_ok=True)
    base.write_json(args.out / 'configuration.json', {
        'process_id': os.getpid(),
        'extraction_run': str(args.extraction_run), 'queries': queries,
        'reader': 'fixed QFAD over entity, fact and source nodes',
        'answerer': 'fixed Qwen3.8-27B-FP8', 'answer_evidence': 'verbatim original source windows',
        'scope': 'Three-query execution probe; not a patient-level performance estimate',
    })
    evaluator = Evaluator(args.out / 'answers')
    results = []
    for patient, qid in queries:
        graph = graphs[patient]
        stage = int(qid.split('_')[1]) // 10 - 1
        question = next(q for q in graph.patient.groups[stage][1] if q['query_id'] == qid)
        answer = evaluator.answer(graph, question, stage, f'p{patient}_{qid}')
        evidence = answer['retrieval']
        assert len(evidence['documents']) == len(evidence['source_windows'])
        for document, window in zip(evidence['documents'], evidence['source_windows']):
            source = graph.sources[window['source_id']]
            assert source['stage'] <= stage
            lines = source['text'].splitlines()
            for segment in window['segments']:
                assert segment['text'] == '\n'.join(lines[i] for i in segment['line_ids'])
                assert segment['text'] in document
        result = {'patient': patient, 'query_id': qid, 'correct': answer['score']['correct'],
                  'source_ids': evidence['source_ids'], 'fact_ids': evidence['fact_ids'],
                  'context_tokens': evidence['context_tokens'], 'verbatim_source_windows_verified': True}
        results.append(result)
        print(result, flush=True)
    base.write_json(args.out / 'summary.json', {
        'queries': len(results), 'correct': sum(r['correct'] for r in results),
        'results': results, 'scope': 'Execution probe; no training improvement claim',
    })


if __name__ == '__main__':
    main()
