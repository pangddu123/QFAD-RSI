"""Replay saved extraction atoms to check admission and original-source navigation."""
import argparse
from pathlib import Path

from medical_navigation import NavigationGraph, admit_extraction, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extraction-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = base.read(args.extraction_run / 'configuration.json')
    assert base.read(args.extraction_run / 'complete.json')['worker_exit'] == 0
    graphs, admissions = {}, []
    for index, item in enumerate(config['sources']):
        pid = item['patient']
        if pid not in graphs:
            graphs[pid] = NavigationGraph(base.Patient(pid, store_names=('chunk',)))
        saved = base.read(args.extraction_run / f'block{index}.result.json')
        if saved['extraction'] is None:
            raise ValueError('Replay requires a parsed extraction and per-atom source verdicts')
        observation = base.read(args.extraction_run / f'block{index}.observation.json')
        import json
        existing = json.loads(observation[1]['content'])['existing_facts']
        for fid, fact in existing.items():
            assert graphs[pid].state['facts'].get(fid) == fact
        accepted, admission = admit_extraction(saved['extraction'], saved['support'], existing)
        graphs[pid] = graphs[pid].extract_into(accepted, [item['source_id']])
        admissions.append({**item, **admission})
    base.write_json(args.out / 'graph_state.json', {str(p): g.state for p, g in graphs.items()})
    base.write_json(args.out / 'admissions.json', admissions)
    results = []
    for pid, qid in [(1, 'session_10_sua_1'), (1, 'session_30_tla_2'), (2, 'session_10_tla_2')]:
        graph = graphs[pid]
        stage = int(qid.split('_')[1]) // 10 - 1
        question = next(q for q in graph.patient.groups[stage][1] if q['query_id'] == qid)
        evidence = graph.retrieve(question['question'], stage)
        assert evidence['source_windows'], 'Supported source entries must produce nonempty navigation'
        for window in evidence['source_windows']:
            source = graph.sources[window['source_id']]
            assert source['stage'] <= stage
            for segment in window['segments']:
                assert segment['text'] == '\n'.join(source['text'].splitlines()[i] for i in segment['line_ids'])
        base.write_json(args.out / f'p{pid}_{qid}.retrieval.json', evidence)
        results.append({'patient': pid, 'query_id': qid, 'source_ids': evidence['source_ids'],
                        'context_tokens': evidence['context_tokens'], 'verbatim_windows': True})
    summary = {'facts': sum(len(g.state['facts']) for g in graphs.values()),
               'relations': sum(len(g.state['relations']) for g in graphs.values()),
               'retrieval_checks': results, 'extraction_run': str(args.extraction_run),
               'scope': 'Saved-generation admission replay and real QFAD retrieval; no new generation, QA score or parameter update'}
    base.write_json(args.out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
