"""Evaluate the complete two-patient query set under a fixed graph and Reader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

from graph_environment import GraphEnvironment, base
from train_writer import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--paired-with', type=Path, help='Reuse the identical answer realization when prompts match')
    parser.add_argument('--patients', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--navigation', action='store_true', help='Read original windows via the extraction graph')
    parser.add_argument('--queries', type=Path, help='JSON list of patient-qualified query IDs')
    parser.add_argument('--raw-scores', type=Path,
                        default=base.OUT / 'semantic/evaluation')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = base.read(args.state) if args.state else {}
    if args.paired_with:
        prior = base.read(args.paired_with / 'configuration.json')
        assert prior['answer_model'] == 'Qwen3.8-27B-FP8' and prior['answer_max_tokens'] == 16384
    base.write_json(args.out / 'graph_snapshot.json', state)
    graph_type = GraphEnvironment
    reader = 'fixed QFAD source retrieval plus graph facts'
    if args.navigation:
        from medical_navigation import NavigationGraph
        graph_type = NavigationGraph
        reader = 'QFAD extraction graph with verbatim original windows'
        if set(map(str, args.patients)) != set(state):
            raise ValueError('Missing or unexpected patient graph')
    graphs = {p: graph_type(base.Patient(p, store_names=('chunk',)), state.get(str(p))) for p in args.patients}
    evaluator = Evaluator(args.out / 'answers')
    jobs = []
    for persona, graph in graphs.items():
        for stage, (_, queries) in enumerate(graph.patient.groups):
            for q in queries:
                jobs.append((persona, stage, q))
    if args.patients == [1, 2]:
        assert len(jobs) == 197, len(jobs)
    if args.queries:
        ids = base.read(args.queries)
        allowed = set(ids)
        jobs = [(p, s, q) for p, s, q in jobs if f'p{p}_{q["query_id"]}' in allowed]
        if not ids or len(jobs) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError('Missing or duplicate evaluation query IDs')
    if args.navigation and args.paired_with and prior['reader'] != reader:
        raise ValueError('Paired evaluation requires the same navigation Reader')
    base.write_json(args.out / 'configuration.json', {
        'patients': args.patients, 'queries': len(jobs), 'process_id': os.getpid(), 'reader': reader,
        'raw_scores': str(args.raw_scores),
        'answer_model': 'Qwen3.8-27B-FP8', 'answer_served_name': 'macaq',
        'graph_state': str(args.state) if args.state else None, 'context_cap': 8000,
        'answer_max_tokens': 16384, 'concurrency': args.concurrency,
        'paired_with': str(args.paired_with) if args.paired_with else None})
    def evaluate(job):
        persona, stage, question = job
        name = f'p{persona}_{question["query_id"]}'
        path = args.out / 'answers' / f'{name}.json'
        prior_path = args.paired_with / 'answers' / f'{name}.json' if args.paired_with else None
        if prior_path:
            while not prior_path.exists():
                try:
                    os.kill(prior['process_id'], 0)
                except ProcessLookupError:
                    raise RuntimeError(f'Paired evaluation ended without {prior_path}') from None
                time.sleep(1)
        paired = base.read(prior_path) if prior_path else None
        row = base.read(path) if path.exists() else evaluator.answer(graphs[persona], question, stage, name,
                                                                    paired_answer=paired)
        raw = base.read(args.raw_scores / f'p{persona}' / f'{question["query_id"]}_raw.json')
        print(name, row['score']['correct'], 'raw', raw['correct'], flush=True)
        return {'episode_id': name, 'correct': row['score']['correct'], 'raw_correct': raw['correct'],
                'usage': row['usage'], 'retrieval': row['retrieval'],
                'prior_correct': paired['score']['correct'] if paired else None,
                'paired_reuse': row.get('paired_reuse', False)}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(evaluate, jobs))
    summary = {'queries': len(rows), 'correct': sum(r['correct'] for r in rows),
        'raw_correct': sum(r['raw_correct'] for r in rows),
        'wins': [r['episode_id'] for r in rows if r['correct'] and not r['raw_correct']],
        'losses': [r['episode_id'] for r in rows if not r['correct'] and r['raw_correct']],
        'answer_usage': {k: sum(r['usage'][k] for r in rows) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'rerank_document_tokens': sum(r['retrieval']['rerank_document_tokens'] for r in rows),
        'rerank_native_input_tokens': sum(r['retrieval']['rerank_native_input_tokens'] for r in rows)
                                      if all('rerank_native_input_tokens' in r['retrieval'] for r in rows) else None,
        'paired_reused_answers': sum(r['paired_reuse'] for r in rows),
        'net_correct_vs_prior': sum(int(r['correct']) - int(r['prior_correct']) for r in rows)
                                if all(r['prior_correct'] is not None for r in rows) else None,
        'scope': 'Fixed Reader and graph; this evaluation alone does not establish an RL gain'}
    base.write_json(args.out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
