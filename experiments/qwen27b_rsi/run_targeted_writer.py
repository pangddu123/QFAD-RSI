"""Prepare, pilot and evaluate the source-only QFAD writer with targeted guidance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

from medical_navigation import NavigationGraph, base
from prepare_extraction_training import training_plan
from train_extractor import REMOTE


HERE = Path(__file__).resolve().parent
ARMS = ('frozen', 'scalar', 'guided')


def make_cohort(patients, last_stage, excluded):
    sources, queries = [], []
    for pid in patients:
        patient = base.Patient(pid, store_names=('chunk',))
        graph = NavigationGraph(patient)
        seen = set()
        lookup = patient.stores['chunk']['lookup']
        for stage, original in patient.history:
            if (last_stage is not None and stage > last_stage) or original in seen:
                continue
            seen.add(original)
            sid = patient.stores['chunk']['ids'][lookup[original]]
            if graph.sources[sid]['text'] != original:
                raise ValueError('Source ID disagrees with patient history')
            sources.append({'patient': pid, 'source_id': sid})
        for stage, (_, questions) in enumerate(patient.groups):
            if last_stage is None or stage <= last_stage:
                queries.extend(f'p{pid}_{q["query_id"]}' for q in questions
                               if f'p{pid}_{q["query_id"]}' not in excluded)
    if not sources or not queries or len(queries) != len(set(queries)):
        raise ValueError('Empty cohort or duplicate evaluation queries')
    return sources, queries


def prepare(out, weight, plan_path):
    if not 0 < weight < float('inf'):
        raise ValueError('Expected a positive finite distillation weight')
    plan = base.read(plan_path) if plan_path else training_plan()
    training = {f'p{e["patient"]}_{q["query_id"]}' for e in plan['episodes'] for q in e['queries']}
    if not training or any(e['patient'] not in (1, 2) for e in plan['episodes']):
        raise ValueError('Training requires queries from patients 1 or 2')
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / 'training.json', plan)
    counts = {}
    for phase, patients, stage in [('pilot', [1, 2], 0), ('full', [3, 4], None)]:
        sources, queries = make_cohort(patients, stage, training)
        if phase == 'pilot':
            # Same compact source pool in all arms. Read only the original
            # retriever's delivered IDs, without reading its answer or grade as a feature.
            traces = base.ROOT / 'results/qwen27b_rsi/initial_full/answers'
            pool = {(int(q.split('_', 1)[0][1:]), sid) for q in queries
                    for sid in base.read(traces / f'{q}.json')['retrieval']['source_ids'][:6]}
            sources = [s for s in sources if (s['patient'], s['source_id']) in pool]
            if not pool or len(sources) != len(pool):
                raise ValueError('Pilot retrieval refers to sources outside the visible history')
        base.write_json(out / f'{phase}_sources.json', sources)
        base.write_json(out / f'{phase}_queries.json', queries)
        counts[phase] = {'patients': patients, 'source_blocks': len(sources), 'queries': len(queries)}
    base.write_json(out / 'protocol.json', {
        'id': uuid.uuid4().hex, 'guidance_weight': weight, 'cohorts': counts,
        'arms': ARMS, 'training_queries': sorted(training),
        'pilot_memory': 'Union of first six delivered sources per pilot query from the saved initial QFAD Reader',
        'pilot_pass': 'guided correct count strictly exceeds frozen on paired nontraining queries',
        'full_pass': 'guided correct count strictly exceeds both frozen and RAW-RAG on patients 3 and 4'})
    print(json.dumps(counts))


def deploy_worker():
    source = (HERE / 'policy_worker.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    path = f'{REMOTE}/policy_worker_{digest}.py'
    # Transfer only this module. Concurrent deployments publish identical bytes atomically.
    # The jump host runs Windows cmd.exe. Send shell syntax to Linux on stdin.
    delimiter = f'QFAD_WORKER_{digest}'
    if not source.endswith(b'\n') or delimiter.encode() in source.splitlines():
        raise ValueError('Worker source cannot be embedded verbatim in the deployment script')
    header = (f'set -eu\nt=$(mktemp {REMOTE}/worker.XXXXXX)\n'
              f'trap \'rm -f "$t"\' EXIT\ncat > "$t" <<\'{delimiter}\'\n')
    footer = (f'{delimiter}\necho "{digest}  $t" | sha256sum -c - >&2\n'
              f'mv "$t" {path}\n')
    subprocess.run(['ssh', '-T', 'win', 'ssh -T 2A100 bash -s'],
                   input=header.encode() + source + footer.encode(), check=True)
    return path


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True)


def paired_result(rows, queries):
    if any(set(by_query) != set(queries) for by_query in rows.values()) or set(rows) != {*ARMS, 'raw'}:
        raise ValueError('Evaluation arms did not cover identical queries')
    if any(type(v) is not bool for by_query in rows.values() for v in by_query.values()):
        raise ValueError('Expected binary semantic verdicts')
    counts = {arm: sum(scores.values()) for arm, scores in rows.items()}
    pairs = {arm: {
        'wins': [q for q in queries if rows['guided'][q] and not rows[arm][q]],
        'losses': [q for q in queries if rows[arm][q] and not rows['guided'][q]]}
        for arm in ('frozen', 'scalar', 'raw')}
    return {'queries': len(queries), 'correct': counts, 'guided_vs': pairs,
            'rl_gain': counts['guided'] > counts['frozen'],
            'guidance_gain_over_scalar': counts['guided'] > counts['scalar'],
            'beats_raw': counts['guided'] > counts['raw']}


def execute(out, phase):
    config = base.read(out / 'protocol.json')
    checkpoints = out / 'checkpoints.json'
    if phase == 'full' and not base.read(out / 'pilot_result.json')['rl_gain']:
        raise ValueError('Pilot did not show paired QA improvement')
    worker = deploy_worker()
    # Keep every attempt independent on both hosts. Reuse only completed stages.
    prefix = 'tg_' + config['id'] + '_'
    policies = base.read(checkpoints) if checkpoints.exists() else {'frozen': None}
    for arm, weight in [('scalar', 0.), ('guided', config['guidance_weight'])]:
        if arm in policies:
            continue
        train = out / (prefix + arm + '_' + uuid.uuid4().hex[:8])
        run('train_extractor.py', '--plan', out / 'training.json', '--out', train,
            '--guidance-weight', weight, '--worker-script', worker)
        summary = base.read(train / 'summary.json')
        if arm == 'guided' and not summary['groups_with_positive_guidance_kl']:
            raise RuntimeError(f'No update received a positive feedback KL; inspect {train}')
        policies[arm] = summary['remote_checkpoint']
        base.write_json(checkpoints, policies)
    queries = base.read(out / f'{phase}_queries.json')
    patients = config['cohorts'][phase]['patients']
    raw = (base.OUT / 'semantic/evaluation' if phase == 'pilot'
           else base.ROOT / 'results/qwen27b_rsi/transfer_p34/raw_scores')
    rows = {'raw': {}}
    for q in queries:
        patient, qid = q.split('_', 1)
        rows['raw'][q] = base.read(raw / patient / f'{qid}_raw.json')['correct']
    for arm in ARMS:
        graph = out / (prefix + phase + '_' + arm)
        if not (graph / 'complete.json').exists():
            run('extract_navigation.py', '--out', graph, '--source-plan', out / f'{phase}_sources.json',
                '--worker-script', worker, *(['--resume', policies[arm]] if policies[arm] else []))
        evaluation = out / f'{phase}_{arm}_qa'
        if not (evaluation / 'summary.json').exists():
            run('evaluate_graph.py', '--navigation', '--state', graph / 'graph_state.json',
                '--out', evaluation, '--queries', out / f'{phase}_queries.json', '--patients', *patients,
                '--raw-scores', raw, *(['--paired-with', out / f'{phase}_frozen_qa'] if arm != 'frozen' else []))
        if base.read(evaluation / 'graph_snapshot.json') != base.read(graph / 'graph_state.json'):
            raise ValueError('Evaluated graph differs from generated graph')
        rows[arm] = {path.stem: base.read(path)['score']['correct']
                     for path in (evaluation / 'answers').glob('*.json')}
    report = paired_result(rows, queries)
    base.write_json(out / f'{phase}_result.json', report)
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'pilot', 'full'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weight', type=float, default=.1, help='Distillation weight at preparation')
    parser.add_argument('--train-plan', type=Path, help='Optional source-anchored training curriculum')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out, args.weight, args.train_plan)
    else:
        execute(args.out, args.stage)


if __name__ == '__main__':
    main()
