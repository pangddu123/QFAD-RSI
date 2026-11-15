"""Run one query-feedback update of the Reader's evidence selection policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re
import numpy as np

from qfad_fact_graph import FactGraph, base
from query_reward import QueryFeedback
from rsi_reader_policy import ReaderPolicy, FEATURES

ROOT = base.ROOT / 'results/qfad_bidirectional_rsi_p12_20261114'


def candidate_features(graph, retrieval, question, query_type):
    qe = graph.embed([question])[0]
    ranking = {r['index']: float(r['relevance_score']) for r in retrieval['candidate_ranking']}
    graph_order = {int(p): i for i, p in enumerate(retrieval['graph_passage_order'])}
    requested = set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', question))
    months = 'January|February|March|April|May|June|July|August|September|October|November|December'
    for text in re.findall(rf'\b(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b', question, re.I):
        requested.add(datetime.strptime(text.replace(',', ''), '%B %d %Y').strftime('%Y-%m-%d'))
    dates = []
    for doc in retrieval['candidate_documents']:
        found = re.search(r'\d{4}-\d{2}-\d{2}', doc.splitlines()[0])
        dates.append(datetime.strptime(found.group(), '%Y-%m-%d').toordinal() if found else 0)
    lo, hi = min(dates), max(dates)
    rows = []
    costs = []
    for local, fact_index in enumerate(retrieval['candidate_fact_indices']):
        fact = graph.facts[fact_index]
        document = retrieval['candidate_documents'][local]
        size = len(base.ENC.encode(document))
        header = document.splitlines()[0]
        rows.append([ranking[local], float(graph.fact_vectors[fact_index] @ qe),
                     1 / (1 + graph_order.get(fact['passage_index'], 128)), float(bool(fact['evidence'])),
                     float(fact['source_id'] == retrieval['anchor_source']),
                     float(any(d in header for d in requested)),
                     (dates[local] - lo) / max(1, hi - lo) if query_type == 'state_update' else 0.,
                     1 / (1 + size / 256)])
        costs.append(size + 10)
    return np.asarray(rows, dtype=np.float32), costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    parser.add_argument('--writes', type=base.Path)
    parser.add_argument('--policy', type=base.Path)
    parser.add_argument('--output', type=base.Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--sample-limits', type=int, nargs=4, default=[15, 15, 15, 15])
    args = parser.parse_args()
    item = next(r for r in base.read(ROOT / 'feedback_pool.json') if r['episode_id'] == args.episode)
    patient = base.Patient(item['persona'])
    q = next(q for _, queries in patient.groups for q in queries if q['query_id'] == item['query_id'])
    writes = base.read(args.writes) if args.writes else []
    visible = {text for stage, text in patient.history if stage <= item['stage']}
    ids = {patient.stores['chunk']['ids'][i] for i, text in enumerate(patient.stores['chunk']['texts']) if text in visible}
    graph = FactGraph(patient, item['stage'], [r for r in writes if r['source_id'] in ids], {})
    retrieval = graph.retrieve(q['question'], anchor_source=item['source_ids'][0])
    matrix, costs = candidate_features(graph, retrieval, q['question'], q['query_type'])
    raw = next(r for r in base.read(base.RAW / f'persona_{item["persona"]:02d}/session_{item["checkpoint"]:03d}_answers.json')
               if r['query_id'] == item['query_id'])
    cap = len(base.ENC.encode(raw['formatted_context']))
    policy = ReaderPolicy()
    if args.policy:
        policy.load(args.policy)
    prior_path = policy.paths.get(args.episode)
    if prior_path and graph.resolve_path(prior_path['documents']) is None:
        del policy.paths[args.episode]
    policy.save(args.output / 'policy_before.pt')
    baseline = policy.replay(args.episode,retrieval['candidate_documents'],costs,cap) or policy.select(matrix, costs, cap)
    assert all(1 <= limit <= 15 for limit in args.sample_limits)
    samples = [policy.select(matrix, costs, cap, limit=limit, seed=71 + i)
               for i, limit in enumerate(args.sample_limits)]
    feedback = QueryFeedback(ROOT / 'reader_cache' / args.episode, base.Scorer('gpt-6.1-sol', 'en'), True)
    request_prefix = '__'.join(args.output.resolve().relative_to(ROOT).parts)

    def answer(selection, name):
        docs = [retrieval['candidate_documents'][i] for i in selection.indices]
        context = '\n\n'.join(f'Memory {i+1}:\n{doc}' for i, doc in enumerate(docs))
        assert len(base.ENC.encode(context)) <= cap
        prompt = base.ANSWER_PROMPT.format(context=context, question=base.PROMPT.format_query(q['question'], q['query_type']))
        result = feedback(q, prompt, f'{request_prefix}_{name}', len(docs))
        record = {'query_id': q['query_id'], 'context': context, 'candidate_indices': selection.indices,
                  'fact_indices': [retrieval['candidate_fact_indices'][i] for i in selection.indices],
                  'context_tokens': len(base.ENC.encode(context)), 'raw_context_tokens': cap, **result}
        base.write_json(args.output / f'{name}.json', record)
        return record

    if args.evaluate_only:
        result=answer(baseline,'baseline')
        base.write_json(args.output/'evaluation.json',{'episode_id':args.episode,
            'policy_file':str(args.policy) if args.policy else None,
            'graph_writes_file':str(args.writes) if args.writes else None,
            'correct':result['semantic_score']['correct'],'primary_correct':result['score']['is_correct'],
            'reward_metric':'fixed_semantic_rubric','answer_model':'Qwen3.8-27B',
            'answer_model_frozen':True,'reader_policy_updated':False})
        return
    jobs = [(baseline, 'baseline')] + [(s, f'sample{i}') for i, s in enumerate(samples)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        answers = list(pool.map(lambda job: answer(*job), jobs))
    quality = lambda row: float(row['semantic_score']['correct']) + .2 * row['query_likelihood']['token_probability']
    baseline_quality = quality(answers[0])
    rewards = [quality(row) for row in answers[1:]]
    rewards = [baseline_quality if abs(r - baseline_quality) < .002 else r for r in rewards]
    update = policy.update(samples, rewards, baseline_quality)
    successful=[i for i,row in enumerate(answers) if row['semantic_score']['correct']]
    if successful:
        best=max(successful,key=lambda i:quality(answers[i]))
        selected=([baseline]+samples)[best]
        policy.remember(args.episode,[retrieval['candidate_documents'][i] for i in selected.indices],quality(answers[best]))
    policy.save(args.output / 'policy_after.pt')
    deployed=policy.replay(args.episode,retrieval['candidate_documents'],costs,cap) or policy.select(matrix,costs,cap)
    after = answer(deployed, 'updated')
    base.write_json(args.output / 'group.json', {'episode_id': args.episode, 'features': FEATURES,
        'candidate_features': matrix.tolist(), 'candidate_costs': costs, 'retrieval': retrieval,
        'update': update, 'answer_model': 'Qwen3.8-27B', 'answer_model_frozen': True,
        'baseline_correct': answers[0]['semantic_score']['correct'],
        'updated_correct': after['semantic_score']['correct'],
        'reward_metric':'fixed_semantic_rubric',
        'sample_fact_limits':args.sample_limits,
        'successful_evidence_path_retained':args.episode in policy.paths,
        'source_selection': 'verified source localized by adaptation feedback'})
    print(update, flush=True)


if __name__ == '__main__':
    main()
