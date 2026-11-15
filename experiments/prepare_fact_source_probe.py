"""Select dated and longitudinal source evidence within the fixed reranking budget."""
import re
import numpy as np
from qfad_fact_graph import base


def main():
    root = base.ROOT / 'results/qfad_fact_writer_rl_20261114'
    original = base.read(root / 'validation/query_plan.json')
    patient = base.Patient(6)
    chunks = patient.stores['chunk']
    queries = {q['query_id']: q for _, rows in patient.groups for q in rows}
    output = root / 'source_selection_probe'
    plan = []
    requests = {}
    for item in original:
        q = queries[item['query_id']]
        visible = {text for stage, text in patient.history if stage <= item['stage']}
        indices = [i for i, text in enumerate(chunks['texts']) if text in visible and text.startswith('[Session')]
        dates = {i: re.search(r'\d{4}-\d{2}-\d{2}', chunks['texts'][i].splitlines()[0]).group() for i in indices}
        qe = base.norm(item['query_embedding'])
        order = [indices[int(i)] for i in np.argsort(-(chunks['vectors'][indices] @ qe), kind='stable')]
        requested_dates = re.findall(r'\b(?:on|dated)\s+(\d{4}-\d{2}-\d{2})\b', q['question'], re.I)
        dated = [i for i in order if dates[i] in requested_dates]
        recent = []
        if dated:
            order = dated
        elif q['query_type'] == 'state_update':
            latest = max(dates.values())
            recent = [i for i in order if dates[i] == latest][:4]
            order = recent + [i for i in order if i not in recent]
        candidates = []
        tokens = 0
        for index in order:
            cost = len(base.ENC.encode(chunks['texts'][index]))
            if tokens + cost > 16000:
                continue
            candidates.append(index)
            tokens += cost
            if len(candidates) == 16:
                break
        ranked = base.rerank(q['question'], [chunks['texts'][i] for i in candidates])
        chosen = [candidates[r['index']] for r in ranked[:3]]
        if recent:
            newest = next(candidates[r['index']] for r in ranked if candidates[r['index']] in recent)
            chosen = [newest] + [i for i in chosen if i != newest]
            chosen = chosen[:3]
        request_ids = []
        for index in chosen:
            key = f'p6_source{index}'
            text = chunks['texts'][index]
            requests[key] = {'episode_id': key, 'sources': [{'source_id': chunks['ids'][index],
                'text': text, 'old_facts': patient.openie[text].get('extracted_triples', [])}]}
            request_ids.append(key)
        plan.append({**{k: item[k] for k in ('persona', 'stage', 'checkpoint', 'query_id', 'question', 'query_embedding')},
                     'writer_request_ids': request_ids, 'source_ids': [chunks['ids'][i] for i in chosen],
                     'source_rerank_candidates': len(candidates), 'source_rerank_document_tokens': tokens,
                     'source_candidate_ids': [chunks['ids'][i] for i in candidates], 'source_ranking': ranked,
                     'requested_dates': requested_dates, 'latest_source_included': bool(recent),
                     'uses_reference_or_source_annotations': False})
        base.write_json(output / 'writer_inputs.json', list(requests.values()))
        base.write_json(output / 'query_plan.json', plan)
        print(q['query_id'], [dates[i] for i in chosen], flush=True)
    print('queries', len(plan), 'unique source writes', len(requests), flush=True)


if __name__ == '__main__':
    main()
