"""Source-only fact extraction and QFAD navigation to original dialogue windows."""
from copy import deepcopy
import json

import numpy as np

from graph_environment import GraphEnvironment, FIELDS, base, fact_text, parse_action_text
from retrieval_tokens import rerank_tokens


EXTRACTION_PROMPT = '''Extract reusable clinical navigation facts from the original dialogue.
The facts will guide retrieval of the original text; they are not answers to questions.
Return one JSON object with facts and relations arrays. Extract a collection of useful
facts from each supplied source, rather than a single graph edit. Empty arrays are
appropriate when the sources contain no patient-specific information worth indexing.
Each fact has: id (local identifier such as f0), source_id, evidence_ids (nonempty list
of supplied span integers), subject, relation, object, value, unit, frequency,
polarity, certainty, status, event_time, record_time. All fields except evidence_ids
are strings. Use empty strings for unstated qualifiers. Polarity is affirmed or negated.
Keep subject/relation/object concise navigation cues. Details may stay in the cited text.
Preserve conditions that distinguish the event: measurement setting, actual versus planned
treatment, historical versus current use, uncertainty and transient versus sustained change.
Keep separate events when treatment changes or a measurement is repeated. A recommendation
is not an executed action. An absent observation is not a normal result. Retain relative
time wording when the original does not establish an exact date. Distinguish the patient,
relatives and the person making an assertion; a doctor's hypothesis is not a confirmed diagnosis.
Do not add medical conclusions or causal claims that the original text does not support.
Read this block as one visit in a longitudinal record. First identify whose experience is
described and who is speaking. Then identify what was reported, proposed, actually done
or measured, and when. Write only distinctions another reader needs to find the evidence.
For a result that quotes an older value, preserve which value belongs to which encounter;
record units and meal/test timing only when stated. For a medication, keep advice and
reported use independently retrievable; silence in a later visit does not imply discontinuation.
Use a cited earlier fact as a relation endpoint when the text explicitly compares the events.
Keep patient-reported symptoms distinct from a clinician's possible explanation. A warning
to monitor a lab value does not establish an abnormal result. If a reference cannot be
resolved from the supplied record, leave its qualifier empty. Emit only the requested JSON.
Each relation has: from, to, relation, source_id, evidence_ids. Endpoints are local fact
IDs or IDs in existing_facts. Link explicitly supported temporal, state-change, repeated-report
or other clinical relationships. A shared drug name alone does not establish a state change.
Every populated fact field and every relation must be supported by its cited original spans.
The environment attaches the actual source text; do not copy it into the JSON.'''

SUPPORT_PROMPT = '''Check extracted clinical navigation facts and relations against original sources.
Return JSON: {"facts":[{"id":"...","supported":true,"reason":"..."}],
"relations":[{"index":0,"supported":true,"reason":"..."}]}.
Judge every new fact and every new relation. An index may omit details that remain in its
original source. For a rejected atom also return criterion, choosing the principal error
from subject, assertion, event_time, treatment_status, measurement, monitoring,
patient_constraint, longitudinal_change or source_fidelity. In reason, name the
unsupported field or relation endpoint and what the cited lines actually establish.
Treat absent qualifiers as acceptable when they do not change the stated meaning.
The index must not contradict the source, misattribute a subject, overstate certainty,
turn a recommendation into execution, confuse event and recording time, or invent a causal link.
Citations must support the stated interpretation. Old facts are supplied only as endpoints;
their presence alone does not prove a new relationship. Use the original dialogue, including
which speaker made each assertion. Do not judge whether the dialogue's medical advice is correct.
Do not answer a medical question. Return exactly one result for each supplied new fact/relation.'''


def extraction_messages(sources, existing_facts=None):
    return [{'role': 'system', 'content': EXTRACTION_PROMPT}, {'role': 'user', 'content': json.dumps({
        'sources': [{'source_id': s['source_id'], 'spans': [
            {'id': i, 'text': line} for i, line in enumerate(s['text'].splitlines()) if line.strip()]}
            for s in sources],
        'existing_facts': existing_facts or {},
    }, ensure_ascii=False)}]


def parse_extraction(row, sources, existing_facts=None):
    if not row['finished']:
        raise ValueError('Extraction was truncated')
    result = parse_action_text(row['text'])
    if set(result) != {'facts', 'relations'}:
        raise ValueError('Extraction must contain facts and relations')
    if not isinstance(result['facts'], list) or not isinstance(result['relations'], list):
        raise ValueError('Facts and relations must be arrays')
    lookup = {s['source_id']: s for s in sources}
    def citations(item):
        source = lookup[item['source_id']]
        spans = source['text'].splitlines()
        ids = item['evidence_ids']
        if not isinstance(ids, list) or not ids or any(
                type(i) is not int or i < 0 or i >= len(spans) or not spans[i].strip() for i in ids):
            raise ValueError('Evidence must refer to supplied source spans')
        return [{'source_id': source['source_id'], 'span_id': i, 'quote': spans[i]} for i in sorted(set(ids))]
    ids = set()
    for fact in result['facts']:
        if set(fact) - {'citations'} != set(FIELDS) | {'id', 'source_id', 'evidence_ids'}:
            raise ValueError('Unexpected fact fields')
        if any(not isinstance(fact[k], str) for k in (*FIELDS, 'id', 'source_id')):
            raise ValueError('Fact attributes must be strings')
        if any(not fact[k].strip() for k in ('id', 'subject', 'relation', 'object')):
            raise ValueError('Empty fact identifier or proposition')
        if fact['id'] in ids or fact['id'] in (existing_facts or {}):
            raise ValueError('Duplicate fact identifier')
        if fact['polarity'] not in ('affirmed', 'negated'):
            raise ValueError('Invalid polarity')
        ids.add(fact['id'])
        fact['citations'] = citations(fact)
    endpoints = ids | set(existing_facts or {})
    for relation in result['relations']:
        if set(relation) - {'citations'} != {'from', 'to', 'relation', 'source_id', 'evidence_ids'}:
            raise ValueError('Unexpected relation fields')
        if relation['from'] not in endpoints or relation['to'] not in endpoints:
            raise ValueError('Unknown relation endpoint')
        if relation['from'] == relation['to'] or not isinstance(relation['relation'], str) or not relation['relation'].strip():
            raise ValueError('Empty or self relation')
        relation['citations'] = citations(relation)
    return result


def verify_extraction(extraction, sources, existing_facts, judge):
    if not extraction['facts'] and not extraction['relations']:
        return {'supported': True, 'facts': [], 'relations': [], 'usage': {'total_tokens': 0}}
    response = judge.chat(messages=[{'role': 'system', 'content': SUPPORT_PROMPT},
        {'role': 'user', 'content': json.dumps({'sources': sources, 'new': extraction,
            'existing_facts': existing_facts}, ensure_ascii=False)}], max_tokens=6000,
        reasoning_effort='high', response_format={'type': 'json_object'})
    verdict = json.loads(response.content)
    assert len(verdict['facts']) == len(extraction['facts'])
    assert {r['id'] for r in verdict['facts']} == {r['id'] for r in extraction['facts']}
    assert len(verdict['relations']) == len(extraction['relations'])
    assert {r['index'] for r in verdict['relations']} == set(range(len(extraction['relations'])))
    assert all(type(r['supported']) is bool for r in verdict['facts'] + verdict['relations'])
    return {**verdict, 'supported': all(r['supported'] for r in verdict['facts'] + verdict['relations']),
            'usage': response.usage}


def admit_extraction(extraction, verdict, existing_facts=None):
    """Keep supported atoms and relations whose endpoints remain available."""
    fact_verdicts = {r['id']: r for r in verdict['facts']}
    relation_verdicts = {r['index']: r for r in verdict['relations']}
    assert set(fact_verdicts) == {f['id'] for f in extraction['facts']}
    assert set(relation_verdicts) == set(range(len(extraction['relations'])))
    facts, relations, rejected = [], [], []
    for fact in extraction['facts']:
        check = fact_verdicts[fact['id']]
        if check['supported']:
            facts.append(deepcopy(fact))
        else:
            rejected.append({'kind': 'fact', 'id': fact['id'], 'reason': check['reason'],
                             'criterion': check.get('criterion', 'source_fidelity')})
    endpoints = {f['id'] for f in facts} | set(existing_facts or {})
    for i, relation in enumerate(extraction['relations']):
        check = relation_verdicts[i]
        missing = [relation[k] for k in ('from', 'to') if relation[k] not in endpoints]
        if check['supported'] and not missing:
            relations.append(deepcopy(relation))
        else:
            rejected.append({'kind': 'relation', 'index': i,
                             'criterion': check.get('criterion', 'source_fidelity'),
                             'reason': check['reason'] if not check['supported'] else 'Endpoint was not admitted',
                             'unavailable_endpoints': missing})
    return {'facts': facts, 'relations': relations}, {'rejected': rejected,
        'proposed_facts': len(extraction['facts']), 'accepted_facts': len(facts),
        'proposed_relations': len(extraction['relations']), 'accepted_relations': len(relations)}


class NavigationGraph(GraphEnvironment):
    def resolve_citation(self, citation):
        """Return the original line, rejecting stale pointers in a saved graph."""
        lines = self.sources[citation['source_id']]['text'].splitlines()
        index = citation.get('span_id')
        if index is None:
            matches = [i for i, line in enumerate(lines) if line == citation['quote']]
            if len(matches) != 1:
                raise ValueError('Legacy quote cannot be localized to one source line')
            index = matches[0]
        if type(index) is not int or not 0 <= index < len(lines) or lines[index] != citation['quote']:
            raise ValueError('Citation disagrees with its original source line')
        return index, lines[index]

    def extract_into(self, extraction, source_ids):
        state = deepcopy(self.state)
        removed = {fid for fid, fact in state['facts'].items()
                   if any(c['source_id'] in source_ids for c in fact['citations'])}
        state['facts'] = {fid: f for fid, f in state['facts'].items() if fid not in removed}
        state['relations'] = [r for r in state['relations']
                              if r['from'] not in removed and r['to'] not in removed
                              and not any(c['source_id'] in source_ids for c in r['citations'])]
        aliases = {f['id']: f'{f["source_id"]}:fact:{i}' for i, f in enumerate(extraction['facts'])}
        for fact in extraction['facts']:
            state['facts'][aliases[fact['id']]] = {k: fact[k] for k in (*FIELDS, 'citations')}
        for relation in extraction['relations']:
            if any(relation[key] in removed and relation[key] not in aliases for key in ('from', 'to')):
                raise ValueError('Relation refers to an old fact replaced by this extraction')
            item = {k: relation[k] for k in ('from', 'to', 'relation', 'citations')}
            item['from'] = aliases.get(item['from'], item['from'])
            item['to'] = aliases.get(item['to'], item['to'])
            if item['from'] not in state['facts'] or item['to'] not in state['facts']:
                raise ValueError('Relation refers to a replaced old fact')
            state['relations'].append(item)
        return NavigationGraph(self.patient, state, self.embeddings)

    def retrieve(self, question, stage, context_cap=8000):
        facts = self.visible_facts(stage)
        if not facts:
            return {'documents': [], 'source_ids': [], 'fact_ids': [], 'source_windows': [],
                    'context_tokens': 0, 'rerank_document_tokens': 0,
                    'rerank_native_input_tokens': 0, 'method': 'qfad_fact_navigation'}
        fact_ids = list(facts)
        relations = [r for r in self.state['relations']
                     if r['from'] in facts and r['to'] in facts and all(
                         self.sources[c['source_id']]['stage'] <= stage for c in r['citations'])]
        source_ids = list(dict.fromkeys(c['source_id'] for item in [*facts.values(), *relations]
                                       for c in item['citations']))
        entities = sorted({f[k].lower().strip() for f in facts.values() for k in ('subject', 'object')})
        nf, ne = len(fact_ids), len(entities)
        fi = {key: i for i, key in enumerate(fact_ids)}
        ei = {key: nf + i for i, key in enumerate(entities)}
        si = {key: nf + ne + i for i, key in enumerate(source_ids)}
        edges = set()
        for fid, fact in facts.items():
            edges.update(tuple(sorted((fi[fid], ei[fact[k].lower().strip()]))) for k in ('subject', 'object'))
            edges.update((fi[fid], si[c['source_id']]) for c in fact['citations'])
        for relation in relations:
            edges.add(tuple(sorted((fi[relation['from']], fi[relation['to']]))))
            for citation in relation['citations']:
                edges.update((fi[relation[key]], si[citation['source_id']]) for key in ('from', 'to'))
        names = fact_ids + ['entity:' + name for name in entities] + source_ids
        graph = base.ig.Graph(n=len(names), edges=sorted(edges))
        graph.vs['name'] = names
        graph.es['weight'] = [1.] * len(edges)
        query = self.embed([question])[0]
        fact_vectors = self.embed([fact_text(f) for f in facts.values()])
        chunks = self.patient.stores['chunk']
        source_vectors = np.stack([chunks['vectors'][self.sources[sid]['index']] for sid in source_ids])
        vectors = np.concatenate([fact_vectors, self.embed(entities), source_vectors])
        scores = fact_vectors @ query
        seeds = np.zeros(len(names))
        seed_indices = np.argsort(-scores, kind='stable')[:15]
        for index in seed_indices:
            seeds[index] = max(0., float(scores[index])) / max(1, graph.degree(int(index)))
        order, mass = base.run_igraph_qafd(graph, dict(zip(names, range(len(names)))),
            list(range(nf + ne, len(names))), seeds, dict(zip(names, vectors)), query,
            alpha=1.5, epsilon=.01, max_iterations=500, step_size=.2,
            weight_scheme='multiply', random_seed=42)
        candidates, candidate_cost = [], 0
        for index in order:
            sid = source_ids[int(index)]
            cost = len(base.ENC.encode(self.sources[sid]['text']))
            if candidate_cost + cost <= 32000:
                candidates.append(sid)
                candidate_cost += cost
            if len(candidates) == 48:
                break
        texts = [self.sources[sid]['text'] for sid in candidates]
        native_rerank = rerank_tokens(base.rerank_request(question, texts))
        ranked = base.rerank(question, texts)
        documents, selected_sources, windows, used, selected_facts = [], [], [], 0, []
        skipped_sources = []
        for rank in ranked[:6]:
            sid = candidates[rank['index']]
            linked = [fid for fid, f in facts.items() if any(c['source_id'] == sid for c in f['citations'])]
            lines = self.sources[sid]['text'].splitlines()
            indices = {0}
            cited_items = [facts[fid] for fid in linked] + [r for r in relations
                if any(c['source_id'] == sid for c in r['citations'])]
            for item in cited_items:
                for citation in item['citations']:
                    if citation['source_id'] == sid:
                        i, _ = self.resolve_citation(citation)
                        indices.update(range(max(0, i - 1), min(len(lines), i + 2)))
            selected_lines = sorted(indices)
            segments = []
            for index in selected_lines:
                if not segments or index != segments[-1][-1] + 1:
                    segments.append([index])
                else:
                    segments[-1].append(index)
            document = '\n\n'.join(f'Source {sid}, lines {part[0]}–{part[-1]}:\n' +
                                   '\n'.join(lines[i] for i in part) for part in segments)
            cost = len(base.ENC.encode(document))
            if used + cost > context_cap:
                skipped_sources.append(sid)
                continue
            documents.append(document)
            selected_sources.append(sid)
            selected_facts.extend(linked)
            windows.append({'source_id': sid, 'line_ids': selected_lines,
                            'segments': [{'line_ids': part, 'text': '\n'.join(lines[i] for i in part)} for part in segments]})
            used += cost
        return {'documents': documents, 'source_ids': selected_sources, 'fact_ids': selected_facts,
                'source_windows': windows, 'context_tokens': used,
                'rerank_document_tokens': candidate_cost, 'rerank_native_input_tokens': native_rerank,
                'rerank_source_ids': candidates, 'graph_sources': len(source_ids), 'graph_facts': nf,
                'seed_fact_ids': [fact_ids[int(i)] for i in seed_indices if seeds[i] > 0],
                'rerank_top_source_ids': [candidates[r['index']] for r in ranked[:6]],
                'context_skipped_source_ids': skipped_sources,
                'graph_edges': len(edges), 'qfad_source_order': [source_ids[int(i)] for i in order],
                'qfad_source_mass': [float(value) for value in mass], 'method': 'qfad_fact_navigation'}
