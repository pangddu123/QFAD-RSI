"""Source-scoped atomic graph operations and QFAD retrieval over original dialogue."""
from copy import deepcopy
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import query_writer_rl as base
from calibrate_grounding import check_source
from retrieval_tokens import rerank_tokens
from source_verdicts import VERDICTS, verification_key

FIELDS = ('subject', 'relation', 'object', 'value', 'unit', 'frequency', 'polarity',
          'certainty', 'status', 'event_time', 'record_time')


class QueryBudgetExceeded(ValueError):
    """The policy and evidence leave insufficient tokens for the query."""


WRITER_PROMPT = '''You are the Writer of a longitudinal medical-memory graph.
Use the query feedback to improve one reusable fact or relation using only the supplied original dialogue.
Return one JSON action. Available actions:
ADD_FACT: {"operation":"ADD_FACT","source_id":"...","evidence_ids":[0],"fact":{...}}
CORRECT_FACT: same fields plus "target_id", replacing an incorrectly represented existing fact.
LINK_SOURCE: {"operation":"LINK_SOURCE","target_id":"...","source_id":"...","evidence_ids":[0]}
LINK_STATE_CHANGE: same fields plus "predecessor_id", connecting an earlier state to target_id.
NOOP: {"operation":"NOOP"}.
Every fact has these string fields: subject, relation, object, value, unit, frequency,
polarity, certainty, status, event_time, record_time. Use empty strings for unstated qualifiers.
Polarity is affirmed or negated. Preserve suspected, recommended, reported or confirmed status.
Distinguish the patient from relatives and the clinician. Preserve numbers and units.
A recommendation is not a completed action. A stopped medication is a historical state change;
do not erase the earlier use. Keep event time separate from record time and preserve relative wording.
Preserve partial or temporary effects. A clinician's shorthand must not replace the patient's
more qualified firsthand report; attribute interpretations to the clinician who made them.
Cite source span IDs that substantiate every populated field. No reference answer is available.
No clinical inference beyond the source. Use NOOP if no supported improvement is available.'''


def fact_text(fact):
    return ' | '.join(f'{key}: {fact[key]}' for key in FIELDS if fact.get(key))


def parse_action_text(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    action = json.loads(text)
    if not isinstance(action, dict):
        raise ValueError('Action must be a JSON object')
    return action


class GraphEnvironment:
    def __init__(self, patient, state=None, embeddings=None):
        self.patient = patient
        self.state = deepcopy(state or {'facts': {}, 'relations': []})
        self.embeddings = embeddings if embeddings is not None else {}
        chunks = patient.stores['chunk']
        first_stage = {}
        for stage, text in patient.history:
            first_stage.setdefault(text, stage)
        self.sources = {sid: {'source_id': sid, 'text': text, 'stage': first_stage[text], 'index': i}
                        for i, (sid, text) in enumerate(zip(chunks['ids'], chunks['texts']))
                        if text in first_stage}

    def embed(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.embeddings))
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            result = base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',
                input=batch, encoding_format='float')
            ledger = base.ROOT / 'results/qwen27b_rsi/embedding_usage'
            ledger.mkdir(parents=True, exist_ok=True)
            with (ledger / f'{os.getpid()}.jsonl').open('a') as handle:
                handle.write(json.dumps({'time': time.time(), 'input_count': len(batch),
                    'native_usage': result.usage.model_dump() if result.usage is not None else None,
                    'cl100k_input_tokens': sum(len(base.ENC.encode(text)) for text in batch)}) + '\n')
            for text, row in zip(batch, result.data):
                self.embeddings[text] = base.norm(row.embedding)
        return np.array([self.embeddings[t] for t in texts])

    def visible_facts(self, stage):
        return {fid: fact for fid, fact in self.state['facts'].items()
                if all(self.sources[c['source_id']]['stage'] <= stage for c in fact['citations'])}

    def observation(self, question, source_ids, stage, feedback=None):
        sources = []
        for sid in source_ids:
            source = self.sources[sid]
            assert source['stage'] <= stage
            sources.append({'source_id': sid, 'spans': [
                {'id': i, 'text': line} for i, line in enumerate(source['text'].splitlines()) if line.strip()]})
        visible = self.visible_facts(stage)
        facts = {fid: f for fid, f in visible.items()
                 if any(c['source_id'] in source_ids for c in f['citations'])}
        other_facts = [(fid, f) for fid, f in visible.items() if fid not in facts]
        if other_facts:
            vectors = self.embed([question] + [fact_text(f) for _, f in other_facts])
            order = np.argsort(-(vectors[1:] @ vectors[0]), kind='stable')[:4]
            facts.update(other_facts[int(i)] for i in order)
        relations = [edge for edge in self.state['relations']
                     if edge['from'] in facts and edge['to'] in facts
                     and all(self.sources[c['source_id']]['stage'] <= stage for c in edge['citations'])]
        return [{'role': 'system', 'content': WRITER_PROMPT}, {'role': 'user', 'content': json.dumps({
            'query': question, 'feedback': feedback or 'Inspect the evidence needed by this query and improve the graph.',
            'sources': sources, 'existing_facts': facts, 'existing_relations': relations}, ensure_ascii=False)}]

    def parse(self, row, allowed_sources, stage):
        if not row['finished']:
            raise ValueError('Truncated action')
        action = parse_action_text(row['text'])
        operation = action['operation']
        if operation == 'NOOP':
            return action
        if operation not in ('ADD_FACT', 'CORRECT_FACT', 'LINK_SOURCE', 'LINK_STATE_CHANGE'):
            raise ValueError('Unknown graph operation')
        sid = action['source_id']
        if sid not in allowed_sources or self.sources[sid]['stage'] > stage:
            raise ValueError('Source was not observed at this query cutoff')
        spans = self.sources[sid]['text'].splitlines()
        ids = action['evidence_ids']
        if not ids or any(type(i) is not int or i < 0 or i >= len(spans) or not spans[i].strip() for i in ids):
            raise ValueError('Invalid evidence span')
        action['citations'] = [{'source_id': sid, 'quote': spans[i]} for i in ids]
        visible = self.visible_facts(stage)
        if operation != 'ADD_FACT' and action['target_id'] not in visible:
            raise ValueError('Target fact is not visible')
        if operation in ('ADD_FACT', 'CORRECT_FACT'):
            fact = action['fact']
            if isinstance(fact, dict):
                fact.pop('citations', None)
            if set(fact) != set(FIELDS) or any(not isinstance(fact[k], str) for k in FIELDS):
                raise ValueError('Fact fields do not match the schema')
            if any(not fact[k].strip() for k in ('subject', 'relation', 'object')):
                raise ValueError('Empty subject, relation or object')
            if fact['polarity'] not in ('affirmed', 'negated'):
                raise ValueError('Invalid polarity')
        if operation == 'LINK_STATE_CHANGE' and action['predecessor_id'] not in visible:
            raise ValueError('Predecessor fact is not visible')
        return action

    def verify(self, action, judge):
        if action['operation'] == 'NOOP':
            return {'supported': True, 'validator': 'noop'}
        return VERDICTS.check(verification_key(action, self.state, judge.model),
                              lambda: self._verify_source(action, judge))

    def _verify_source(self, action, judge):
        operation = action['operation']
        if operation in ('ADD_FACT', 'CORRECT_FACT'):
            claim = dict(action['fact'])
        else:
            claim = {k: self.state['facts'][action['target_id']][k] for k in FIELDS}
            if operation == 'LINK_STATE_CHANGE':
                before = self.state['facts'][action['predecessor_id']]
                claim = {k: '' for k in FIELDS}
                claim.update(subject=before['subject'], relation='had a state change from',
                    object=f'{fact_text(before)} TO {fact_text(self.state["facts"][action["target_id"]])}',
                    polarity='affirmed')
        claim['citations'] = action['citations']
        source = self.sources[action['source_id']]
        verdict = check_source(claim, {'source_id': source['source_id'], 'text': source['text']}, judge)
        if verdict['supported'] and operation == 'CORRECT_FACT':
            old = self.state['facts'][action['target_id']]
            response = judge.chat(messages=[{'role': 'system', 'content':
                'Check a correction to a medical graph. Return JSON with supported (boolean) and reason. '
                'A correction must repair the representation of the SAME event or assertion. '
                'Reject replacing an earlier true medication state, measurement or event with a later state. '
                'Such a transition requires adding a separate fact and linking it. '
                'Use only the supplied original sources and citations.'}, {'role': 'user', 'content':
                json.dumps({'old_fact': old, 'replacement': claim,
                    'sources': [self.sources[sid]['text'] for sid in dict.fromkeys(
                        [c['source_id'] for c in old['citations']] + [action['source_id']])]}, ensure_ascii=False)}],
                max_tokens=2048, reasoning_effort='high', response_format={'type': 'json_object'})
            correction = json.loads(response.content)
            assert type(correction['supported']) is bool
            verdict = {'supported': correction['supported'], 'fact_support': verdict,
                       'correction_support': correction, 'usage': response.usage}
        return verdict

    def apply(self, action, verdict):
        if verdict.get('supported') is not True:
            raise ValueError('Unsupported actions cannot change the graph')
        child = GraphEnvironment(self.patient, self.state, self.embeddings)
        operation = action['operation']
        if operation == 'NOOP':
            return child
        if operation in ('ADD_FACT', 'CORRECT_FACT'):
            fid = action.get('target_id') if operation == 'CORRECT_FACT' else f'p{self.patient.persona}-f{len(child.state["facts"])}'
            child.state['facts'][fid] = {**action['fact'], 'citations': action['citations']}
        elif operation == 'LINK_SOURCE':
            citations = child.state['facts'][action['target_id']]['citations']
            citations.extend(c for c in action['citations'] if c not in citations)
        elif operation == 'LINK_STATE_CHANGE':
            edge = {'from': action['predecessor_id'], 'to': action['target_id'],
                    'relation': 'state_change', 'citations': action['citations']}
            if edge not in child.state['relations']:
                child.state['relations'].append(edge)
        return child

    def retrieve(self, question, stage, context_cap=8000, candidate_budget=32000,
                 search_mode='semantic', subject_scope='all', record_date=''):
        if search_mode not in ('semantic', 'lexical', 'hybrid'):
            raise ValueError('Unknown search mode')
        if subject_scope not in ('all', 'patient'):
            raise ValueError('Unknown subject scope')
        sources = [s for s in self.sources.values() if s['stage'] <= stage]
        if subject_scope == 'patient':
            sources = [s for s in sources if not s['text'].startswith('[Health consultation record about ')]
        if record_date:
            sources = [s for s in sources if s['text'].startswith(f'[Session time: {record_date}]')]
        if not sources:
            return {'documents': [], 'source_ids': [], 'candidate_source_ids': [], 'fact_ids': [],
                    'context_tokens': 0, 'rerank_document_tokens': 0, 'rerank_document_count': 0,
                    'rerank_native_input_tokens': 0, 'rerank_source_ids': [],
                    'search_mode': search_mode, 'subject_scope': subject_scope, 'record_date': record_date}
        source_index = {s['source_id']: i for i, s in enumerate(sources)}
        vectors = self.patient.stores['chunk']['vectors'][[s['index'] for s in sources]]
        query = self.embed([question])[0]
        dense = vectors @ query
        order = np.argsort(-dense, kind='stable')
        facts = {fid: fact for fid, fact in self.visible_facts(stage).items()
                 if all(c['source_id'] in source_index for c in fact['citations'])}
        fact_items = list(facts.items())
        fact_order = []
        if facts:
            fact_vectors = self.embed([fact_text(f) for _, f in fact_items])
            scores = fact_vectors @ query
            fact_order = np.argsort(-scores, kind='stable').tolist()
            names = sorted({f[k].lower().strip() for f in facts.values() for k in ('subject', 'object')})
            entity_index = {name: i for i, name in enumerate(names)}
            ne = len(names)
            edges = []
            for fact in facts.values():
                a, b = [entity_index[fact[k].lower().strip()] for k in ('subject', 'object')]
                if a != b:
                    edges.append((a, b))
                for citation in fact['citations']:
                    source = ne + source_index[citation['source_id']]
                    edges.extend([(a, source), (b, source)])
            graph = base.ig.Graph(n=ne + len(sources), edges=edges)
            graph.es['weight'] = [1.] * len(edges)
            node_names = [f'entity:{i}' for i in range(ne)] + [s['source_id'] for s in sources]
            graph.vs['name'] = node_names
            node_vectors = np.concatenate([self.embed(names), vectors])
            seeds = np.zeros(ne + len(sources))
            seeds[ne:] = .05 * (dense - dense.min()) / max(float(np.ptp(dense)), 1e-8)
            for fi in fact_order[:15]:
                for key in ('subject', 'object'):
                    node = entity_index[fact_items[fi][1][key].lower().strip()]
                    seeds[node] += max(0., float(scores[fi])) / max(1, graph.degree(node))
            order, _ = base.run_igraph_qafd(graph, {name: i for i, name in enumerate(node_names)},
                list(range(ne, ne + len(sources))), seeds,
                dict(zip(node_names, node_vectors)), query, alpha=1.5, epsilon=.01,
                max_iterations=500, step_size=.2, weight_scheme='multiply', random_seed=42)
        if search_mode != 'semantic':
            pattern = r'\d{4}-\d{2}-\d{2}|[a-z]+|\d+(?:\.\d+)?'
            cache = self.patient.__dict__.get('rsi_lexical_index')
            if cache is None:
                cache = {sid: Counter(re.findall(pattern, source['text'].lower()))
                         for sid, source in self.sources.items()}
                self.patient.rsi_lexical_index = cache
            terms = set(re.findall(pattern, question.lower()))
            counts = [cache[s['source_id']] for s in sources]
            lengths = np.array([sum(c.values()) for c in counts])
            normalizer = 1.5 * (.25 + .75 * lengths / max(1., float(lengths.mean())))
            lexical = np.zeros(len(sources))
            for term in terms:
                frequencies = np.array([c.get(term, 0) for c in counts])
                df = int((frequencies > 0).sum())
                idf = np.log(1 + (len(sources) - df + .5) / (df + .5))
                lexical += idf * frequencies * 2.5 / (frequencies + normalizer)
            lexical_order = np.argsort(-(lexical + 1e-8 * dense), kind='stable')
            if search_mode == 'lexical':
                order = lexical_order
            else:
                fusion = np.zeros(len(sources))
                for ranking in (order, lexical_order):
                    fusion[ranking] += 1. / (60 + np.arange(len(ranking)))
                order = np.argsort(-fusion, kind='stable')
        candidates, candidate_tokens = [], 0
        for i in order:
            size = len(base.ENC.encode(sources[int(i)]['text']))
            if candidate_tokens + size > candidate_budget:
                continue
            candidates.append(sources[int(i)])
            candidate_tokens += size
            if len(candidates) == 48:
                break
        candidate_texts = [s['text'] for s in candidates]
        native_rerank_tokens = rerank_tokens(base.rerank_request(question, candidate_texts))
        ranked = base.rerank(question, candidate_texts)
        selected = [candidates[r['index']] for r in ranked[:6]]
        documents, selected_fact_ids, used = [], [], 0
        for fi in fact_order[:8]:
            fid, fact = fact_items[fi]
            text = f'Graph fact {fid}: {fact_text(fact)}\n' + '\n'.join(
                f'Source {c["source_id"]}: {c["quote"]}' for c in fact['citations'])
            cost = len(base.ENC.encode(text))
            if used + cost <= min(context_cap // 3, 2000):
                documents.append(text)
                selected_fact_ids.append(fid)
                used += cost
        for edge in self.state['relations']:
            if edge['from'] not in facts or edge['to'] not in facts:
                continue
            if not all(self.sources[c['source_id']]['stage'] <= stage for c in edge['citations']):
                continue
            if not {edge['from'], edge['to']} & set(selected_fact_ids):
                continue
            text = f'State transition: {fact_text(facts[edge["from"]])} -> {fact_text(facts[edge["to"]])}\n'
            text += '\n'.join(f'Source {c["source_id"]}: {c["quote"]}' for c in edge['citations'])
            cost = len(base.ENC.encode(text))
            if used + cost <= min(context_cap // 3, 2000):
                documents.append(text)
                used += cost
        selected_sources = []
        for source in selected:
            text = f'Source {source["source_id"]}:\n{source["text"]}'
            cost = len(base.ENC.encode(text))
            if used + cost <= context_cap:
                documents.append(text)
                selected_sources.append(source['source_id'])
                used += cost
        return {'documents': documents, 'source_ids': selected_sources,
                'candidate_source_ids': [s['source_id'] for s in selected],
                'fact_ids': selected_fact_ids, 'context_tokens': used,
                'search_mode': search_mode,
                'subject_scope': subject_scope, 'record_date': record_date,
                'rerank_document_tokens': candidate_tokens, 'rerank_document_count': len(candidates),
                'rerank_native_input_tokens': native_rerank_tokens,
                'rerank_source_ids': [s['source_id'] for s in candidates]}
