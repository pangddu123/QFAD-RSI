"""Budgeted search, graph traversal and evidence selection for the Reader policy."""
import json

from graph_environment import fact_text, parse_action_text, base

READER_PROTOCOL = 'Select observed sources for answerer delivery; OPEN_SOURCE is optional inspection'

READER_PROMPT = '''You are the Reader retrieval policy. Obtain the evidence needed for the question.
Return one JSON action at a time; do not answer the question.
SEARCH: {"operation":"SEARCH","query":"search terms","mode":"semantic|lexical|hybrid","subject_scope":"patient|all","record_date":"YYYY-MM-DD or empty"}
EXPAND_RELATION: {"operation":"EXPAND_RELATION","fact_id":"an observed fact ID"}
OPEN_SOURCE: {"operation":"OPEN_SOURCE","source_ids":["observed source ID",...]}
SELECT_EVIDENCE: {"operation":"SELECT_EVIDENCE","fact_ids":[...],"source_ids":[...]}
STOP: {"operation":"STOP"}
SEARCH obtains relevant facts and source previews. Lexical search preserves exact dates,
names and numbers; semantic search matches meaning; hybrid combines them.
The patient scope excludes records whose headers explicitly concern relatives or friends.
Use all when such records are relevant. A nonempty record_date restricts the recorded consultation
date, not the date of a historical event mentioned in it. Leave it empty or omit it when no such restriction is justified.
EXPAND_RELATION follows stored state-change
relations and facts with the same subject and object. OPEN_SOURCE reads one to three observed
sources per action, returning their full original dialogue.
SELECT_EVIDENCE ends retrieval and sends selected facts and full selected sources to the answerer.
You may select a source from its SEARCH preview without first opening it. Use OPEN_SOURCE when
you need to inspect the full dialogue before deciding. Selected facts already include their
displayed original quotations; omit source_ids when those quotations are sufficient.
All selected source text counts against the evidence and answer-model token budgets.
An empty fact_ids or source_ids list may be omitted.
STOP ends with all obtained evidence.
Only use IDs supplied by tool observations. Preserve patient identity, dates, uncertainty and status.
You have at most four actions. Obtain sufficient evidence within the stated token budget.
The total model-token budget includes your decisions and the fixed answerer's prompt and completion;
leave enough budget for it to use your selected evidence.'''


class ReaderEpisode:
    def __init__(self, graph, question, stage, context_cap=8000, native_budget=None):
        self.graph = graph
        self.question = question
        self.stage = stage
        self.context_cap = context_cap
        self.native_budget = native_budget
        self.visible = graph.visible_facts(stage)
        self.seen_sources = set()
        self.seen_facts = set()
        self.opened_sources = []
        self.selected_sources = None
        self.selected_facts = []
        self.first_exposure = {}
        self.actions = []
        self.rerank_tokens = 0
        self.rerank_native_tokens = 0
        self.retrieval_inputs = []
        self.terminal = False
        self.messages = [{'role': 'system', 'content': READER_PROMPT}, {'role': 'user', 'content':
            json.dumps({'question': question['question'], 'evidence_token_budget': context_cap,
                        'total_model_token_budget': native_budget, 'rerank_document_token_budget': 32000})}]

    def step(self, row, policy_tokens=0):
        if self.terminal or len(self.actions) == 4:
            raise ValueError('Reader episode already ended')
        if not row['finished']:
            raise ValueError('Truncated Reader action')
        action = parse_action_text(row['text'])
        operation = action['operation']
        observation = {}
        index = len(self.actions)
        if operation == 'SEARCH':
            remaining = 32000 - self.rerank_tokens
            if remaining < 1000:
                raise ValueError('Retrieval budget exhausted')
            retrieved = self.graph.retrieve(action['query'], self.stage, self.context_cap,
                                            candidate_budget=min(16000, remaining), search_mode=action['mode'],
                                            subject_scope=action['subject_scope'], record_date=action.get('record_date', ''))
            self.rerank_tokens += retrieved['rerank_document_tokens']
            self.rerank_native_tokens += retrieved['rerank_native_input_tokens']
            self.retrieval_inputs.append({'query': action['query'], 'source_ids': retrieved['rerank_source_ids'],
                                          'native_input_tokens': retrieved['rerank_native_input_tokens']})
            sources = retrieved['candidate_source_ids']
            facts = retrieved['fact_ids']
            self.seen_sources.update(sources)
            self.seen_facts.update(facts)
            self.seen_sources.update(c['source_id'] for fid in facts for c in self.visible[fid]['citations'])
            observation = {'sources': [{'source_id': sid, 'preview': self.graph.sources[sid]['text'][:500]}
                                        for sid in sources],
                           'facts': {fid: self.visible[fid] for fid in facts}}
            for fid in facts:
                self.first_exposure.setdefault(('fact', fid), index)
        elif operation == 'EXPAND_RELATION':
            fid = action['fact_id']
            if fid not in self.seen_facts:
                raise ValueError('Graph traversal must start from an observed fact')
            fact = self.visible[fid]
            neighbors = {other for other, f in self.visible.items()
                         if (f['subject'], f['object']) == (fact['subject'], fact['object'])}
            relations = [r for r in self.graph.state['relations'] if fid in (r['from'], r['to'])
                         and r['from'] in self.visible and r['to'] in self.visible
                         and all(self.graph.sources[c['source_id']]['stage'] <= self.stage for c in r['citations'])]
            for relation in relations:
                neighbors.update([relation['from'], relation['to']])
            self.seen_facts.update(neighbors)
            for other in neighbors:
                self.first_exposure.setdefault(('fact', other), index)
                self.seen_sources.update(c['source_id'] for c in self.visible[other]['citations'])
            observation = {'facts': {other: self.visible[other] for other in sorted(neighbors)}, 'relations': relations}
        elif operation == 'OPEN_SOURCE':
            ids = action['source_ids']
            if not ids or len(ids) > 3 or any(sid not in self.seen_sources for sid in ids):
                raise ValueError('Open one to three observed sources')
            current = sum(len(base.ENC.encode(self.graph.sources[sid]['text'])) for sid in self.opened_sources)
            opened = {}
            for sid in ids:
                if sid in self.opened_sources:
                    continue
                text = self.graph.sources[sid]['text']
                cost = len(base.ENC.encode(text))
                if current + cost <= self.context_cap:
                    current += cost
                    self.opened_sources.append(sid)
                    self.first_exposure.setdefault(('source', sid), index)
                    opened[sid] = text
            observation = {'opened_sources': opened, 'remaining_evidence_tokens': self.context_cap - current}
        elif operation == 'SELECT_EVIDENCE':
            fact_ids, source_ids = action.get('fact_ids', []), action.get('source_ids', [])
            if any(fid not in self.seen_facts for fid in fact_ids):
                raise ValueError('Unobserved fact selected')
            if any(sid not in self.seen_sources for sid in source_ids):
                raise ValueError('Unobserved source selected')
            self.selected_facts = list(dict.fromkeys(fact_ids))
            self.selected_sources = list(dict.fromkeys(source_ids))
            self.terminal = True
        elif operation == 'STOP':
            self.selected_facts = sorted(self.seen_facts)
            self.terminal = True
        else:
            raise ValueError('Unknown Reader action')
        self.actions.append(action)
        self.messages.extend([{'role': 'assistant', 'content': row['text']},
                              {'role': 'user', 'content': json.dumps({'tool_observation': observation,
                                  'actions_remaining': 4 - len(self.actions),
                                  'model_tokens_remaining': self.native_budget - policy_tokens
                                      if self.native_budget is not None else None,
                                  'rerank_document_tokens_remaining': 32000 - self.rerank_tokens}, ensure_ascii=False)}])
        if len(self.actions) == 4:
            self.terminal = True
        return observation

    def finish_first_search(self, observation):
        if len(self.actions) != 1 or self.actions[0]['operation'] != 'SEARCH':
            raise ValueError('Direct evidence delivery requires one SEARCH action')
        self.selected_sources = [source['source_id'] for source in observation['sources']]
        self.selected_facts = list(observation['facts'])
        self.terminal = True

    def evidence(self):
        documents, used, facts, sources = [], 0, [], []
        for fid in self.selected_facts:
            fact = self.visible[fid]
            text = f'Graph fact {fid}: {fact_text(fact)}\n' + '\n'.join(
                f'Source {c["source_id"]}: {c["quote"]}' for c in fact['citations'])
            cost = len(base.ENC.encode(text))
            if used + cost <= self.context_cap:
                used += cost
                facts.append(fid)
                documents.append(text)
        source_ids = self.opened_sources if self.selected_sources is None else self.selected_sources
        for sid in source_ids:
            text = f'Source {sid}:\n{self.graph.sources[sid]["text"]}'
            cost = len(base.ENC.encode(text))
            if used + cost <= self.context_cap:
                used += cost
                sources.append(sid)
                documents.append(text)
        return {'documents': documents, 'source_ids': sources, 'fact_ids': facts, 'protocol': READER_PROTOCOL,
                'source_delivery': [{'source_id': sid, 'read_by_policy': sid in self.opened_sources,
                                     'delivered_to_answerer': True} for sid in sources],
                'context_tokens': used, 'rerank_document_tokens': self.rerank_tokens,
                'rerank_native_input_tokens': self.rerank_native_tokens, 'retrieval_inputs': self.retrieval_inputs,
                'first_exposure': [{'kind': key[0], 'id': key[1], 'action_index': value}
                                   for key, value in self.first_exposure.items()], 'actions': self.actions}
