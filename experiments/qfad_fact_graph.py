"""QFAD graph whose fact store and relation edges consume writer output."""
from collections import defaultdict
import re
import numpy as np
import query_writer_rl as base
from fact_writer_protocol import fact_text


class FactGraph:
    def __init__(self, patient, stage, replacements, embedding_cache):
        self.patient = patient
        self.stage = stage
        self.embedding_cache = embedding_cache
        visible = {text for s, text in patient.history if s <= stage}
        chunks, entities, facts = (patient.stores[n] for n in ('chunk', 'entity', 'fact'))
        self.ci = [i for i,t in enumerate(chunks['texts']) if t in visible]
        self.source_ids = [chunks['ids'][i] for i in self.ci]
        patch = {r['source_id']: r['facts'] for r in replacements}
        assert set(patch) <= set(self.source_ids)
        self.facts = []
        vectors = []
        for local, index in enumerate(self.ci):
            source = chunks['ids'][index]
            if source in patch:
                for fact in patch[source]:
                    self.facts.append({**fact, 'source_id': source, 'passage_index': local})
                    vectors.append(None)
            else:
                for triple in patient.openie[chunks['texts'][index]].get('extracted_triples', []):
                    if len(triple) != 3:
                        continue
                    triple = tuple(base.text_processing(triple))
                    fi = facts['lookup'].get(str(triple))
                    if fi is None:
                        continue
                    self.facts.append({'subject': triple[0], 'relation': triple[1], 'object': triple[2],
                                       'source_id': source, 'passage_index': local, 'evidence': ''})
                    vectors.append(facts['vectors'][fi])
        new_texts = [fact_text(f) for f,v in zip(self.facts,vectors) if v is None]
        new_vectors = iter(self.embed(new_texts))
        self.fact_vectors = np.stack([next(new_vectors) if v is None else v for v in vectors])
        names = sorted({str(f[k]).lower().strip() for f in self.facts for k in ('subject','object')})
        self.entity_names = names
        self.entity_index = {name:i for i,name in enumerate(names)}
        missing = [name for name in names if name not in entities['lookup']]
        extra = dict(zip(missing, self.embed(missing)))
        entity_vectors = np.stack([entities['vectors'][entities['lookup'][name]] if name in entities['lookup']
                                   else extra[name] for name in names])
        self.ne = len(names)
        edges = defaultdict(list)
        for fi,fact in enumerate(self.facts):
            a,b = [self.entity_index[str(fact[k]).lower().strip()] for k in ('subject','object')]
            passage = self.ne+fact['passage_index']
            for left,right in {(a,passage),(b,passage),(a,b)}:
                if left != right:
                    edges[tuple(sorted((left,right)))].append(fi)
        self.graph = base.ig.Graph(n=self.ne+len(self.ci), edges=list(edges))
        self.graph.es['weight'] = [float(len(fis)) for fis in edges.values()]
        self.graph.es['fact_indices'] = list(edges.values())
        node_names = [f'entity:{i}' for i in range(self.ne)]+self.source_ids
        self.graph.vs['name'] = node_names
        self.node_index = {name:i for i,name in enumerate(node_names)}
        self.node_vectors = np.concatenate([entity_vectors,chunks['vectors'][self.ci]])
        self.passage_indices = list(range(self.ne,len(node_names)))
        self.replacements = replacements

    def embed(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.embedding_cache))
        for start in range(0,len(missing),32):
            batch = missing[start:start+32]
            response = base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=batch,encoding_format='float')
            for text,row in zip(batch,response.data):
                self.embedding_cache[text] = base.norm(row.embedding)
        return [self.embedding_cache[t] for t in texts]

    def fact_document(self, index):
        fact=self.facts[index]
        source=self.patient.stores['chunk']['texts'][self.ci[fact['passage_index']]]
        document=f'{source.partition(chr(10))[0]}\n{fact_text(fact)}\nSource: {fact["source_id"]}'
        if fact['evidence']:
            document+='\nEvidence: '+fact['evidence']
        return document

    def resolve_path(self, documents):
        source_ids={line.removeprefix('Source: ') for doc in documents for line in doc.splitlines() if line.startswith('Source: ')}
        wanted=set(documents);found={}
        for index,fact in enumerate(self.facts):
            if fact['source_id'] in source_ids:
                document=self.fact_document(index)
                if document in wanted:found[document]=index
        return [found[doc] for doc in documents] if all(doc in found for doc in documents) else None

    def retrieve(self, question, candidate_limit=128, document_budget=32000, result_limit=15, anchor_source=None):
        qe = self.embed([question])[0]
        similarities = self.fact_vectors @ qe
        dense = self.node_vectors[self.ne:] @ qe
        seeds = np.zeros(len(self.node_index))
        seeds[self.passage_indices] = .05*(dense-dense.min())/max(float(np.ptp(dense)),1e-8)
        for i in np.argsort(similarities)[-15:]:
            fact = self.facts[i]
            for key in ('subject','object'):
                node = self.entity_index[str(fact[key]).lower().strip()]
                seeds[node] += max(0.,float(similarities[i]))/max(1,self.graph.degree(node))
        order,mass = base.run_igraph_qafd(self.graph,self.node_index,self.passage_indices,seeds,
            {name:self.node_vectors[index] for name,index in self.node_index.items()},qe,
            alpha=1.5,epsilon=.01,max_iterations=500,step_size=.2,weight_scheme='multiply',random_seed=42)
        sources = set(order[:128].tolist())
        candidates = [int(i) for i in np.argsort(-similarities,kind='stable') if self.facts[i]['passage_index'] in sources]
        if anchor_source is not None:
            anchored=[int(i) for i in np.argsort(-similarities,kind='stable') if self.facts[i]['source_id']==anchor_source]
            candidates=anchored+[i for i in candidates if i not in anchored]
        selected=[]; documents=[];cost=0
        for i in candidates:
            document=self.fact_document(i)
            size=len(base.ENC.encode(document))
            if cost+size>document_budget:
                continue
            selected.append(i);documents.append(document);cost+=size
            if len(selected)==candidate_limit:
                break
        ranked=base.rerank(question,documents)
        chosen=ranked[:result_limit]
        if anchor_source is not None:
            anchor_rows=[r for r in ranked if self.facts[selected[r['index']]]['source_id']==anchor_source][:min(5,result_limit)]
            anchor_indices={r['index'] for r in anchor_rows}
            chosen=anchor_rows+[r for r in ranked if r['index'] not in anchor_indices][:result_limit-len(anchor_rows)]
        return {'fact_indices':[selected[r['index']] for r in chosen],
                'documents':[documents[r['index']] for r in chosen],
                'candidate_documents':documents,'candidate_ranking':ranked,
                'candidate_fact_indices':selected,'rerank_document_tokens':cost,
                'anchor_source':anchor_source,
                'graph_passage_order':order[:128].tolist(),'graph_passage_mass':mass[:128].tolist()}

    def manifest(self, source_ids=None):
        changed=set(source_ids) if source_ids is not None else {row['source_id'] for row in self.replacements}
        return {'visible_sources':len(self.source_ids),'entities':self.ne,'facts':len(self.facts),
                'edges':self.graph.ecount(),'written_facts':[f for f in self.facts if f['source_id'] in changed],
                'relation_edges':[{'nodes':[self.entity_names[i] for i in edge.tuple],
                                   'facts':[self.facts[i] for i in edge['fact_indices'] if self.facts[i]['source_id'] in changed]}
                                  for edge in self.graph.es if all(i<self.ne for i in edge.tuple)
                                  and any(self.facts[i]['source_id'] in changed for i in edge['fact_indices'])]}
