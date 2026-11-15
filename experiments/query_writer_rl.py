"""Query-reward source-node writing policy for QFAD."""
import argparse
import ast
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import types

os.environ['JUDGE_REASONING_EFFORT']='high'
os.environ['JUDGE_MIN_MAX_TOKENS']='16000'

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'baselines/QAFD-RAG/.deps'))

import igraph as ig
import numpy as np
from openai import OpenAI
import pandas as pd
import requests
import tiktoken
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'MedMemoryBench')]
package = types.ModuleType('writer_qfad')
package.__path__ = [str(ROOT / 'QFAD-RSI/src/passage_entity')]
sys.modules['writer_qfad'] = package
from writer_qfad.graph_adapter import run_igraph_qafd
from writer_qfad.utils import text_processing
import qafd_medmemorybench_mixed as harness
from medmemory_scoring import Scorer, write_json
from utils.templates import get_prompt_manager

OUT = ROOT / 'results/qfad_query_writer_rl_20261113'
SOURCE = ROOT / 'results/qafd_medmemorybench_en_mixed_gpt4omini_20261111'
RAW = ROOT / 'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'
ENC = tiktoken.get_encoding('cl100k_base')
SIZES = [256, 512, 1024]
CLIENT = OpenAI(api_key='local', base_url=os.environ.get('QFAD_READER_URL','http://127.0.0.1:19107/v1'), timeout=900)
EMBED = OpenAI(api_key='local', base_url='http://127.0.0.1:19085/v1', timeout=600)
PROMPT = get_prompt_manager(dataset='medmemorybench', method='embedding_rag', language='en')
CONFIG = yaml.safe_load((ROOT.parent/'baselines/RAW-RAG/methods/EverCore/evaluation/config/prompts.yaml').read_text())
ANSWER_PROMPT = CONFIG['online_api']['default']['answer_prompt_cot']


def read(path):
    return json.loads(Path(path).read_text())


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


class Patient:
    def __init__(self, persona, store_names=('chunk', 'entity', 'fact')):
        self.persona = persona
        harness.DATA = ROOT/'MedMemoryBench/data/MedMemoryBench_EN_mixed'
        harness.rawrag.DATA = harness.DATA
        harness.LANGUAGE = harness.rawrag.LANGUAGE = 'en'
        _, groups = harness.load_units([persona])
        self.groups = groups[persona]
        self.history = harness.load_passages([persona])[persona]
        work = SOURCE / f'persona_{persona:02d}/qafd'
        self.openie = ({x['passage']: x for x in read(work/'openie_results_ner_gpt-4o-mini.json')['docs']}
                       if 'entity' in store_names or 'fact' in store_names else {})
        self.stores = {}
        for name in store_names:
            frame = pd.read_parquet(work/f'gpt-4o-mini_qwen3-embedding-4b/{name}_embeddings/vdb_{name}.parquet')
            self.stores[name] = dict(ids=frame.hash_id.tolist(), texts=frame.content.tolist(), vectors=norm(np.stack(frame.embedding)))
            self.stores[name]['lookup'] = {text:i for i,text in enumerate(frame.content)}
        self.views = {}

    def view(self, stage):
        if stage in self.views:
            return self.views[stage]
        visible = set(t for s,t in self.history if s <= stage)
        cs, es, fs = (self.stores[n] for n in ('chunk','entity','fact'))
        ci = [i for i,t in enumerate(cs['texts']) if t in visible]
        entities, facts, links, relations = set(), set(), set(), Counter()
        for i in ci:
            doc = self.openie[cs['texts'][i]]
            for triple in doc.get('extracted_triples', []):
                if len(triple) != 3:
                    continue
                triple = tuple(text_processing(triple))
                fi = fs['lookup'].get(str(triple))
                if fi is None:
                    continue
                facts.add(fi)
                pair = []
                for entity in (triple[0], triple[2]):
                    ei = es['lookup'].get(entity)
                    if ei is not None:
                        entities.add(ei)
                        links.add((cs['ids'][i], es['ids'][ei]))
                        pair.append(es['ids'][ei])
                if len(pair)==2 and pair[0] != pair[1]:
                    relations[tuple(sorted(pair))] += 1
        ei, fi = sorted(entities), sorted(facts)
        names = [es['ids'][i] for i in ei] + [cs['ids'][i] for i in ci]
        name_idx = {n:i for i,n in enumerate(names)}
        edges = list(links) + list(relations)
        graph = ig.Graph(n=len(names), edges=[(name_idx[a],name_idx[b]) for a,b in edges])
        graph.vs['name'] = names
        graph.es['weight'] = [1.0]*len(links) + [float(relations[e]) for e in relations]
        vectors = np.concatenate([es['vectors'][ei], cs['vectors'][ci]])
        cards = []
        for size in SIZES:
            children = []
            for parent, idx in enumerate(ci):
                passage = cs['texts'][idx]
                header, _, body = passage.partition('\n')
                encoded = ENC.encode(body)
                # Each node is written independently of the query and benchmark labels.
                chunks = []
                for offset in range(0, max(1,len(encoded)), size):
                    text = header + '\n' + ENC.decode(encoded[offset:offset+size])
                    chunks.append({'parent':parent,'offset':offset,'text':text,'source':cs['ids'][idx]})
                children.append(chunks)
            cards.append(children)
        value = dict(graph=graph, names=name_idx, vectors=vectors, ci=ci, ei=ei, fi=fi, cards=cards,
                     passage_idxs=list(range(len(ei),len(names))))
        self.views[stage] = value
        write_json(OUT/f'graphs/p{self.persona}/stage{stage}.json', {
            'visible_passages':len(ci), 'entities':len(ei), 'facts':len(fi), 'edges':len(edges),
            'source_nodes':[[{'source': c['source'], 'offset':c['offset'], 'text':c['text']} for group in mode for c in group] for mode in cards],
            'source_scope':'visible original dialogue only; relation and source edges rebuilt from visible triples; no future synonym edges'})
        return value

    def candidates(self, stage, query, query_vector=None):
        path = OUT/f'candidates/p{self.persona}/{query["query_id"]}.json'
        if path.exists():
            return read(path)
        v = self.view(stage)
        qe = query_vector if query_vector is not None else norm(EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B', input=[query['question']], encoding_format='float').data[0].embedding)
        cs, es, fs = (self.stores[n] for n in ('chunk','entity','fact'))
        dense = cs['vectors'][v['ci']] @ qe
        seeds = np.zeros(len(v['names']))
        seeds[v['passage_idxs']] = (dense-dense.min())/max(float(np.ptp(dense)),1e-8)*0.05
        if v['fi']:
            fact_scores = fs['vectors'][v['fi']] @ qe
            for local in np.argsort(fact_scores)[-15:]:
                triple = ast.literal_eval(fs['texts'][v['fi'][local]])
                for entity in (triple[0],triple[2]):
                    ei = es['lookup'].get(entity)
                    if ei is not None and es['ids'][ei] in v['names']:
                        node = v['names'][es['ids'][ei]]
                        seeds[node] += max(0,float(fact_scores[local]))/max(1,v['graph'].degree(node))
        order, scores = run_igraph_qafd(v['graph'],v['names'],v['passage_idxs'],seeds,
            {name:v['vectors'][idx] for name,idx in v['names'].items()},qe,
            alpha=1.5,epsilon=.01,max_iterations=500,step_size=.2,weight_scheme='multiply',random_seed=42)
        result={'dense':np.argsort(dense)[::-1].tolist(),'graph':order.tolist(),'graph_scores':scores.tolist()}
        write_json(path,result)
        return result


def rerank_request(question, docs):
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    instruction='Given a question and a passage, determine if the passage contains information relevant to answering the question.'
    suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return {'model':'Qwen/Qwen3-Reranker-4B','query':f'{prefix}<Instruct>: {instruction}\n<Query>: {question}\n','documents':[f'<Document>: {x}{suffix}' for x in docs],'top_n':len(docs)}


def rerank(question, docs):
    s=requests.Session();s.trust_env=False
    r=s.post('http://127.0.0.1:19085/v1/rerank',json=rerank_request(question,docs),timeout=600)
    r.raise_for_status()
    return r.json()['results']


def lexical_leaves(patient, stage, action, question):
    cache=patient.__dict__.setdefault('lexical_indexes',{})
    key=(stage,action)
    if key not in cache:
        cards=[c for group in patient.view(stage)['cards'][action] for c in group]
        terms=[Counter(re.findall(r'[a-z0-9]+',c['text'].lower())) for c in cards]
        lengths=np.array([sum(t.values()) for t in terms],dtype=np.float64)
        postings={}
        for i,counts in enumerate(terms):
            for term,count in counts.items():postings.setdefault(term,[]).append((i,count))
        cache[key]=(cards,lengths,postings)
    cards,lengths,postings=cache[key]
    scores=np.zeros(len(cards));normalizer=1.5*(.25+.75*lengths/max(1,lengths.mean()))
    for term in set(re.findall(r'[a-z0-9]+',question.lower())):
        hits=postings.get(term,[])
        if not hits:continue
        idx=np.array([i for i,_ in hits]);tf=np.array([n for _,n in hits])
        idf=np.log1p((len(cards)-len(hits)+.5)/(len(hits)+.5))
        scores[idx]+=idf*tf*2.5/(tf+normalizer[idx])
    return [cards[i] for i in np.argsort(-scores,kind='stable')[:16] if scores[i]>0]


def episode(patient, stage, checkpoint, q, action, split, scorer, max_evidence_nodes=15, query_method="hybrid"):
    path=OUT/f'{split}/p{patient.persona}/{q["query_id"]}_a{action}.json'
    if path.exists():
        return read(path)
    started=time.time()
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} retrieval',flush=True)
    v=patient.view(stage)
    ranking=patient.candidates(stage,q)
    selected=[]; seen=set(); selected_tokens=0
    # Three dense candidates per graph candidate; writer granularity changes leaf capacity.
    merged=[]
    if query_method in ('date_anchored','lexical_leaf'):
        dates=set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',q['question']))
        for parent,idx in enumerate(v['ci']):
            header=patient.stores['chunk']['texts'][idx].split('\n',1)[0]
            if dates.intersection(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',header)):
                merged.append(parent)
        if query_method=='lexical_leaf' and not dates:
            for card in lexical_leaves(patient,stage,action,q['question']):
                if card['text'] in {c['text'] for c in selected}:continue
                cost=len(ENC.encode(card['text']))
                if selected_tokens+cost>32000:continue
                selected.append(card);selected_tokens+=cost
    for i in range(max(len(ranking['dense']),len(ranking['graph']))):
        merged.extend(ranking['dense'][3*i:3*i+3]); merged.extend(ranking['graph'][i:i+1])
    for parent in merged:
        if parent in seen:
            continue
        seen.add(parent)
        for card in v['cards'][action][parent]:
            if card['text'] not in {c['text'] for c in selected}:
                cost=len(ENC.encode(card['text']))
                if selected_tokens+cost>32000:continue
                selected.append(card);selected_tokens+=cost
            if len(selected)==128:break
        if len(selected)==128:break
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} rerank candidates={len(selected)} tokens={selected_tokens}',flush=True)
    rr=rerank(q['question'],[c['text'] for c in selected])
    raw_answers=read(RAW/f'persona_{patient.persona:02d}/session_{checkpoint:03d}_answers.json')
    baseline=next(a for a in raw_answers if a['query_id']==q['query_id'])
    budget=len(ENC.encode(baseline['formatted_context']))
    context=''; used=[]
    for rank,item in enumerate(rr[:max_evidence_nodes]):
        card=selected[item['index']]
        part=f'Memory {rank+1}:\n{card["text"]}\n\n'
        remaining=budget-len(ENC.encode(context))
        if remaining<=0:break
        part=ENC.decode(ENC.encode(part)[:remaining])
        context+=part;used.append(card)
    prompt=ANSWER_PROMPT.format(context=context,question=PROMPT.format_query(q['question'],q['query_type']))
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} answering',flush=True)
    response=CLIENT.chat.completions.create(model='Qwen3.8-27B',messages=[{'role':'user','content':prompt}],temperature=0,seed=0,max_tokens=16384,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    content=response.choices[0].message.content or ''
    matches=list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*',content,re.I))
    answer=content[matches[-1].end():].strip() if matches else content.strip()
    data={'persona':patient.persona,'query_id':q['query_id'],'checkpoint':checkpoint,'action':action,
          'question':q['question'],'answer':answer,'response':content,'context':context,'used_sources':used,
          'context_tokens':len(ENC.encode(context)),'raw_context_tokens':budget,'rerank_candidates':len(selected),
          'rerank_document_tokens':sum(len(ENC.encode(c['text'])) for c in selected),
          'candidate_sources':[{k:c[k] for k in ('source','offset','parent')} for c in selected],
          'rerank_results':rr,'max_evidence_nodes':max_evidence_nodes,'query_method':query_method,
          'judge_configuration':{'model':'gpt-6.1-sol','reasoning_effort':'high','max_tokens_floor':16000},
          'usage':response.usage.model_dump(),'finish_reason':response.choices[0].finish_reason,'seconds':time.time()-started}
    # Persist model output before invoking an independent judge.
    write_json(path.with_suffix('.answer.json'),data)
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} scoring',flush=True)
    score=scorer({'question':{'query_id':q['query_id'],'query_type':q['query_type'],'model_output':answer,'expected_answers':q['correct_answers'],'question':q['question'],'answers_data':q['answers'],'metadata':q['metadata']},'seconds':data['seconds'],'retrieved_count':len(used)})
    data['score']=score
    data['reward']=float(score['is_correct'])-.02*data['context_tokens']/max(1,budget)
    write_json(path,data)
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} correct={score["is_correct"]}',flush=True)
    return data


def train():
    rng=np.random.default_rng(17)
    logits=np.zeros(3);baseline=.5;trace=[]
    scorer=Scorer('gpt-6.1-sol','en')
    for persona in (3,4,5):
        patient=Patient(persona)
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in (20,40,60,80):continue
            seen=set();batch=[]
            for q in queries:
                if q['query_type'] in seen:continue
                seen.add(q['query_type']);batch.append(q)
            patient.view(stage)
            probs=softmax(logits)
            actions=[int(rng.choice(3,p=probs)) for _ in batch]
            with ThreadPoolExecutor(max_workers=6) as pool:
                rows=list(pool.map(lambda qa:episode(patient,stage,checkpoint,qa[0],qa[1],'train',scorer),zip(batch,actions)))
            gradient=np.zeros(3)
            for row in rows:
                one=np.eye(3)[row['action']]
                gradient+=(row['reward']-baseline)*(one-probs)/len(rows)
            logits+=.8*gradient
            baseline=.8*baseline+.2*np.mean([r['reward'] for r in rows])
            trace.append({'persona':persona,'checkpoint':checkpoint,'actions':actions,'rewards':[r['reward'] for r in rows],
                          'probabilities_before':probs.tolist(),'logits_after':logits.tolist(),'baseline_after':float(baseline)})
            write_json(OUT/'training.json',trace)
            write_json(OUT/'policy.json',{'algorithm':'REINFORCE with batch running baseline','seed':17,'logits':logits.tolist(),'probabilities':softmax(logits).tolist(),'action':int(np.argmax(logits)),'writer_sizes':SIZES,'training_personas':[3,4,5],'episodes':sum(len(t['actions']) for t in trace),'frozen':False})
            print('POLICY',softmax(logits),flush=True)
        del patient
    policy=read(OUT/'policy.json');policy['frozen']=True;write_json(OUT/'policy.json',policy)


def evaluate(personas,split,all_actions=False):
    policy=read(OUT/('policy.json' if all_actions else 'frozen_policy.json'))
    if not policy['frozen'] and not all_actions:raise ValueError('Writer policy must be frozen before evaluation')
    scorer=Scorer('gpt-6.1-sol','en')
    query_config={'max_evidence_nodes':15,'query_method':'hybrid'} if all_actions else read(OUT/'query_policy.json')
    packing=query_config['max_evidence_nodes']
    for persona in personas:
        patient=Patient(persona)
        with ThreadPoolExecutor(max_workers=8) as answers:
            pending=[]
            for stage,(checkpoint,queries) in enumerate(patient.groups):
                if split=='validation' and checkpoint not in (20,40,60,80):continue
                patient.view(stage)
                with ThreadPoolExecutor(max_workers=4) as retrieval:
                    list(retrieval.map(lambda q:patient.candidates(stage,q),queries))
                actions=range(3) if all_actions else sorted(set([policy['action'],2]))
                for q in queries:
                    for action in actions:
                        pending.append(answers.submit(episode,patient,stage,checkpoint,q,action,split,scorer,
                            max_evidence_nodes=packing,query_method=query_config['query_method']))
            for future in pending:future.result()
        del patient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['smoke','train','validate','evaluate'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    if args.command=='smoke':
        p=Patient(3);v=p.view(1);print({k:len(v[k]) for k in ('ci','ei','fi')})
        q=p.groups[1][1][0];print(episode(p,1,20,q,1,'smoke',Scorer('gpt-6.1-sol','en'))['score'])
    elif args.command=='train':train()
    elif args.command=='validate':evaluate([6],'validation',True)
    else:evaluate([1,2],'evaluation')

if __name__=='__main__':main()
