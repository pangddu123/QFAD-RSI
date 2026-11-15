"""Fine source anchors and query-reward context writing on the QFAD graph."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import time
import numpy as np
import query_writer_rl as base

OUT=base.ROOT/'results/qfad_anchor_writer_rl_20261113'
ENC=base.ENC

class AnchorPatient:
    def __init__(self,persona):
        self.persona=persona
        self.coarse=base.Patient(persona)
        self.groups=self.coarse.groups
        cache=OUT/f'indexes/p{persona}';cache.mkdir(parents=True,exist_ok=True)
        if (cache/'metadata.json').exists():
            self.raw=base.read(cache/'metadata.json')
            self.vectors=np.load(cache/'vectors.npy',mmap_mode='r')
        else:
            index=base.read(base.RAW/f'persona_{persona:02d}/rag_index.json')
            self.raw=next(iter(index['conversations'].values()))
            self.vectors=base.norm(np.asarray(self.raw.pop('embeddings'),dtype=np.float32))
            self.raw.pop('source_fingerprint',None)
            np.save(cache/'vectors.npy',self.vectors)
            base.write_json(cache/'metadata.json',self.raw)
        self.turn_context=self.raw['turn_context']
        self.turn_starts=[s[0] for s in self.turn_context['turn_spans']]
        self.turn_ends=[s[1] for s in self.turn_context['turn_spans']]
        from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset
        dataset=MedMemoryBenchDataset(base.ROOT/'MedMemoryBench/data/MedMemoryBench_EN_mixed',{
            'evaluation_mode':'independent','persona_ids':[persona],'evaluation_interval':10,'inject_noise':True})
        dataset.load();self.lines=[];self.headers=[];self.turn_stages=[]
        for stage,unit in enumerate(dataset.get_evaluation_units()):
            for session in unit.sessions_to_inject:
                date='unknown date' if session.is_noise or not session.timestamp else datetime.fromisoformat(session.timestamp).date().isoformat()
                header=base.harness.session_header(session) if session.is_noise else f'[Session time: {session.timestamp}]' if session.timestamp else '[Session]'
                for turn,message in enumerate(session.metadata.get('messages',[])):
                    if message.get('role') not in ('user','assistant'):continue
                    content=message.get('content','')
                    if session.is_noise and turn==0:content=base.harness.session_header(session)+'\n'+content
                    role='Patient' if message['role']=='user' else 'Doctor'
                    self.lines.append(f'[{date}] {role}: {content}')
                    self.headers.append(header);self.turn_stages.append(stage)
        assert self.lines==self.turn_context['turn_lines'],'Cached index differs from the original visible dialogue order'
        self.views={}
        self.query_vectors=None
        self.entity_phrases=[' '.join(re.findall(r'\w+',s.lower())) for s in self.coarse.stores['entity']['texts']]
        link_path=cache/'entity_mentions.npz'
        if link_path.exists():
            links=np.load(link_path);self.anchor_edges=links['anchors'];self.entity_edges=links['entities']
        else:
            normalized=[' '+' '.join(re.findall(r'\w+',c['content'].lower()))+' ' for c in self.raw['chunks']]
            postings=defaultdict(list)
            for i,s in enumerate(normalized):
                for token in set(s.split()):postings[token].append(i)
            aa=[];ee=[]
            for entity,phrase in enumerate(self.entity_phrases):
                terms=phrase.split()
                if not terms:continue
                candidates=min((postings.get(t,[]) for t in terms),key=len)
                for anchor in candidates:
                    if ' '+phrase+' ' in normalized[anchor]:aa.append(anchor);ee.append(entity)
            self.anchor_edges=np.asarray(aa,dtype=np.int32);self.entity_edges=np.asarray(ee,dtype=np.int32)
            np.savez(link_path,anchors=self.anchor_edges,entities=self.entity_edges)

    def view(self,stage):
        if stage in self.views:return self.views[stage]
        turn_count=sum(s<=stage for s in self.turn_stages)
        text=self.turn_context['header']+'\n\n'+'\n'.join(self.lines[:turn_count])
        tokens=ENC.encode(text,disallowed_special=())
        anchors=[];boundary_vector=None
        for i,start in enumerate(range(0,len(tokens),224)):
            end=min(start+256,len(tokens));content=ENC.decode(tokens[start:end])
            cached=self.raw['chunks'][i]
            if cached['content']!=content:
                assert end==len(tokens),'Only the final visible anchor may differ from the full-history cache'
                path=OUT/f'boundary_vectors/p{self.persona}/stage{stage}.json'
                if path.exists():
                    saved=base.read(path);assert saved['content']==content
                    boundary_vector=np.asarray(saved['embedding'],dtype=np.float32)
                else:
                    response=base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=[content],encoding_format='float')
                    boundary_vector=base.norm(response.data[0].embedding)
                    base.write_json(path,{'content':content,'embedding':boundary_vector.tolist()})
            anchors.append({'index':i,'start_token':start,'end_token':end,'content':content})
            if end==len(tokens):break
        view={'stage':stage,'turn_count':turn_count,'tokens':tokens,'anchors':anchors,
              'boundary_vector':boundary_vector,'coarse':self.coarse.view(stage)}
        edge_mask=self.anchor_edges<len(anchors)-1
        aa=self.anchor_edges[edge_mask];ee=self.entity_edges[edge_mask]
        last=' '+' '.join(re.findall(r'\w+',anchors[-1]['content'].lower()))+' '
        last_entities=[e for e,s in enumerate(self.entity_phrases) if s and ' '+s+' ' in last]
        view['anchor_edges']=np.concatenate([aa,np.full(len(last_entities),len(anchors)-1,dtype=np.int32)])
        view['entity_edges']=np.concatenate([ee,np.asarray(last_entities,dtype=np.int32)])
        view['entity_leaf_degree']=np.bincount(view['entity_edges'],minlength=len(self.entity_phrases))
        coarse=view['coarse'];ne=len(coarse['ei']);parents=[];entities=[]
        for left,right in coarse['graph'].get_edgelist():
            if right<ne<=left:left,right=right,left
            if left<ne<=right:parents.append(right-ne);entities.append(coarse['ei'][left])
        view['coarse_parents']=np.asarray(parents,dtype=np.int32)
        view['coarse_entities']=np.asarray(entities,dtype=np.int32)
        view['coarse_parent_degree']=np.bincount(parents,minlength=len(coarse['ci']))
        self.views[stage]=view
        base.write_json(OUT/f'scopes/p{self.persona}/stage{stage}.json',{
            'visible_turns':turn_count,'visible_tokens':len(tokens),'anchors':len(anchors),
            'original_index':str(base.RAW/f'persona_{self.persona:02d}/rag_index.json'),
            'cached_full_anchors':len(anchors)-int(boundary_vector is not None),
            'boundary_reembedded':boundary_vector is not None})
        return view

    def source_headers(self,view,start,end):
        first=bisect_right(self.turn_ends,start)
        last=min(view['turn_count'],bisect_left(self.turn_starts,end))
        return list(dict.fromkeys(self.headers[first:last]))

    def payload(self,view,anchor_index,size):
        anchor=view['anchors'][anchor_index]
        center=(anchor['start_token']+anchor['end_token'])//2
        start=max(0,center-size//2);end=min(len(view['tokens']),start+size)
        start=max(0,end-size)
        headers=self.source_headers(view,start,end)
        body=ENC.decode(view['tokens'][start:end])
        return {'anchor_index':anchor_index,'start_token':start,'end_token':end,
                'source_headers':headers,'text':'\n'.join(headers)+'\n'+body}

    def query_vector(self,q):
        if self.query_vectors is None:
            path=OUT/f'query_vectors/p{self.persona}.json'
            if path.exists():self.query_vectors=base.read(path)
            else:
                self.query_vectors={}
                log=base.RAW/f'persona_{self.persona:02d}/logs/rag_requests.jsonl'
                with log.open() as f:
                    for line in f:
                        if '"call_type": "embedding"' not in line[:500] or '"question_id": null' in line[:500]:continue
                        row=json.loads(line)
                        if row.get('error') or not row.get('response'):continue
                        self.query_vectors[row['question_id']]={'question':row['request']['input'][0],
                            'embedding':row['response']['data'][0]['embedding']}
                base.write_json(path,self.query_vectors)
        if q['query_id'] in self.query_vectors:
            saved=self.query_vectors[q['query_id']];assert saved['question']==q['question']
            return base.norm(saved['embedding'])
        response=base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=[q['question']],encoding_format='float')
        return base.norm(response.data[0].embedding)

    def retrieve(self,stage,q):
        path=OUT/f'retrieval/p{self.persona}/{q["query_id"]}.json'
        if path.exists():return base.read(path)
        view=self.view(stage);n=len(view['anchors']);qe=self.query_vector(q)
        dense=self.vectors[:n]@qe
        if view['boundary_vector'] is not None:dense[-1]=view['boundary_vector']@qe
        coarse=self.coarse.candidates(stage,q,query_vector=qe)
        doc_mass=np.zeros(len(view['coarse']['ci']))
        doc_mass[coarse['graph']]=coarse['graph_scores']
        parents=view['coarse_parents'];entities=view['coarse_entities']
        entity_mass=np.bincount(entities,weights=doc_mass[parents]/np.maximum(1,view['coarse_parent_degree'][parents]),minlength=len(self.entity_phrases))
        projected=np.bincount(view['anchor_edges'],weights=entity_mass[view['entity_edges']]/np.maximum(1,view['entity_leaf_degree'][view['entity_edges']]),minlength=n)
        dense_order=np.argsort(-dense,kind='stable').tolist()
        graph_order=[int(i) for i in np.argsort(-projected,kind='stable') if projected[i]>0]
        dates=set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',q['question']));merged=[]
        if dates:
            for i in dense_order:
                anchor=view['anchors'][i];headers=self.source_headers(view,anchor['start_token'],anchor['end_token'])
                if dates.intersection(re.findall(r'\b\d{4}-\d{2}-\d{2}\b',' '.join(headers))):merged.append(i)
        for i in range(max(len(dense_order),len(graph_order))):
            merged.extend(dense_order[3*i:3*i+3]);merged.extend(graph_order[i:i+1])
        selected=[];seen=set();cost=0
        for i in merged:
            if i in seen:continue
            seen.add(i);tokens=len(ENC.encode(view['anchors'][i]['content']))
            if cost+tokens>32000:continue
            selected.append(i);cost+=tokens
            if len(selected)==128:break
        ranked=base.rerank(q['question'],[view['anchors'][i]['content'] for i in selected])
        result={'candidate_indices':selected,'document_tokens':cost,'ranked':ranked,
                'visible_anchors':n,'graph_positive_anchors':int(np.count_nonzero(projected)),
                'entity_mention_edges':len(view['anchor_edges'])}
        base.write_json(path,result);return result


def episode(patient,stage,checkpoint,q,action,split,scorer,client=None):
    path=OUT/f'{split}/p{patient.persona}/{q["query_id"]}_a{action}.json'
    if path.exists():return base.read(path)
    started=time.time();view=patient.view(stage);retrieval=patient.retrieve(stage,q)
    saved=path.with_suffix('.answer.json')
    if saved.exists():data=base.read(saved)
    else:
        baseline=next(r for r in base.read(base.RAW/f'persona_{patient.persona:02d}/session_{checkpoint:03d}_answers.json') if r['query_id']==q['query_id'])
        budget=len(ENC.encode(baseline['formatted_context']));context='';used=[]
        for item in retrieval['ranked']:
            card=patient.payload(view,retrieval['candidate_indices'][item['index']],base.SIZES[action])
            if any(max(0,min(card['end_token'],c['end_token'])-max(card['start_token'],c['start_token']))>.5*min(card['end_token']-card['start_token'],c['end_token']-c['start_token']) for c in used):continue
            part=f'Memory {len(used)+1}:\n{card["text"]}\n\n'
            remaining=budget-len(ENC.encode(context))
            if remaining<=0:break
            context+=ENC.decode(ENC.encode(part)[:remaining]);used.append(card)
            if len(used)==15:break
        prompt=base.ANSWER_PROMPT.format(context=context,question=base.PROMPT.format_query(q['question'],q['query_type']))
        print(f'{split} p{patient.persona} {q["query_id"]} a{action} answering',flush=True)
        reader=client or base.CLIENT
        response=reader.chat.completions.create(model='Qwen3.8-27B',messages=[{'role':'user','content':prompt}],temperature=0,seed=0,max_tokens=16384,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        content=response.choices[0].message.content or ''
        matches=list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*',content,re.I))
        answer=content[matches[-1].end():].strip() if matches else content.strip()
        data={'persona':patient.persona,'query_id':q['query_id'],'checkpoint':checkpoint,'stage':stage,'action':action,
              'question':q['question'],'answer':answer,'response':content,'context':context,'used_sources':used,
              'context_tokens':len(ENC.encode(context)),'raw_context_tokens':budget,
              'rerank_candidates':len(retrieval['candidate_indices']),'rerank_document_tokens':retrieval['document_tokens'],
              'visible_anchors':len(view['anchors']),'visible_tokens':len(view['tokens']),
              'usage':response.usage.model_dump(),'finish_reason':response.choices[0].finish_reason,'seconds':time.time()-started,
              'answer_endpoint':str(reader.base_url),'max_answer_tokens':16384,
              'judge_configuration':{'model':'gpt-6.1-sol','reasoning_effort':'high','max_tokens_floor':16000}}
        base.write_json(saved,data)
    score=scorer({'question':{'query_id':q['query_id'],'query_type':q['query_type'],'model_output':data['answer'],
        'expected_answers':q['correct_answers'],'question':q['question'],'answers_data':q['answers'],'metadata':q['metadata']},
        'seconds':data['seconds'],'retrieved_count':len(data['used_sources'])})
    data['score']=score;data['reward']=float(score['is_correct'])-.02*data['context_tokens']/max(1,data['raw_context_tokens'])
    base.write_json(path,data)
    print(f'{split} p{patient.persona} {q["query_id"]} a{action} correct={score["is_correct"]}',flush=True)
    return data

if __name__=='__main__':
    patient=AnchorPatient(6);view=patient.view(1)
    for index in (0,len(view['anchors'])//2,len(view['anchors'])-1):
        for size in (256,512,1024):
            payload=patient.payload(view,index,size)
            assert 0<=payload['start_token']<payload['end_token']<=len(view['tokens'])
            anchor=view['anchors'][index]
            assert payload['start_token']<=anchor['start_token'] and payload['end_token']>=anchor['end_token']
    result=patient.retrieve(1,patient.groups[1][1][0])
    print(json.dumps({'persona':6,'checkpoint':20,'visible_turns':view['turn_count'],
                      'anchors':len(view['anchors']),'visible_tokens':len(view['tokens']),
                      'source_text_matches_original':True,'payload_scope_checks':9,
                      'rerank_candidates':len(result['candidate_indices']),'rerank_document_tokens':result['document_tokens'],
                      'graph_positive_anchors':result['graph_positive_anchors']},indent=2))
