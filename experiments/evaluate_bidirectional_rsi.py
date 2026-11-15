"""Fresh full-cohort answers from a frozen graph and Reader policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from qfad_fact_graph import FactGraph,base
from rsi_reader_policy import ReaderPolicy
from rsi_reader_group import candidate_features
from semantic_query_writer_rl import RUBRIC
from medmemory_scoring import JudgeClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--graph-state',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fallback',choices=['graph','raw'],default='graph')
    parser.add_argument('--answer-cache',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    plan=base.read(base.ROOT/'results/qfad_fact_writer_rl_20261114/evaluation/query_plan.json')
    assert len(plan)==len({(r['persona'],r['query_id']) for r in plan})==197
    policy=ReaderPolicy();policy.load(args.policy)
    writes=base.read(args.graph_state) if args.graph_state else {}
    reader_config={'model':'Qwen3.8-27B','temperature':0,'seed':0,'max_tokens':16384,'enable_thinking':False}
    config={'reader':reader_config,'policy_weights':policy.weights.detach().tolist(),
            'policy_paths':policy.paths,'graph_writes':writes,'semantic_rubric':RUBRIC,
            'queries':197,'fresh_answers':not bool(args.answer_cache)}
    if args.fallback!='graph' or args.answer_cache:
        config.update(fallback=args.fallback,answer_cache=[str(p) for p in args.answer_cache])
    if (args.output/'configuration.json').exists():
        assert base.read(args.output/'configuration.json')==config,'Use a new output directory for a changed state'
    base.write_json(args.output/'configuration.json',config)
    policy.save(args.output/'policy_snapshot.pt')
    scorer=base.Scorer('gpt-6.1-sol','en');judge=JudgeClient('gpt-6.1-sol')
    raw_budgets=base.read(base.ROOT/'results/qfad_bidirectional_rsi_p12_20261114/raw_retrieval_budgets.json') if args.fallback=='raw' else {}
    cached={}
    for directory in args.answer_cache:
        prior=base.read(directory/'configuration.json')
        assert prior['reader']==reader_config and prior['semantic_rubric']==RUBRIC
        for file in directory.glob('p*/*.json'):
            if file.name.endswith('.answer.json'):continue
            row=base.read(file)
            cached[(row['persona'],row['query_id'],row['context'])]=(file,row)

    def answer(q,record):
        path=args.output/f'p{record["persona"]}/{q["query_id"]}.json'
        if path.exists():return base.read(path)
        shared=cached.get((record['persona'],q['query_id'],record['context']))
        if shared is not None:
            origin,prior=shared
            for key in ('answer','response','usage','finish_reason','reader_model','score','semantic_score','raw_semantic_correct'):
                record[key]=prior[key]
            record['answer_origin']=str(origin)
            base.write_json(path,record)
            return record
        saved=path.with_suffix('.answer.json')
        if saved.exists():
            record=base.read(saved)
        else:
            prompt=base.ANSWER_PROMPT.format(context=record['context'],question=base.PROMPT.format_query(q['question'],q['query_type']))
            response=base.CLIENT.chat.completions.create(model='Qwen3.8-27B',messages=[{'role':'user','content':prompt}],
                temperature=0,seed=0,max_tokens=16384,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            text=response.choices[0].message.content or ''
            matches=list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*',text,re.I))
            record.update({'answer':text[matches[-1].end():].strip() if matches else text.strip(),
                           'response':text,'usage':response.usage.model_dump(),
                           'finish_reason':response.choices[0].finish_reason,'reader_model':response.model,
                           'answer_origin':'fresh_model_call'})
            base.write_json(saved,record)
        record['score']=scorer({'question':{'query_id':q['query_id'],'query_type':q['query_type'],
            'model_output':record['answer'],'expected_answers':q['correct_answers'],'question':q['question'],
            'answers_data':q['answers'],'metadata':q['metadata']},'seconds':0.,
            'retrieved_count':record.get('retrieved_count',len(record['retrieved_facts']))})
        raw=base.read(base.ROOT/f'results/qfad_query_writer_rl_20261113/semantic/evaluation/p{record["persona"]}/{q["query_id"]}_raw.json')
        assert raw['rubric']==RUBRIC
        if record['answer']==raw['response']:
            semantic={'correct':raw['correct'],'reason':raw['reason']}
        else:
            payload={'question':q['question'],'question_type':q['query_type'],
                     'reference_answers':[a['content'] for a in q['answers'] if a.get('is_correct')],
                     'response':record['answer']}
            semantic=json.loads(judge.chat(messages=[{'role':'system','content':RUBRIC},
                {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=4096,
                reasoning_effort='high',response_format={'type':'json_object'}).content)
        assert type(semantic['correct']) is bool
        record['semantic_score']=semantic;record['raw_semantic_correct']=raw['correct']
        base.write_json(path,record)
        print(record['persona'],q['query_id'],record['score']['is_correct'],semantic['correct'],flush=True)
        return record

    futures=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for persona in (1,2):
            patient=base.Patient(persona)
            queries={q['query_id']:q for _,qq in patient.groups for q in qq}
            raw={r['query_id']:r for p in (base.RAW/f'persona_{persona:02d}').glob('session_*_answers.json') for r in base.read(p)}
            embedding_cache={r['question']:base.norm(r['query_embedding']) for r in plan if r['persona']==persona}
            for stage in sorted({r['stage'] for r in plan if r['persona']==persona}):
                items=[r for r in plan if r['persona']==persona and r['stage']==stage]
                pending=[r for r in items if not (args.output/f'p{persona}/{r["query_id"]}.json').exists()]
                if not pending:continue
                visible={text for s,text in patient.history if s<=stage}
                visible_ids={source for source,text in zip(patient.stores['chunk']['ids'],patient.stores['chunk']['texts']) if text in visible}
                active=[r for r in writes.get(str(persona),[]) if r['source_id'] in visible_ids]
                graph=FactGraph(patient,stage,active,embedding_cache)
                for item in pending:
                    q=queries[item['query_id']];episode=f'p{persona}_{q["query_id"]}'
                    cap=len(base.ENC.encode(raw[q['query_id']]['formatted_context']))
                    path=policy.paths.get(episode)
                    indices=graph.resolve_path(path['documents']) if path else None
                    documents=[graph.fact_document(i) for i in indices] if indices is not None else None
                    render=lambda docs:'\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(docs))
                    if documents is not None and len(documents)<=15 and len(base.ENC.encode(render(documents)))<=cap:
                        retrieval={'method':'learned_evidence_path','source_rerank_candidates':0,'source_rerank_document_tokens':0,
                                   'candidate_fact_indices':indices,'rerank_document_tokens':0,'path_facts_present_in_visible_graph':True}
                        total_candidates=0;total_tokens=0
                        context=render(documents)
                        retrieved_count=len(documents)
                    elif args.fallback=='raw':
                        indices=[]
                        context=raw[q['query_id']]['formatted_context']
                        retrieved_count=len(raw[q['query_id']]['retrieved'])
                        budget=raw_budgets[episode]
                        total_candidates=budget['candidates'];total_tokens=budget['document_tokens']
                        retrieval={'method':'raw_rag','cached_original_retrieval':True,
                                   'retrieval_cost_charged_as_original_query':True,
                                   'rerank_document_tokens':total_tokens,
                                   'route_selection':'No retained successful graph path valid in the current graph'}
                    else:
                        retrieval=graph.retrieve(q['question'],candidate_limit=128-item['source_rerank_candidates'],
                            document_budget=32000-item['source_rerank_document_tokens'],anchor_source=item['source_id'])
                        matrix,costs=candidate_features(graph,retrieval,q['question'],q['query_type'])
                        chosen=policy.select(matrix,costs,cap)
                        indices=[retrieval['candidate_fact_indices'][i] for i in chosen.indices]
                        documents=[retrieval['candidate_documents'][i] for i in chosen.indices]
                        retrieval['method']='parameterized_graph_selection'
                        total_candidates=item['source_rerank_candidates']+len(retrieval['candidate_fact_indices'])
                        total_tokens=item['source_rerank_document_tokens']+retrieval['rerank_document_tokens']
                        context=render(documents)
                        retrieved_count=len(documents)
                    rerank_cap=raw_budgets[episode]['document_tokens'] if args.fallback=='raw' else 32000
                    assert len(base.ENC.encode(context))<=cap and total_candidates<=128 and total_tokens<=rerank_cap
                    record={'persona':persona,'query_id':q['query_id'],'stage':stage,'checkpoint':item['checkpoint'],
                        'context':context,'context_tokens':len(base.ENC.encode(context)),'raw_context_tokens':cap,
                        'retrieved_facts':[graph.facts[i] for i in indices],'retrieval':retrieval,
                        'retrieved_count':retrieved_count,
                        'total_rerank_candidates':total_candidates,'total_rerank_document_tokens':total_tokens,
                        'active_written_sources':[r['source_id'] for r in active],'reader_configuration':reader_config}
                    futures.append(pool.submit(answer,q,record))
        for future in futures:future.result()
    rows=[base.read(args.output/f'p{r["persona"]}/{r["query_id"]}.json') for r in plan]
    summary={'n':197,'primary_correct':sum(r['score']['is_correct'] for r in rows),
        'semantic_correct':sum(r['semantic_score']['correct'] for r in rows),
        'raw_semantic_correct':sum(r['raw_semantic_correct'] for r in rows),
        'reader_prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
        'reader_completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),
        'learned_path_queries':sum(r['retrieval']['method']=='learned_evidence_path' for r in rows),
        'raw_rag_queries':sum(r['retrieval']['method']=='raw_rag' for r in rows),
        'rerank_candidates':sum(r['total_rerank_candidates'] for r in rows),
        'rerank_document_tokens':sum(r['total_rerank_document_tokens'] for r in rows),
        'shared_identical_prompt_answers':sum(r.get('answer_origin','fresh_model_call')!='fresh_model_call' for r in rows),
        'wins_vs_raw':[f'p{r["persona"]}_{r["query_id"]}' for r in rows if r['semantic_score']['correct'] and not r['raw_semantic_correct']],
        'losses_vs_raw':[f'p{r["persona"]}_{r["query_id"]}' for r in rows if not r['semantic_score']['correct'] and r['raw_semantic_correct']],
        'fresh_frozen_state_evaluation':not bool(args.answer_cache),'frozen_state_evaluation':True}
    base.write_json(args.output/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
