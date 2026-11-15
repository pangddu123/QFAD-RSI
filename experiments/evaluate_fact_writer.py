"""Evaluate cumulative fact writes with frozen policies and a fixed Reader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import re
from qfad_fact_graph import FactGraph, base
from fact_writer_protocol import parse_output

ROOT_OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', default='')
    parser.add_argument('--split',choices=['validation','evaluation'],required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--facts',type=int,default=15)
    parser.add_argument('--query-method',choices=['global','focused'],default='global')
    args=parser.parse_args()
    root=ROOT_OUT/args.run/args.split
    plan=base.read(root/'query_plan.json')
    requests={r['episode_id']:r for r in base.read(root/'writer_inputs.json')}
    predictions={key:base.read(root/f'writer_outputs/{args.condition}/{key}.json') for key in requests} if args.condition!='original' else {}
    label=args.condition if args.query_method=='global' else args.condition+'_focused'
    destination=root/f'{label}_k{args.facts}'
    scorer=base.Scorer('gpt-6.1-sol','en')
    control={}
    if args.condition!='initial':
        control_label='initial' if args.query_method=='global' else 'initial_focused'
        control_directory=root/f'{control_label}_k{args.facts}'
        for item in plan:
            key=(item['persona'],item['query_id'])
            control[key]=base.read(control_directory/f'p{key[0]}/{key[1]}.json')

    def answer(q,record):
        path=destination/f'p{record["persona"]}/{q["query_id"]}.json'
        if path.exists():return base.read(path)
        baseline=control.get((record['persona'],q['query_id']))
        saved=path.with_suffix('.answer.json')
        if saved.exists():
            record=base.read(saved)
        elif baseline is not None and baseline['context']==record['context']:
            assert baseline['reader_configuration']==record['reader_configuration']
            record.update({k:baseline[k] for k in ('answer','response','usage','finish_reason','reader_model')})
            record['paired_answer_reused_from']='initial'
            base.write_json(saved,record)
        else:
            prompt=base.ANSWER_PROMPT.format(context=record['context'],question=base.PROMPT.format_query(q['question'],q['query_type']))
            response=base.CLIENT.chat.completions.create(model='Qwen3.8-27B',messages=[{'role':'user','content':prompt}],
                temperature=0,seed=0,max_tokens=16384,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            text=response.choices[0].message.content or ''
            matches=list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*',text,re.I))
            record.update({'answer':text[matches[-1].end():].strip() if matches else text.strip(),
                           'response':text,'usage':response.usage.model_dump(),'finish_reason':response.choices[0].finish_reason,
                           'reader_model':response.model})
            base.write_json(saved,record)
        if baseline is not None and baseline['answer']==record['answer']:
            record['score']=baseline['score']
            record['paired_score_reused_from']='initial'
        else:
            record['score']=scorer({'question':{'query_id':q['query_id'],'query_type':q['query_type'],
                'model_output':record['answer'],'expected_answers':q['correct_answers'],'question':q['question'],
                'answers_data':q['answers'],'metadata':q['metadata']},'seconds':0.,'retrieved_count':len(record['retrieved_facts'])})
        base.write_json(path,record)
        print(args.condition,record['persona'],q['query_id'],record['score']['is_correct'],flush=True)
        return record

    futures=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for persona in sorted({r['persona'] for r in plan}):
            patient=base.Patient(persona)
            queries={q['query_id']:q for _,qq in patient.groups for q in qq}
            active={};attempted=set();embedding_cache={}
            for item in [r for r in plan if r['persona']==persona]:
                q=queries[item['query_id']]
                key=item['writer_request_id'];prediction=predictions.get(key)
                new_write=args.condition!='original' and key not in attempted
                accepted=None;reason=None
                if new_write:
                    try:
                        if not prediction['finished']:raise ValueError('Writer output truncated')
                        written=parse_output(prediction['text'],requests[key]['sources'])
                        for row in written:active[row['source_id']]=row
                        accepted=True
                    except (ValueError,KeyError,TypeError) as error:
                        accepted=False;reason=str(error)
                    attempted.add(key)
                    write_path=destination/f'writes/{key}.json'
                    write_record={'request_id':key,'accepted':accepted,
                        'reason':reason,'first_query_id':q['query_id'],'stage':item['stage'],
                        'adapter_path':prediction['adapter_path'],'prompt_tokens':prediction['prompt_tokens'],
                        'completion_tokens':prediction['completion_tokens'],
                        'facts':active.get(item['source_id'],{}).get('facts',[])}
                    if not write_path.exists():base.write_json(write_path,write_record)
                path=destination/f'p{persona}/{q["query_id"]}.json'
                if path.exists():continue
                embedding_cache[q['question']]=base.norm(item['query_embedding'])
                graph=FactGraph(patient,item['stage'],list(active.values()),embedding_cache)
                if new_write and accepted:
                    write_record['graph']=graph.manifest([item['source_id']])
                    base.write_json(write_path,write_record)
                retrieval=graph.retrieve(q['question'],candidate_limit=128-item['source_rerank_candidates'],
                    document_budget=32000-item['source_rerank_document_tokens'],result_limit=args.facts,
                    anchor_source=item['source_id'] if args.query_method=='focused' else None)
                raw=next(r for r in base.read(base.RAW/f'persona_{persona:02d}/session_{item["checkpoint"]:03d}_answers.json') if r['query_id']==q['query_id'])
                cap=len(base.ENC.encode(raw['formatted_context']))
                context='\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(retrieval['documents']))
                context=base.ENC.decode(base.ENC.encode(context)[:cap])
                record={'persona':persona,'query_id':q['query_id'],'stage':item['stage'],'checkpoint':item['checkpoint'],
                    'condition':args.condition,'query_method':args.query_method,'context':context,'context_tokens':len(base.ENC.encode(context)),
                    'raw_context_tokens':cap,'retrieved_facts':[graph.facts[i] for i in retrieval['fact_indices']],
                    'retrieval':retrieval,'source_rerank_candidates':item['source_rerank_candidates'],
                    'source_rerank_document_tokens':item['source_rerank_document_tokens'],
                    'total_rerank_candidates':item['source_rerank_candidates']+len(retrieval['candidate_fact_indices']),
                    'total_rerank_document_tokens':item['source_rerank_document_tokens']+retrieval['rerank_document_tokens'],
                    'writer_request_id':key,'new_write':new_write,'write_accepted':accepted,'write_rejection':reason,
                    'writer_prompt_tokens':prediction['prompt_tokens'] if new_write else 0,
                    'writer_completion_tokens':prediction['completion_tokens'] if new_write else 0,
                    'active_written_sources':list(active),'graph_fact_count':len(graph.facts),'graph_edge_count':graph.graph.ecount()}
                record['reader_configuration']={'model':'Qwen3.8-27B','temperature':0,'seed':0,
                    'max_tokens':16384,'enable_thinking':False}
                record['judge_configuration']={'model':'gpt-6.1-sol','reasoning_effort':'high','max_tokens_floor':16000}
                assert record['total_rerank_candidates']<=128 and record['total_rerank_document_tokens']<=32000
                futures.append(pool.submit(answer,q,record))
        for future in futures:future.result()
    print('Finished',args.split,args.condition,flush=True)


if __name__=='__main__':main()
