"""Question-only source selection with a shared total reranking budget."""
import argparse
import numpy as np
from qfad_fact_graph import base

ROOT_OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['validation','evaluation'],required=True)
    args=parser.parse_args()
    output=ROOT_OUT/args.split;requests={};plan=[]
    for persona in ([6] if args.split=='validation' else [1,2]):
        patient=base.Patient(persona);chunks=patient.stores['chunk']
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if args.split=='validation' and checkpoint not in (20,40,60,80):continue
            visible={text for s,text in patient.history if s<=stage}
            indices=[i for i,text in enumerate(chunks['texts']) if text in visible and text.startswith('[Session')]
            for q in queries:
                qe=base.norm(base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=[q['question']],encoding_format='float').data[0].embedding)
                order=np.argsort(-(chunks['vectors'][indices]@qe),kind='stable')
                candidate=[];tokens=0
                for local in order:
                    index=indices[int(local)];cost=len(base.ENC.encode(chunks['texts'][index]))
                    if tokens+cost>16000:continue
                    candidate.append(index);tokens+=cost
                    if len(candidate)==16:break
                ranked=base.rerank(q['question'],[chunks['texts'][i] for i in candidate])
                index=candidate[ranked[0]['index']]
                source_id=chunks['ids'][index];request_id=f'p{persona}_source{index}'
                text=chunks['texts'][index]
                requests[request_id]={'episode_id':request_id,'sources':[{'source_id':source_id,'text':text,
                    'old_facts':patient.openie[text].get('extracted_triples',[])}]}
                plan.append({'persona':persona,'stage':stage,'checkpoint':checkpoint,'query_id':q['query_id'],
                             'question':q['question'],'writer_request_id':request_id,'source_id':source_id,
                             'source_rerank_candidates':len(candidate),'source_rerank_document_tokens':tokens,
                             'source_ranking':ranked,'query_embedding':qe.tolist(),
                             'uses_reference_or_source_annotations':False})
                base.write_json(output/'writer_inputs.json',list(requests.values()))
                base.write_json(output/'query_plan.json',plan)
                print(args.split,persona,q['query_id'],request_id,flush=True)
    expected=41 if args.split=='validation' else 197
    assert len(plan)==expected
    print(f'Prepared {len(plan)} queries and {len(requests)} unique source writes',flush=True)


if __name__=='__main__':main()
