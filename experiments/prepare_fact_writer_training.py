"""Select training-source episodes using questions and visible records only."""
import numpy as np
from qfad_fact_graph import base

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114/training'


def main():
    inputs=[];plans=[]
    for persona in (3,4,5):
        patient=base.Patient(persona)
        chunks=patient.stores['chunk']
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in (20,60):
                continue
            visible={text for s,text in patient.history if s<=stage}
            indices=[i for i,text in enumerate(chunks['texts']) if text in visible and text.startswith('[Session')]
            for kind in ('state_update','multi_hop_clinical_deduction'):
                q=next(q for q in queries if q['query_type']==kind)
                qe=base.norm(base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=[q['question']],encoding_format='float').data[0].embedding)
                order=np.argsort(-(chunks['vectors'][indices]@qe),kind='stable')
                candidate=[];cost=0
                for local in order:
                    i=indices[int(local)];size=len(base.ENC.encode(chunks['texts'][i]))
                    if cost+size>32000:
                        continue
                    candidate.append(i);cost+=size
                    if len(candidate)==32:
                        break
                ranked=base.rerank(q['question'],[chunks['texts'][i] for i in candidate])
                index=candidate[ranked[0]['index']]
                text=chunks['texts'][index];source_id=chunks['ids'][index]
                episode_id=f'p{persona}_{q["query_id"]}'
                inputs.append({'episode_id':episode_id,'sources':[{'source_id':source_id,'text':text,
                    'old_facts':patient.openie[text].get('extracted_triples',[])}]})
                plans.append({'episode_id':episode_id,'persona':persona,'stage':stage,'checkpoint':checkpoint,
                              'query_ids':[q['query_id']],'source_selection':{'primary_patient_records_only':True,
                              'candidate_sources':[chunks['ids'][i] for i in candidate],'ranked':ranked,
                              'document_tokens':cost,'uses_reference_answers':False}})
                base.write_json(OUT/'writer_inputs.json',inputs)
                base.write_json(OUT/'training_plan.json',plans)
                print(episode_id,source_id,flush=True)
    assert len(inputs)==12
    print('Prepared 12 source-writing episodes',flush=True)


if __name__=='__main__':main()
