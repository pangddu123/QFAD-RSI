"""Assign training-query credit to annotated source sessions, exporting raw text only."""
from qfad_fact_graph import base

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114/feedback_training'


def main():
    inputs=[];plans=[]
    for persona in (3,4,5):
        patient=base.Patient(persona)
        directory=base.ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{persona}/eval'
        sessions={int(s['session_id']):s for s in base.read(directory/'generated_dialogues.json')['sessions']}
        chunks=patient.stores['chunk']
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in (20,60):continue
            visible={text for s,text in patient.history if s<=stage}
            for kind in ('state_update','multi_hop_clinical_deduction'):
                q=next(q for q in queries if q['query_type']==kind)
                localized=base.read(OUT/f'localization/p{persona}_{q["query_id"]}.json')
                session_id=localized['session_id']
                assert 0<session_id<=checkpoint
                date=sessions[session_id]['event_info']['date']
                indices=[i for i,text in enumerate(chunks['texts']) if text in visible and
                         text.partition('\n')[0]==f'[Session time: {date}]']
                assert indices,(persona,session_id,date)
                feedback=q['question']+'\nTraining reference target: '+'; '.join(q['correct_answers'])
                ranked=base.rerank(feedback,[chunks['texts'][i] for i in indices])
                index=indices[ranked[0]['index']]
                text=chunks['texts'][index];source_id=chunks['ids'][index]
                episode_id=f'p{persona}_{q["query_id"]}'
                inputs.append({'episode_id':episode_id,'sources':[{'source_id':source_id,'text':text,
                    'old_facts':patient.openie[text].get('extracted_triples',[])}]})
                plans.append({'episode_id':episode_id,'persona':persona,'stage':stage,'checkpoint':checkpoint,
                    'query_ids':[q['query_id']],'source_selection':{'training_annotation_used':True,
                    'session_id':session_id,'record_date':date,'source_id':source_id,
                    'writer_receives_query_or_reference':False}})
                base.write_json(OUT/'writer_inputs.json',inputs)
                base.write_json(OUT/'training_plan.json',plans)
                print(episode_id,session_id,date,flush=True)
    assert len(inputs)==12
    print('Prepared 12 source-grounded feedback episodes',flush=True)


if __name__=='__main__':main()
