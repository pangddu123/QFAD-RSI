"""Prepare a training-only source episode for the factual writer loop."""
from qfad_fact_graph import base

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'


def main():
    patient=base.Patient(3)
    stage=1
    checkpoint,queries=patient.groups[stage]
    selected=[q for q in queries if q['query_id'] in ('session_20_eem_1','session_20_sua_1')]
    visible={text for s,text in patient.history if s<=stage}
    chunks=patient.stores['chunk']
    indices=[i for i,text in enumerate(chunks['texts']) if text in visible]
    sources=[];seen=set()
    for q in selected:
        qe=base.norm(base.EMBED.embeddings.create(model='Qwen/Qwen3-Embedding-4B',input=[q['question']],encoding_format='float').data[0].embedding)
        scores=chunks['vectors'][indices]@qe
        i=indices[int(scores.argmax())]
        cid=chunks['ids'][i]
        if cid in seen:
            continue
        seen.add(cid)
        text=chunks['texts'][i]
        sources.append({'source_id':cid,'text':text,'old_facts':patient.openie[text].get('extracted_triples',[])})
    identifier='p3_checkpoint20_medication_and_sleep'
    base.write_json(OUT/'writer_inputs.json',[{'episode_id':identifier,'sources':sources}])
    base.write_json(OUT/'training_plan.json',[{'episode_id':identifier,'persona':3,'stage':stage,
                                            'checkpoint':checkpoint,'query_ids':[q['query_id'] for q in selected]}])
    print({'episode_id':identifier,'sources':len(sources),'queries':len(selected)},flush=True)


if __name__=='__main__':main()
