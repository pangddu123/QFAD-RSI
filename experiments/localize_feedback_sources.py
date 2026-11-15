"""Use training annotations to locate the source of the query's decisive fact."""
from concurrent.futures import ThreadPoolExecutor
import json
from qfad_fact_graph import base
from medmemory_scoring import JudgeClient

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114/feedback_training'
PROMPT='''Select one annotated source point that supplies the decisive information requested by this training question and its reference. For a state update, select the updated state, not the previous state quoted in the question. For multi-hop reasoning, select one indispensable patient-specific fact. Return JSON with point_index (the supplied integer index) and reason (one sentence). You are assigning training feedback to an original source session; you are not writing a graph or creating clinical facts.'''


def main():
    jobs=[]
    for persona in (3,4,5):
        queries=base.read(base.ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{persona}/eval/generated_queries.json')['queries']
        for checkpoint in (20,60):
            for kind in ('state_update','multi_hop_clinical_deduction'):
                q=next(q for q in queries if q['query_id'].startswith(f'session_{checkpoint}_') and q['query_type']==kind)
                points=[p for p in q['source_key_points'] if 0<int(p.get('session_id',0))<=checkpoint]
                jobs.append((persona,q,points))
    judge=JudgeClient('gpt-6.1-sol')
    def run(job):
        persona,q,points=job
        path=OUT/f'localization/p{persona}_{q["query_id"]}.json'
        if path.exists():return
        payload={'question':q['question'],'references':[a['content'] for a in q['answers'] if a.get('is_correct')],
                 'points':[{'index':i,**p} for i,p in enumerate(points)]}
        result=json.loads(judge.chat(messages=[{'role':'system','content':PROMPT},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=4096,
            reasoning_effort='high',response_format={'type':'json_object'}).content)
        index=result['point_index']
        assert type(index) is int and 0<=index<len(points)
        base.write_json(path,{'persona':persona,'query_id':q['query_id'],'session_id':int(points[index]['session_id']),
            'selected_annotation':points[index],'judge':'gpt-6.1-sol','reasoning_effort':'high',**result})
        print(path.stem,points[index]['session_id'],flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(run,jobs))


if __name__=='__main__':main()
