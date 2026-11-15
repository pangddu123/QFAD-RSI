"""Collect fixed-behavior query rewards for the QFAD anchor writer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from qfad_anchor_writer import AnchorPatient, OUT, ENC, base, episode

def index_training():
    rng=np.random.default_rng(17);plan=[]
    for persona in (3,4,5):
        patient=AnchorPatient(persona)
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in (20,40,60,80):continue
            count=sum(s<=stage for s in patient.turn_stages)
            text=patient.turn_context['header']+'\n\n'+'\n'.join(patient.lines[:count])
            length=len(ENC.encode(text,disallowed_special=()))
            anchors=(max(0,length-256)+223)//224+1
            seen=set()
            for q in queries:
                if q['query_type'] in seen:continue
                seen.add(q['query_type'])
                if (persona,q['query_id'])==(3,'session_80_tla_1'):continue
                plan.append({'persona':persona,'stage':stage,'checkpoint':checkpoint,
                             'query_id':q['query_id'],'query_type':q['query_type'],
                             'action':int(rng.choice(3)),'behavior_probabilities':[1/3]*3,
                             'visible_anchors':anchors})
        print(f'Indexed training patient {persona}',flush=True)
        del patient
    assert len(plan)==70
    base.write_json(OUT/'behavior_plan.json',{'seed':17,'sampling':'uniform categorical',
        'training_personas':[3,4,5],'excluded':{'persona':3,'query_id':'session_80_tla_1','reason':'Question record date conflicts with reference target time'},'episodes':plan})
    print(f'Prepared {len(plan)} behavior episodes',flush=True)

def collect():
    plan=base.read(OUT/'behavior_plan.json')['episodes'];patients={p:AnchorPatient(p) for p in (3,4,5)}
    queries={(p,q['query_id']):q for p,patient in patients.items() for _,qq in patient.groups for q in qq}
    for item in plan:
        view=patients[item['persona']].view(item['stage'])
        assert len(view['anchors'])==item['visible_anchors']
    with ThreadPoolExecutor(max_workers=8) as retrieval:
        list(retrieval.map(lambda item:patients[item['persona']].retrieve(item['stage'],queries[item['persona'],item['query_id']]),plan))
    print('Prepared all training retrievals',flush=True)
    client=base.OpenAI(api_key='local',base_url='http://127.0.0.1:19106/v1',timeout=900)
    scorer=base.Scorer('gpt-6.1-sol','en')
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures=[]
        for item in plan:
            futures.append(pool.submit(episode,patients[item['persona']],item['stage'],item['checkpoint'],
                queries[item['persona'],item['query_id']],item['action'],'train',scorer,client=client))
        for future in futures:future.result()
    print(f'Collected {len(plan)} query rewards',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['index','collect']);args=parser.parse_args()
    if args.command=='index':index_training()
    else:collect()
