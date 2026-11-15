"""Evaluate a frozen memory-size-conditioned QFAD writer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from query_writer_rl import OUT, Patient, Scorer, episode, read, write_json

def choose(policy,visible_passages):
    x=np.array([1.,(np.log1p(visible_passages)-policy['feature_mean'])/policy['feature_scale']])
    return int((np.array(policy['weights'])@x).argmax())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('persona',type=int);args=parser.parse_args()
    policy=read(OUT/'frozen_contextual_policy.json');assert policy['frozen']
    patient=Patient(args.persona);scorer=Scorer('gpt-6.1-sol','en');manifest=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[]
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            view=patient.view(stage);action=choose(policy,len(view['ci']))
            for q in queries:
                path=OUT/f'evaluation/p{args.persona}/{q["query_id"]}_a{action}.json'
                manifest.append({'persona':args.persona,'query_id':q['query_id'],'checkpoint':checkpoint,
                                 'visible_passages':len(view['ci']),'action':action,'path':str(path.relative_to(OUT))})
                if not path.exists():
                    patient.candidates(stage,q)
                    pending.append(pool.submit(episode,patient,stage,checkpoint,q,action,'evaluation',scorer,
                        max_evidence_nodes=15,query_method=policy['query_method']))
        for future in pending:future.result()
    write_json(OUT/f'contextual_evaluation_p{args.persona}.json',manifest)

if __name__=='__main__':main()
