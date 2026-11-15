"""Evaluate the trained contextual writer on patient 6 only."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
from query_writer_rl import OUT, RAW, Patient, Scorer, episode, read, write_json

def main():
    policy=read(OUT/'contextual_policy_candidate.json')
    actions={r['checkpoint']:r['action'] for r in policy['validation_actions']}
    patient=Patient(6);scorer=Scorer('gpt-6.1-sol','en');manifest=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[]
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in actions:continue
            action=actions[checkpoint]
            for q in queries:
                dated=bool(re.search(r'\b\d{4}-\d{2}-\d{2}\b',q['question']))
                split='date_probe' if dated and action==0 else 'contextual_probe' if dated else 'validation'
                path=OUT/f'{split}/p6/{q["query_id"]}_a{action}.json'
                manifest.append({'query_id':q['query_id'],'checkpoint':checkpoint,'action':action,'path':str(path.relative_to(OUT))})
                if not path.exists():
                    patient.view(stage);patient.candidates(stage,q)
                    pending.append(pool.submit(episode,patient,stage,checkpoint,q,action,split,scorer,
                        max_evidence_nodes=15,query_method='date_anchored'))
        for future in pending:future.result()
    scores=[read(OUT/r['path']) for r in manifest]
    baseline={r['query_id']:r for checkpoint in actions for r in read(RAW/f'persona_06/session_{checkpoint:03d}_scores.json')}
    result={'n':len(scores),'primary_correct':sum(r['score']['is_correct'] for r in scores),
            'raw_correct':sum(baseline[r['query_id']]['is_correct'] for r in scores),'rows':manifest}
    write_json(OUT/'contextual_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
