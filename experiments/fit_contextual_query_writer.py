"""Fit a memory-size-conditioned writer from logged query rewards."""
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/qfad_query_writer_rl_20261113'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'
def read(p):return json.loads(p.read_text())
def softmax(x):
    p=np.exp(x-x.max());return p/p.sum()
def count(persona,checkpoint):
    return read(OUT/f'graphs/p{persona}/stage{checkpoint//10-1}.json')['visible_passages']

def fit():
    steps=read(OUT/'training.json')
    sizes=np.log1p([count(s['persona'],s['checkpoint']) for s in steps])
    mean=float(sizes.mean());scale=float(sizes.std())
    weights=np.zeros((3,2));updates=[]
    for step in steps:
        persona,checkpoint=step['persona'],step['checkpoint']
        x=np.array([1.,(np.log1p(count(persona,checkpoint))-mean)/scale])
        pi=softmax(weights@x);mu=np.array(step['probabilities_before']);gradient=np.zeros_like(weights)
        rows=[read(p) for p in (OUT/f'train/p{persona}').glob(f'session_{checkpoint}_*.json') if not p.name.endswith('.answer.json')]
        baseline={r['query_id']:r for r in read(RAW/f'persona_{persona:02d}/session_{checkpoint:03d}_scores.json')}
        episodes=[]
        for original in rows:
            row=original
            if re.search(r'\b\d{4}-\d{2}-\d{2}\b',row['question']):
                row=read(OUT/f'train_date/p{persona}/{row["query_id"]}_a{row["action"]}.json')
            assert row['judge_configuration']['reasoning_effort']=='high'
            action=row['action'];advantage=row['reward']-(float(baseline[row['query_id']]['is_correct'])-.02)
            importance=pi[action]/mu[action]
            gradient+=importance*advantage*np.outer(np.eye(3)[action]-pi,x)/len(rows)
            episodes.append({'query_id':row['query_id'],'action':action,'advantage':advantage,'importance':float(importance)})
        weights+=.8*gradient
        updates.append({'persona':persona,'checkpoint':checkpoint,'features':x.tolist(),'probabilities':pi.tolist(),'episodes':episodes,'weights_after':weights.tolist()})
    validation=[]
    for checkpoint in (20,40,60,80):
        x=np.array([1.,(np.log1p(count(6,checkpoint))-mean)/scale]);p=softmax(weights@x)
        validation.append({'checkpoint':checkpoint,'visible_passages':count(6,checkpoint),'probabilities':p.tolist(),'action':int(p.argmax())})
    result={'algorithm':'Contextual importance-weighted REINFORCE','features':['intercept','standardized log1p visible source passages'],
            'feature_mean':mean,'feature_scale':scale,'weights':weights.tolist(),'writer_sizes':[256,512,1024],
            'training_personas':[3,4,5],'episodes':sum(len(s['episodes']) for s in updates),'learning_rate':.8,'epochs':1,
            'query_method':'date_anchored','uses_validation_rewards':False,'validation_actions':validation,'updates':updates}
    return result

def main():
    result=fit()
    (OUT/'contextual_policy_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='updates'},indent=2))

if __name__=='__main__':main()
