"""Fit a writer actor with a per-query RAW-RAG reward control variate."""
from collections import Counter
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/qfad_query_writer_rl_20261113'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'


def read(p):return json.loads(p.read_text())

def softmax(x):
    p=np.exp(x-x.max());return p/p.sum()


def fit(require_frozen=True,query_method="hybrid"):
    behavior=read(OUT/'policy.json')
    if require_frozen and not behavior['frozen']:
        raise ValueError('Reward collection must finish before fitting the evaluation actor')
    theta=np.zeros(3);updates=[]
    for step in read(OUT/'training.json'):
        persona,checkpoint=step['persona'],step['checkpoint']
        rows=[read(p) for p in (OUT/f'train/p{persona}').glob('*.json') if not p.name.endswith('.answer.json')]
        rows=[r for r in rows if r['checkpoint']==checkpoint]
        if Counter(r['action'] for r in rows)!=Counter(step['actions']):
            raise ValueError('Recorded behavior actions do not match completed reward rows')
        raw={r['query_id']:r for r in read(RAW/f'persona_{persona:02d}/session_{checkpoint:03d}_scores.json')}
        pi=softmax(theta);mu=np.array(step['probabilities_before']);gradient=np.zeros(3)
        episodes=[]
        for original in rows:
            r=original
            if query_method=='date_anchored' and re.search(r'\b\d{4}-\d{2}-\d{2}\b',r['question']):
                path=OUT/f'train_date/p{persona}/{r["query_id"]}_a{r["action"]}.json'
                if not path.exists():raise ValueError(f'Missing refreshed query reward: {path}')
                r=read(path)
            if require_frozen and r.get('judge_configuration',{}).get('reasoning_effort')!='high':
                raise ValueError('Final actor requires high-reasoning reward scores')
            action=r['action']
            # The control variate is independent of this writer's sampled action.
            baseline=float(raw[r['query_id']]['is_correct'])-.02
            advantage=r['reward']-baseline
            importance=float(pi[action]/mu[action])
            gradient+=importance*advantage*(np.eye(3)[action]-pi)/len(rows)
            episodes.append({'query_id':r['query_id'],'action':action,'reward':r['reward'],
                             'raw_reward':baseline,'advantage':advantage,'importance':importance})
        theta+=.8*gradient
        updates.append({'persona':persona,'checkpoint':checkpoint,'probabilities_before':pi.tolist(),
                        'behavior_probabilities':mu.tolist(),'episodes':episodes,'logits_after':theta.tolist()})
    result={'algorithm':'Importance-weighted REINFORCE with per-query RAW-RAG reward control variate',
            'logits':theta.tolist(),'probabilities':softmax(theta).tolist(),'action':int(theta.argmax()),
            'writer_sizes':[256,512,1024],'training_personas':[3,4,5],'episodes':sum(len(x['episodes']) for x in updates),
            'frozen':bool(behavior['frozen']),'query_method':query_method,'reward_judge':{'model':'gpt-6.1-sol','reasoning_effort':'high','max_tokens_floor':16000},'learning_rate':.8,'epochs':1,'uses_validation_rewards':False}
    if require_frozen:
        (OUT/'paired_training.json').write_text(json.dumps(updates,indent=2)+'\n')
        (OUT/'frozen_policy.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    config=read(OUT/'query_policy.json')
    print(json.dumps(fit(query_method=config['query_method']),indent=2))
