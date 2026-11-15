"""Fit writer policies from fixed-behavior query-reward episodes."""
from collections import defaultdict
import json
import numpy as np
from qfad_anchor_writer import OUT, base

def fit(contextual):
    plan=base.read(OUT/'behavior_plan.json')['episodes'];batches=defaultdict(list)
    for item in plan:batches[item['persona'],item['checkpoint']].append(item)
    sizes=np.log1p([items[0]['visible_anchors'] for items in batches.values()])
    mean=float(sizes.mean());scale=float(sizes.std());weights=np.zeros((3,2 if contextual else 1));updates=[]
    for (persona,checkpoint),items in batches.items():
        x=np.array([1.,(np.log1p(items[0]['visible_anchors'])-mean)/scale] if contextual else [1.])
        pi=base.softmax(weights@x);gradient=np.zeros_like(weights);episodes=[]
        baseline={r['query_id']:r for r in base.read(base.RAW/f'persona_{persona:02d}/session_{checkpoint:03d}_scores.json')}
        for item in items:
            action=item['action'];row=base.read(OUT/f'train/p{persona}/{item["query_id"]}_a{action}.json')
            assert row['judge_configuration']['reasoning_effort']=='high'
            assert row['action']==action and row['visible_anchors']==item['visible_anchors']
            raw_reward=float(baseline[row['query_id']]['is_correct'])-.02
            advantage=row['reward']-raw_reward;importance=pi[action]/item['behavior_probabilities'][action]
            gradient+=importance*advantage*np.outer(np.eye(3)[action]-pi,x)/len(items)
            episodes.append({'query_id':row['query_id'],'action':action,'reward':row['reward'],
                             'raw_reward':raw_reward,'advantage':advantage,'importance':float(importance)})
        weights+=.8*gradient
        updates.append({'persona':persona,'checkpoint':checkpoint,'features':x.tolist(),
                        'probabilities_before':pi.tolist(),'episodes':episodes,'weights_after':weights.tolist()})
    return {'algorithm':'Importance-weighted REINFORCE','contextual':contextual,'weights':weights.tolist(),
            'feature_mean':mean,'feature_scale':scale,'writer_sizes':base.SIZES,
            'training_personas':[3,4,5],'episodes':len(plan),'learning_rate':.8,'epochs':1,'updates':updates,
            'reward_judge':{'model':'gpt-6.1-sol','reasoning_effort':'high'},'uses_validation_rewards':False}

def choose(policy,anchors):
    x=np.array([1.,(np.log1p(anchors)-policy['feature_mean'])/policy['feature_scale']] if policy['contextual'] else [1.])
    return int((np.asarray(policy['weights'])@x).argmax())

if __name__=='__main__':
    for contextual in (False,True):
        policy=fit(contextual);name='contextual' if contextual else 'global'
        base.write_json(OUT/f'{name}_policy_candidate.json',policy)
        actions={checkpoint:choose(policy,base.read(OUT/f'scopes/p6/stage{checkpoint//10-1}.json')['anchors']) for checkpoint in (20,40,60,80)}
        print(json.dumps({'policy':name,'validation_actions':actions,'weights':policy['weights']},indent=2))
