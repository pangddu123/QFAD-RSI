"""Paired correctness and token accounting for the QFAD writer policy."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/qfad_query_writer_rl_20261113'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'


def read(p):return json.loads(p.read_text())

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def raw_usage():
    path=OUT/'raw_usage.json'
    if path.exists():return read(path)
    data={}
    for persona in (1,2):
        rows={}
        with (RAW/f'persona_{persona:02d}/logs/rag_requests.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r['call_type']=='answer' and r.get('response') and not r.get('error'):
                    rows[r['question_id']]=r['response']['usage']
        data[str(persona)]=rows
    save(path,data)
    return data


def summarize():
    policy_path=OUT/('frozen_policy.json' if (OUT/'frozen_policy.json').exists() else 'policy.json')
    policy=read(policy_path) if policy_path.exists() else None
    summary={'policy':policy,'training':{},'validation':{},'evaluation':{}}
    for split in ('train','train_date','validation','budget_probe','date_probe','lexical_probe','contextual_probe','evaluation'):
        rows=[read(p) for p in (OUT/split).glob('p*/*.json') if not p.name.endswith('.answer.json')]
        for action in range(3):
            group=[r for r in rows if r['action']==action]
            if not group:continue
            target=summary.setdefault('training' if split=='train' else split,{})
            target[str(action)]={'n':len(group),'correct':sum(r['score']['is_correct'] for r in group),
                'context_tokens':sum(r['context_tokens'] for r in group),
                'raw_context_tokens':sum(r['raw_context_tokens'] for r in group),
                'answer_prompt_tokens':sum(r['usage']['prompt_tokens'] for r in group),
                'answer_completion_tokens':sum(r['usage']['completion_tokens'] for r in group),
                'max_rerank_document_tokens':max(r['rerank_document_tokens'] for r in group),
                'max_candidates':max(r['rerank_candidates'] for r in group),
                'context_budget_violations':sum(r['context_tokens']>r['raw_context_tokens'] for r in group),
                'truncated_answers':sum(r['finish_reason']=='length' for r in group),
                'high_judge_scores':sum(r.get('judge_configuration',{}).get('reasoning_effort')=='high' for r in group)}
    paired=[]
    if policy and policy['frozen']:
        for persona in (1,2):
            raw={r['query_id']:r for f in (RAW/f'persona_{persona:02d}').glob('session_*_scores.json') for r in read(f)}
            for qid,r in raw.items():
                file=OUT/f'evaluation/p{persona}/{qid}_a{policy["action"]}.json'
                if not file.exists():continue
                x=read(file)
                paired.append({'persona':persona,'query_id':qid,'query_type':r['query_type'],
                    'raw_correct':int(r['is_correct']),'rl_correct':int(x['score']['is_correct']),
                    'raw_answer':r['model_output'],'rl_answer':x['answer'],
                    'context_tokens':x['context_tokens'],'raw_context_tokens':x['raw_context_tokens']})
    if paired:
        with (OUT/'paired.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
        delta=np.array([r['rl_correct']-r['raw_correct'] for r in paired])
        rng=np.random.default_rng(42)
        boot=np.mean(rng.choice(delta,size=(10000,len(delta))),axis=1)
        summary['paired']={'n':len(paired),'complete':len(paired)==197,'raw_correct':sum(r['raw_correct'] for r in paired),
            'rl_correct':sum(r['rl_correct'] for r in paired),'delta_pp':float(delta.mean()*100),
            'query_bootstrap_95ci_pp':(np.quantile(boot,[.025,.975])*100).tolist(),
            'wins':int(sum(delta==1)),'losses':int(sum(delta==-1))}
    if (OUT/'raw_usage.json').exists():
        raw=read(OUT/'raw_usage.json')
        summary['raw_answer_usage']={p:{'questions':len(rows),'prompt_tokens':sum(r['prompt_tokens'] for r in rows.values()),'completion_tokens':sum(r['completion_tokens'] for r in rows.values())} for p,rows in raw.items()}
    for split in ('validation','evaluation','budget_probe','date_probe','lexical_probe','contextual_probe'):
        sem=[read(p) | {'name':p.stem,'persona':p.parent.name} for p in (OUT/'semantic'/split).glob('p*/*.json')]
        if sem:
            summary['semantic_'+split]={}
            for name in ('raw','a0','a1','a2'):
                group=[r for r in sem if r['name'].endswith('_'+name)]
                if group:summary['semantic_'+split][name]={'n':len(group),'correct':sum(r['correct'] for r in group)}
    if policy and policy['query_method']=='date_anchored':
        validation={p.stem:read(p) for p in (OUT/'validation/p6').glob('*_a0.json')}
        validation.update({p.stem:read(p) for p in (OUT/'date_probe/p6').glob('*_a0.json')})
        semantic={p.stem:read(p) for p in (OUT/'semantic/validation/p6').glob('*_a0.json')}
        semantic.update({p.stem:read(p) for p in (OUT/'semantic/date_probe/p6').glob('*_a0.json')})
        summary['selected_query_validation']={'n':len(validation),
            'primary_correct':sum(r['score']['is_correct'] for r in validation.values()),
            'semantic_n':len(semantic),'semantic_correct':sum(r['correct'] for r in semantic.values())}
    if paired:
        summary['by_patient']={}
        for persona in (1,2):
            group=[r for r in paired if r['persona']==persona]
            summary['by_patient'][str(persona)]={'n':len(group),'raw_correct':sum(r['raw_correct'] for r in group),
                'rl_correct':sum(r['rl_correct'] for r in group)}
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':summarize()
