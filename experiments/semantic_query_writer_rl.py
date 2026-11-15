"""Blinded semantic scoring for all methods under one fixed rubric."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'MedMemoryBench')]
from medmemory_scoring import JudgeClient, write_json

OUT=ROOT/'results/qfad_query_writer_rl_20261113'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'
EXTERNAL_CACHE_ROOTS=[]
RUBRIC='''Evaluate one medical-memory benchmark answer against its reference. The answer is research evaluation data, not patient care. Use the question and reference as the grading target. Mark correct only when the response supplies the decisive requested facts and contains no material contradiction. Accept synonyms, equivalent clinical terminology, date formats, and paraphrases; do not require verbatim string containment. For a multiple-choice question require the correct option(s). For a multi-hop question require the key causal connections in the reference, not merely a list of related terms. Do not require extra details from the reference that the question did not request. Do not reward verbosity. Return only JSON with keys correct (boolean) and reason (one brief sentence).'''


def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['train','validation','validation_fixed','evaluation','budget_probe','date_probe','lexical_probe','contextual_probe']);args=ap.parse_args()
    personas={'train':[3,4,5],'validation':[6],'validation_fixed':[6],'evaluation':[1,2],'budget_probe':[6],'date_probe':[6],'lexical_probe':[6],'contextual_probe':[6]}[args.split]
    jobs=[]
    for persona in personas:
        queries=read(ROOT/f'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_{persona}/eval/generated_queries.json')['queries']
        qmap={q['query_id']:q for q in queries}
        rows=[(p.stem,read(p)) for p in (OUT/args.split/f'p{persona}').glob('*.json') if not p.name.endswith('.answer.json')]
        qids={r['query_id'] for _,r in rows}
        for f in (RAW/f'persona_{persona:02d}').glob('session_*_answers.json'):
            rows.extend((r['query_id']+'_raw',r) for r in read(f) if r['query_id'] in qids)
        for name,row in rows:
            target=OUT/'semantic'/args.split/f'p{persona}/{name}.json'
            q=qmap[row['query_id']]
            payload={'question':q['question'],'question_type':q['query_type'],
                     'reference_answers':[a['content'] for a in q['answers'] if a.get('is_correct')],
                     'response':row['answer']}
            jobs.append((target,row['query_id'],payload))
    groups={}
    for job in jobs:
        key=(job[0].parent.name,job[1],json.dumps(job[2],sort_keys=True,ensure_ascii=False))
        groups.setdefault(key,[]).append(job)
    known={}
    for root in [*EXTERNAL_CACHE_ROOTS,OUT]:
        for p in (root/'semantic').glob('*/p*/*.json'):
            row=read(p)
            if row.get('rubric')==RUBRIC:
                known[(p.parent.name,row['query_id'],row['response'])]=row
    client=JudgeClient('gpt-6.1-sol')
    def score(group):
        target,qid,payload=group[0]
        cached=[read(p) for p,_,_ in group if p.exists()]
        prior=known.get((target.parent.name,qid,payload['response']))
        if cached or prior:
            result=cached[0] if cached else prior
            for p,_,_ in group:
                if not p.exists():write_json(p,result)
            return
        text=client.chat(messages=[{'role':'system','content':RUBRIC},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=4096,reasoning_effort='high',response_format={'type':'json_object'}).content
        result=json.loads(text)
        if type(result.get('correct')) is not bool:raise ValueError('Judge returned no boolean verdict')
        row={'query_id':qid,'response':payload['response'],'judge':'gpt-6.1-sol','reasoning_effort':'high','rubric':RUBRIC,**result}
        for p,_,_ in group:write_json(p,row)
        print([p.name for p,_,_ in group],result['correct'],flush=True)
    with ThreadPoolExecutor(max_workers=15) as pool:list(pool.map(score,groups.values()))

if __name__=='__main__':main()
