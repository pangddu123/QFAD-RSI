"""Score saved model outputs with the experiment's fixed judge configuration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from query_writer_rl import OUT, Patient, Scorer, read, write_json

CONFIG={'model':'gpt-6.1-sol','reasoning_effort':'high','max_tokens_floor':16000}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=['training','validation','evaluation'])
    ap.add_argument('--personas',nargs='+',type=int);args=ap.parse_args()
    splits={'training':['train','train_date'],'validation':['validation','budget_probe','date_probe'],'evaluation':['evaluation']}[args.group]
    jobs=[]
    for split in splits:
        for answer_path in (OUT/split).glob('p*/*.answer.json'):
            path=answer_path.with_name(answer_path.name.replace('.answer.json','.json'))
            row=read(path if path.exists() else answer_path)
            if args.personas and row['persona'] not in args.personas:continue
            if path.exists() and row.get('judge_configuration')==CONFIG:continue
            jobs.append((path,row))
    if not jobs:
        print('All saved answers have the configured scores',flush=True)
        return
    import qafd_medmemorybench_mixed as harness
    from query_writer_rl import ROOT
    harness.DATA=ROOT/'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    harness.rawrag.DATA=harness.DATA
    harness.LANGUAGE=harness.rawrag.LANGUAGE='en'
    personas=sorted({r['persona'] for _,r in jobs})
    _,groups=harness.load_units(personas)
    queries={(p,q['query_id']):q for p,g in groups.items() for _,qq in g for q in qq}
    scorer=Scorer('gpt-6.1-sol','en')
    def score(job):
        path,row=job;q=queries[row['persona'],row['query_id']]
        result=scorer({'question':{'query_id':q['query_id'],'query_type':q['query_type'],'model_output':row['answer'],'expected_answers':q['correct_answers'],'question':q['question'],'answers_data':q['answers'],'metadata':q['metadata']},'seconds':row['seconds'],'retrieved_count':len(row['used_sources'])})
        if path.parts[-3]=='train' and 'reward' in row:row.setdefault('behavior_reward',row['reward'])
        row['score']=result
        row['reward']=float(result['is_correct'])-.02*row['context_tokens']/max(1,row['raw_context_tokens'])
        row['judge_configuration']=CONFIG
        write_json(path,row)
        print(path.relative_to(OUT),result['is_correct'],flush=True)
    with ThreadPoolExecutor(max_workers=15) as pool:list(pool.map(score,jobs))

if __name__=='__main__':main()
