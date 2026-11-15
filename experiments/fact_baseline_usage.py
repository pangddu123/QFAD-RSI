"""Read the baseline's actual answer and reranking request budgets."""
import json
from qfad_fact_graph import base

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'


def load_baseline_usage():
    cache=OUT/'raw_usage.json'
    if cache.exists():return base.read(cache)
    prior_usage=base.read(base.OUT/'raw_usage.json')
    prior_rerank=base.read(base.OUT/'raw_rerank_usage.json')
    result={p:{qid:{'usage':usage,'rerank':prior_rerank[p][qid]} for qid,usage in rows.items()}
            for p,rows in prior_usage.items()}
    result['6']={}
    directory=base.RAW/'persona_06'
    answers={r['query_id']:r['answer'] for p in directory.glob('session_*_answers.json') for r in base.read(p)}
    with (directory/'logs/rag_requests.jsonl').open() as stream:
        for line in stream:
            if not any(f'"call_type": "{kind}"' in line[:600] for kind in ('answer','rerank')):continue
            row=json.loads(line)
            qid=row.get('question_id')
            if qid not in answers or row.get('error') or not row.get('response'):continue
            entry=result['6'].setdefault(qid,{})
            if row['call_type']=='answer':
                response=row['response']['choices'][0]['message']['content'] or ''
                if answers[qid].strip() not in response:continue
                entry['usage']=row['response']['usage']
            else:
                docs=row['request']['documents']
                bodies=[doc.removeprefix('<Document>: ').rsplit('<|im_end|>',1)[0] for doc in docs]
                entry['rerank']={'candidates':len(docs),'document_tokens':sum(len(base.ENC.encode(s)) for s in bodies)}
    assert all('usage' in row and 'rerank' in row for row in result['6'].values())
    base.write_json(cache,result)
    return result


if __name__=='__main__':
    result=load_baseline_usage()
    for persona,rows in result.items():print(persona,len(rows))
