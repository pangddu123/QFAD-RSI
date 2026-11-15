"""Verify evaluated source provenance, time scope, frozen policy and budgets."""
import argparse
import json
from pathlib import Path
import numpy as np
import tiktoken

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/qfad_query_writer_rl_20261113'
SOURCE=ROOT/'results/qafd_medmemorybench_en_mixed_gpt4omini_20261111'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'

def read(path):return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contextual',action='store_true');args=parser.parse_args()
    from query_writer_rl import harness
    from fit_query_writer_policy import fit
    harness.DATA=ROOT/'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    harness.rawrag.DATA=harness.DATA
    harness.LANGUAGE=harness.rawrag.LANGUAGE='en'
    histories=harness.load_passages([1,2])
    _,groups=harness.load_units([1,2])
    enc=tiktoken.get_encoding('cl100k_base')
    policy=read(OUT/('frozen_contextual_policy.json' if args.contextual else 'frozen_policy.json'))
    if args.contextual:
        from fit_contextual_query_writer import fit as contextual_fit
        from evaluate_contextual_query_writer import choose
        replay=contextual_fit()
        assert np.allclose(policy['weights'],replay['weights'],rtol=0,atol=1e-12)
        assert policy['feature_mean']==replay['feature_mean'] and policy['feature_scale']==replay['feature_scale']
    else:
        replay=fit(require_frozen=False,query_method=policy['query_method'])
        assert np.allclose(policy['logits'],replay['logits'],rtol=0,atol=1e-12)
    assert policy['training_personas']==[3,4,5]
    raw_usage=read(OUT/'raw_usage.json')
    raw_rerank=read(OUT/'raw_rerank_usage.json')
    report={'policy_replay_matches':True,'patients':{},'violations':[]}
    for persona in (1,2):
        docs=read(SOURCE/f'persona_{persona:02d}/qafd/openie_results_ner_gpt-4o-mini.json')['docs']
        passages={d['idx']:d['passage'] for d in docs}
        stages={checkpoint:stage for stage,(checkpoint,_) in enumerate(groups[persona])}
        expected={q['query_id'] for _,queries in groups[persona] for q in queries}
        if args.contextual:
            manifest=read(OUT/f'contextual_evaluation_p{persona}.json')
            paths=[OUT/r['path'] for r in manifest]
            graph_counts={checkpoint:read(OUT/f'graphs/p{persona}/stage{stage}.json')['visible_passages'] for checkpoint,stage in stages.items()}
            selected_actions={r['query_id']:r['action'] for r in manifest}
            for r in manifest:
                assert r['visible_passages']==graph_counts[r['checkpoint']]
                assert r['action']==choose(policy,graph_counts[r['checkpoint']])
        else:
            paths=list((OUT/f'evaluation/p{persona}').glob(f'*_a{policy["action"]}.json'))
        rows=[read(p) for p in paths]
        seen={r['query_id'] for r in rows}
        assert seen==expected, (persona,'missing or extra questions',expected^seen)
        counts={'questions':len(rows),'source_nodes_checked':0,'native_prompt_over_raw':0,
                'native_prompt_tokens':0,'native_completion_tokens':0,
                'raw_prompt_tokens':0,'raw_completion_tokens':0,
                'rerank_document_tokens':0,'raw_rerank_document_tokens':0}
        for path,row in zip(paths,rows):
            if args.contextual:assert row['action']==selected_actions[row['query_id']]
            saved=read(path.with_suffix('.answer.json'))
            assert row['answer']==saved['answer'] and row['response']==saved['response']
            assert row['judge_configuration']['reasoning_effort']=='high'
            assert row['query_method']==policy['query_method']
            stage=stages[row['checkpoint']]
            visible={text for s,text in histories[persona] if s<=stage}
            for card in row['used_sources']:
                parent=passages[card['source']]
                assert parent in visible,(persona,row['query_id'],'future source')
                header,_,body=parent.partition('\n')
                offset=card['offset'];size=policy['writer_sizes'][row['action']]
                expected_text=header+'\n'+enc.decode(enc.encode(body)[offset:offset+size])
                assert card['text']==expected_text,(persona,row['query_id'],'source modified')
                counts['source_nodes_checked']+=1
            assert row['context_tokens']==len(enc.encode(row['context']))
            assert row['context_tokens']<=row['raw_context_tokens']
            assert row['rerank_candidates']<=128 and row['rerank_document_tokens']<=32000
            base_rerank=raw_rerank[str(persona)][row['query_id']]
            assert row['rerank_candidates']<=base_rerank['candidates']
            assert row['rerank_document_tokens']<=base_rerank['document_tokens']
            counts['rerank_document_tokens']+=row['rerank_document_tokens']
            counts['raw_rerank_document_tokens']+=base_rerank['document_tokens']
            assert len(row['used_sources'])<=15
            usage=row['usage'];base=raw_usage[str(persona)][row['query_id']]
            counts['native_prompt_over_raw']+=usage['prompt_tokens']>base['prompt_tokens']
            counts['native_prompt_tokens']+=usage['prompt_tokens']
            counts['native_completion_tokens']+=usage['completion_tokens']
            counts['raw_prompt_tokens']+=base['prompt_tokens']
            counts['raw_completion_tokens']+=base['completion_tokens']
        report['patients'][str(persona)]=counts
    report['complete']=sum(p['questions'] for p in report['patients'].values())==197
    (OUT/('contextual_audit.json' if args.contextual else 'audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
