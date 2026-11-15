"""Inspect the actual fact-writing and query-reward training artifacts."""
import json
from pathlib import Path
from fact_writer_protocol import parse_output, fact_text

OUT=Path(__file__).resolve().parents[2]/'results/qfad_fact_writer_rl_20261114'


def read(path):
    return json.loads(path.read_text())


def main():
    source=read(OUT/'writer_inputs.json')[0]
    assert set(source)=={'episode_id','sources'}
    assert all(set(s)=={'source_id','text','old_facts'} for s in source['sources'])
    update=read(OUT/'update_0.json')
    rewards=read(OUT/'rewards_0.json')
    values=[r['query_accuracy'] for r in rewards['samples'] if r['valid']]
    summary={'writer':'Qwen3-4B-Instruct-2507','training_patient':3,
             'writer_input_contains_only_sources':True,
             'adapter_delta_l2':update['adapter_delta_l2'],'gradient_norm':update['gradient_norm'],
             'query_reward_values':values,'query_rewards_differ':len(set(values))>1,
             'total_rewards':update['rewards'],'advantages':update['advantages'],
             'source_results':{}}
    fact_sets={}
    for path in sorted((OUT/'pilot').iterdir()):
        reward=read(path/'reward.json')
        if not reward['valid']:
            summary['source_results'][path.name]=reward
            continue
        output=read(path/'writer_output.json')
        rows=parse_output(output['text'],source['sources'])
        graph=read(path/'graph.json')
        assert len(graph['written_facts'])==sum(len(s['facts']) for s in rows)
        assert graph['relation_edges']
        expected={s['source_id'] for s in rows}
        answer_paths=sorted(path.glob('*.answer.json'))
        assert len(answer_paths)==2
        used=0
        usage={'prompt_tokens':0,'completion_tokens':0}
        for answer_path in answer_paths:
            answer=read(answer_path)
            assert answer['context_tokens']<=answer['raw_context_tokens']
            assert answer['retrieval']['rerank_document_tokens']<=32000
            assert len(answer['retrieval']['candidate_fact_indices'])<=128
            assert read(answer_path.with_name(answer_path.name.replace('.answer.json','.json')))['answer']==answer['answer']
            used+=sum(f['source_id'] in expected for f in answer['retrieved_facts'])
            for key in usage:
                usage[key]+=answer['usage'][key]
        fact_sets[path.name]={(s['source_id'],fact_text(f)) for s in rows for f in s['facts']}
        summary['source_results'][path.name]={**reward,'written_facts_retrieved':used,'reader_usage':usage,
                                              'relation_edges':len(graph['relation_edges'])}
    comparable='step0_initial' in fact_sets and 'step0_updated' in fact_sets
    summary['greedy_outputs_comparable']=comparable
    summary['greedy_facts_added']=None
    summary['greedy_facts_removed']=None
    if comparable:
        before=fact_sets['step0_initial'];after=fact_sets['step0_updated']
        summary['greedy_facts_added']=[list(v) for v in sorted(after-before)]
        summary['greedy_facts_removed']=[list(v) for v in sorted(before-after)]
    summary['parameter_update_verified']=update['adapter_delta_l2']>0 and update['gradient_norm']>0
    summary['written_facts_used_by_query']=any(r.get('written_facts_retrieved',0)>0 for r in summary['source_results'].values())
    (OUT/'pilot_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
