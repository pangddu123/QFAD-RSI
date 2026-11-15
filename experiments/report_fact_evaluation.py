"""Audit complete frozen-writer comparisons and account for writing plus answering."""
import argparse
import csv
from collections import defaultdict
from qfad_fact_graph import base
from fact_writer_protocol import parse_output, fact_text
from fact_baseline_usage import load_baseline_usage

ROOT_OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', default='')
    parser.add_argument('--split',choices=['validation','evaluation'],required=True)
    parser.add_argument('--conditions',nargs='+',default=['initial','trained'])
    parser.add_argument('--facts',type=int,default=15)
    parser.add_argument('--query-method',choices=['global','focused'],default='global')
    args=parser.parse_args()
    suffix='_focused' if args.query_method=='focused' else ''
    root=ROOT_OUT/args.run/args.split
    plan=base.read(root/'query_plan.json')
    requests={r['episode_id']:r for r in base.read(root/'writer_inputs.json')}
    expected={(r['persona'],r['query_id']) for r in plan}
    assert len(expected)==len(plan)==(41 if args.split=='validation' else 197)
    personas=sorted({r['persona'] for r in plan})
    base.harness.DATA=base.ROOT/'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    base.harness.rawrag.DATA=base.harness.DATA
    base.harness.LANGUAGE=base.harness.rawrag.LANGUAGE='en'
    histories=base.harness.load_passages(personas)
    documents={};first_stage={};raw_scores={}
    for p in personas:
        documents[p]={d['idx']:d for d in base.read(base.SOURCE/f'persona_{p:02d}/qafd/openie_results_ner_gpt-4o-mini.json')['docs']}
        first_stage[p]={}
        for stage,text in histories[p]:first_stage[p].setdefault(text,stage)
        raw_scores[p]={r['query_id']:r for path in (base.RAW/f'persona_{p:02d}').glob('session_*_scores.json') for r in base.read(path)}
    raw_usage=load_baseline_usage()
    for r in plan:
        assert r['uses_reference_or_source_annotations'] is False
        source=requests[r['writer_request_id']]['sources'][0]
        doc=documents[r['persona']][r['source_id']]
        assert source['text']==doc['passage'] and source['old_facts']==doc['extracted_triples']
        assert first_stage[r['persona']][source['text']]<=r['stage']
        assert set(source)=={'source_id','text','old_facts'}
    summary={'split':args.split,'query_method':args.query_method,'n':len(plan),'conditions':{},'raw':{
        'correct':sum(raw_scores[p][qid]['is_correct'] for p,qid in expected),
        'reader_prompt_tokens':sum(raw_usage[str(p)][qid]['usage']['prompt_tokens'] for p,qid in expected),
        'reader_completion_tokens':sum(raw_usage[str(p)][qid]['usage']['completion_tokens'] for p,qid in expected)}}
    paired=[]
    for condition in args.conditions:
        directory=root/f'{condition}{suffix}_k{args.facts}'
        found={(int(path.parent.name[1:]),base.read(path)['query_id']) for path in directory.glob('p*/*.json') if not path.name.endswith('.answer.json')}
        assert found==expected,('incomplete answers',condition,expected-found)
        active=defaultdict(dict);seen=set();correct=0;writer_usage=[0,0];reader_usage=[0,0]
        accepted=0;rejected=0;linked_facts=0;relations=0;truncated=0;wins=0;losses=0
        for item in plan:
            p=item['persona'];qid=item['query_id'];key=item['writer_request_id']
            prediction=base.read(root/f'writer_outputs/{condition}/{key}.json') if condition!='original' else None
            write=base.read(directory/f'writes/{key}.json') if condition!='original' else None
            fresh=condition!='original' and key not in seen
            if fresh:
                seen.add(key)
                assert write['first_query_id']==qid and write['stage']==item['stage']
                assert write['prompt_tokens']==prediction['prompt_tokens'] and write['completion_tokens']==prediction['completion_tokens']
                writer_usage[0]+=write['prompt_tokens'];writer_usage[1]+=write['completion_tokens']
                if write['accepted']:
                    assert prediction['finished']
                    parsed=parse_output(prediction['text'],requests[key]['sources'])
                    assert len(parsed)==1 and write['facts']==parsed[0]['facts']
                    active[p][item['source_id']]=write['facts'];accepted+=1
                    graph=write['graph']
                    assert len(graph['written_facts'])==len(write['facts'])
                    for fact in write['facts']:
                        endpoints={fact['subject'].lower().strip(),fact['object'].lower().strip()}
                        if len(endpoints)==2:
                            assert any(set(e['nodes'])==endpoints for e in graph['relation_edges'])
                    relations+=len(graph['relation_edges'])
                else:rejected+=1
            path=directory/f'p{p}/{qid}.json'
            row=base.read(path);saved=base.read(path.with_suffix('.answer.json'))
            assert row['answer']==saved['answer'] and row['response']==saved['response']
            assert row['stage']==item['stage'] and row['condition']==condition
            assert row.get('query_method','global')==args.query_method
            if args.query_method=='focused':assert row['retrieval']['anchor_source']==item['source_id']
            assert row['new_write']==fresh and set(row['active_written_sources'])==set(active[p])
            assert row['reader_configuration']=={'model':'Qwen3.8-27B','temperature':0,'seed':0,'max_tokens':16384,'enable_thinking':False}
            assert row['judge_configuration']['reasoning_effort']=='high'
            assert row['context_tokens']==len(base.ENC.encode(row['context']))<=row['raw_context_tokens']
            baseline=raw_usage[str(p)][qid]
            assert row['total_rerank_candidates']<=baseline['rerank']['candidates']
            assert row['total_rerank_document_tokens']<=baseline['rerank']['document_tokens']
            rendered=[]
            for fact in row['retrieved_facts']:
                source_id=fact['source_id'];doc=documents[p][source_id]
                assert first_stage[p][doc['passage']]<=item['stage']
                if source_id in active[p]:
                    fields={k:v for k,v in fact.items() if k not in ('source_id','passage_index')}
                    assert fields in active[p][source_id]
                    linked_facts+=1
                else:
                    triple=tuple(fact[k] for k in ('subject','relation','object'))
                    assert triple in {tuple(base.text_processing(t)) for t in doc['extracted_triples'] if len(t)==3}
                text=doc['passage'].partition('\n')[0]+'\n'+fact_text(fact)+'\nSource: '+source_id
                if fact['evidence']:text+='\nEvidence: '+fact['evidence']
                rendered.append(text)
            assert rendered==row['retrieval']['documents']
            context='\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(rendered))
            assert row['context']==base.ENC.decode(base.ENC.encode(context)[:row['raw_context_tokens']])
            for i,key_usage in enumerate(('prompt_tokens','completion_tokens')):reader_usage[i]+=row['usage'][key_usage]
            value=int(row['score']['is_correct']);raw=int(raw_scores[p][qid]['is_correct'])
            correct+=value;wins+=value>raw;losses+=value<raw;truncated+=row['finish_reason']=='length'
            paired.append({'condition':condition,'persona':p,'query_id':qid,'correct':value,'raw_correct':raw,
                           'writer_request_id':key,'new_write':fresh,'retrieved_written_facts':sum(f['source_id'] in active[p] for f in row['retrieved_facts']),
                           'reader_prompt_tokens':row['usage']['prompt_tokens'],'reader_completion_tokens':row['usage']['completion_tokens'],
                           'writer_prompt_tokens':row['writer_prompt_tokens'],'writer_completion_tokens':row['writer_completion_tokens']})
        condition_rows=[r for r in paired if r['condition']==condition]
        assert sum(r['writer_prompt_tokens'] for r in condition_rows)==writer_usage[0]
        assert sum(r['writer_completion_tokens'] for r in condition_rows)==writer_usage[1]
        total=sum(writer_usage)+sum(reader_usage)
        raw_total=summary['raw']['reader_prompt_tokens']+summary['raw']['reader_completion_tokens']
        summary['conditions'][condition]={'correct':correct,'delta_pp':100*(correct-summary['raw']['correct'])/len(plan),
            'reader_prompt_tokens':reader_usage[0],'reader_completion_tokens':reader_usage[1],
            'writer_prompt_tokens':writer_usage[0],'writer_completion_tokens':writer_usage[1],
            'writer_and_reader_tokens':total,'token_savings_vs_raw':1-total/raw_total,
            'accepted_writes':accepted,'rejected_writes':rejected,'retrieved_written_facts':linked_facts,
            'written_relation_edges_checked':relations,'wins_vs_raw':wins,'losses_vs_raw':losses,
            'truncated_answers':truncated,'source_links_and_visible_scope_verified':True,
            'context_and_rerank_budget_verified':True,
            'by_patient':{str(p):{'n':sum(r['persona']==p for r in condition_rows),
                                'correct':sum(r['correct'] for r in condition_rows if r['persona']==p)} for p in personas}}
    if 'initial' in args.conditions and 'trained' in args.conditions:
        initial={(r['persona'],r['query_id']):r for r in paired if r['condition']=='initial'}
        trained={(r['persona'],r['query_id']):r for r in paired if r['condition']=='trained'}
        wins=[{'persona':p,'query_id':qid} for p,qid in sorted(expected)
              if trained[p,qid]['correct']>initial[p,qid]['correct']]
        losses=[{'persona':p,'query_id':qid} for p,qid in sorted(expected)
                if trained[p,qid]['correct']<initial[p,qid]['correct']]
        changed=[]
        matched_contexts=0
        for p,qid in sorted(expected):
            before=base.read(root/f'initial{suffix}_k{args.facts}/p{p}/{qid}.json')
            after=base.read(root/f'trained{suffix}_k{args.facts}/p{p}/{qid}.json')
            if before['context']==after['context']:
                assert before['response']==after['response'] and before['answer']==after['answer']
                assert after['paired_answer_reused_from']=='initial'
                matched_contexts+=1
            if before['answer']==after['answer']:
                assert before['score']==after['score'] and after['paired_score_reused_from']=='initial'
        for key in requests:
            before=base.read(root/f'initial{suffix}_k{args.facts}/writes/{key}.json')
            after=base.read(root/f'trained{suffix}_k{args.facts}/writes/{key}.json')
            if before['accepted']!=after['accepted'] or before['facts']!=after['facts']:
                changed.append(key)
        summary['writer_comparison']={
            'correct_delta':len(wins)-len(losses),
            'delta_pp':100*(len(wins)-len(losses))/len(plan),
            'wins':wins,'losses':losses,'changed_source_writes':changed,
            'matched_context_queries':matched_contexts,
            'source_selection_fixed':True,'query_and_reader_fixed':True}
    base.write_json(root/f'summary{suffix}_k{args.facts}.json',summary)
    with (root/f'paired{suffix}_k{args.facts}.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print(summary)


if __name__=='__main__':main()
