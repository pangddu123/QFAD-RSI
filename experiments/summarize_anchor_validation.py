"""Validate source evidence and summarize the complete anchor-index pilot."""
import json
from qfad_anchor_writer import AnchorPatient, OUT, ENC, base

def main():
    patient=AnchorPatient(6);checkpoints=(20,40,60,80)
    queries={q['query_id']:(stage,ck,q) for stage,(ck,qq) in enumerate(patient.groups) if ck in checkpoints for q in qq}
    paths=list((OUT/'validation_fixed/p6').glob('*_a1.json'))
    rows=[base.read(p) for p in paths];assert {r['query_id'] for r in rows}==set(queries)
    source_count=0;raw={};semantic=[];raw_semantic=[]
    for ck in checkpoints:
        raw.update({r['query_id']:r for r in base.read(base.RAW/f'persona_06/session_{ck:03d}_scores.json')})
    for path,row in zip(paths,rows):
        stage,ck,q=queries[row['query_id']];view=patient.view(stage)
        assert row['checkpoint']==ck and row['stage']==stage and row['action']==1
        assert row['answer']==base.read(path.with_suffix('.answer.json'))['answer']
        assert row['judge_configuration']['reasoning_effort']=='high'
        retrieval=patient.retrieve(stage,q)
        assert len(retrieval['candidate_indices'])<=128 and retrieval['document_tokens']<=32000
        assert all(0<=i<len(view['anchors']) for i in retrieval['candidate_indices'])
        for card in row['used_sources']:
            start,end=card['start_token'],card['end_token']
            assert 0<=start<end<=len(view['tokens']) and end-start<=512
            headers=patient.source_headers(view,start,end)
            assert card['source_headers']==headers
            assert card['text']=='\n'.join(headers)+'\n'+ENC.decode(view['tokens'][start:end])
            source_count+=1
        assert row['context_tokens']==len(ENC.encode(row['context']))<=row['raw_context_tokens']
        assert len(row['used_sources'])<=15
        semantic.append(base.read(OUT/f'semantic/validation_fixed/p6/{path.name}')['correct'])
        raw_semantic.append(base.read(OUT/f'semantic/validation_fixed/p6/{row["query_id"]}_raw.json')['correct'])
    summary={'n':len(rows),'primary_correct':sum(r['score']['is_correct'] for r in rows),
             'raw_primary_correct':sum(raw[r['query_id']]['is_correct'] for r in rows),
             'semantic_correct':sum(semantic),'raw_semantic_correct':sum(raw_semantic),
             'source_nodes_checked':source_count,'source_scope_verified':True,
             'context_budget_violations':sum(r['context_tokens']>r['raw_context_tokens'] for r in rows),
             'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
             'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),
             'truncated_answers':sum(r['finish_reason']=='length' for r in rows)}
    base.write_json(OUT/'validation_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
