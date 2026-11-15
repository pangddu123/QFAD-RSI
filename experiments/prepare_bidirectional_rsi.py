"""Build the two-patient RSI feedback pool from recorded queries and verified sources."""
import csv
from pathlib import Path
from qfad_fact_graph import base


def main():
    out = base.ROOT / 'results/qfad_bidirectional_rsi_p12_20261114'
    controls = base.ROOT / 'QFAD-RSI/results/p12_manual_controls_20261113'
    cases = {r['case_id']: r for r in csv.DictReader((controls / 'case_results.csv').open())}
    base.harness.DATA = base.ROOT / 'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    base.harness.rawrag.DATA = base.harness.DATA
    base.harness.LANGUAGE = base.harness.rawrag.LANGUAGE = 'en'
    history = base.harness.load_passages([1, 2])
    pool = []
    inputs = []
    for persona in (1, 2):
        documents = base.read(base.SOURCE / f'persona_{persona:02d}/qafd/openie_results_ner_gpt-4o-mini.json')['docs']
        by_text = {d['passage']: d for d in documents}
        visible = {}
        for stage, text in history[persona]:
            visible.setdefault(text, stage)
        questions = base.read(base.harness.DATA / f'persona_{persona}/eval/generated_queries.json')['queries']
        raw_primary = {s['query_id']: s for p in (base.RAW / f'persona_{persona:02d}').glob('session_*_scores.json') for s in base.read(p)}
        for q in questions:
            qid = q['query_id']
            key = f'p{persona}_{qid}'
            if key not in cases:
                continue
            checkpoint = int(qid.split('_')[1])
            stage = checkpoint // 10 - 1
            semantic = base.read(base.ROOT / f'results/qfad_query_writer_rl_20261113/semantic/evaluation/p{persona}/{qid}_raw.json')
            selected = None
            for directory in (controls / 'source_control', controls / 'source_control/supplement',
                              controls / 'source_control/manual_supplement'):
                candidate = directory / 'candidates' / f'{key}.json'
                verification = directory / 'verification' / f'{key}.json'
                score = directory / 'scores' / f'{key}.json'
                if candidate.exists() and verification.exists() and score.exists():
                    if base.read(verification).get('supported') is True:
                        selected = (base.read(candidate), base.read(score)[0], str(candidate), str(verification))
            source_rows = []
            missing = []
            if selected:
                for text in selected[0]['context']:
                    if text not in by_text:
                        missing.append(text)
                        continue
                    assert visible[text] <= stage
                    d = by_text[text]
                    source_rows.append({'source_id': d['idx'], 'text': text, 'old_facts': d['extracted_triples']})
            c_correct = bool(selected and selected[1]['is_correct'])
            row = {'episode_id': key, 'persona': persona, 'query_id': qid, 'checkpoint': checkpoint,
                   'stage': stage, 'query_type': q['query_type'], 'question': q['question'],
                   'raw_primary_correct': bool(raw_primary[qid]['is_correct']),
                   'raw_semantic_correct': semantic['correct'],
                   'qfad_primary_correct': cases[key]['baseline_correct'] == 'True',
                   'verified_source_reader_correct': c_correct,
                   'priority_gain_over_raw': c_correct and not semantic['correct'],
                   'source_ids': [s['source_id'] for s in source_rows],
                   'unmapped_context_count': len(missing),
                   'feedback_source_file': selected[2] if selected else None,
                   'source_verification_file': selected[3] if selected else None}
            pool.append(row)
            if source_rows and not missing:
                inputs.append({'episode_id': key, 'sources': source_rows})
    assert len(pool) == 197 and len({r['episode_id'] for r in pool}) == 197
    ready = {r['episode_id'] for r in inputs}
    first = sorted([r for r in pool if r['priority_gain_over_raw'] and r['episode_id'] in ready],
                   key=lambda r: (r['checkpoint'], r['persona'], r['query_id']))
    base.write_json(out / 'feedback_pool.json', pool)
    base.write_json(out / 'writer_source_inputs.json', inputs)
    base.write_json(out / 'round0_priority_queries.json', first)
    summary = {'queries': len(pool), 'patients': [1, 2],
               'raw_primary_correct': sum(r['raw_primary_correct'] for r in pool),
               'raw_semantic_correct': sum(r['raw_semantic_correct'] for r in pool),
               'verified_source_inputs': len(inputs),
               'priority_queries_with_complete_source_mapping': len(first),
               'learning_scope': 'repeated adaptation on the same two patients',
               'reader_answer_model_frozen': 'Qwen3.8-27B',
               'learned_components': ['writer facts and relations', 'reader evidence selection']}
    base.write_json(out / 'prepared.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
