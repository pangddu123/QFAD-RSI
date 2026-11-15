"""Compare frozen Writer checkpoints on a label-free additional-patient cohort."""
import argparse
import json
from pathlib import Path

from graph_environment import base


def embedding_usage(directory):
    config = base.read(directory / 'configuration.json')
    path = base.ROOT / 'results/qwen27b_rsi/embedding_usage' / f'{config["process_id"]}.jsonl'
    if not path.exists():
        return {'process_id': config['process_id'], 'requests': 0, 'tokens': 0}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {'process_id': config['process_id'], 'requests': len(rows),
            'tokens': sum(row['native_usage']['total_tokens'] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    root, prefix = cohort.parent, cohort.name
    config = base.read(cohort / 'configuration.json')
    expected = set(config['query_ids'])
    raw_summary = base.read(cohort / 'raw_summary.json')
    raw_usage = base.read(cohort / 'raw_usage.json')
    retrieval = base.read(cohort / 'retrieval_budget.json')
    raw = {}
    for episode in expected:
        persona, qid = episode.split('_', 1)
        raw[episode] = base.read(cohort / 'raw_scores' / persona / f'{qid}_raw.json')['correct']
    assert sum(raw.values()) == raw_summary['correct'] and len(expected) == config['queries']
    source_index = sum(p['graph_source_index_native_tokens'] for p in retrieval['patients'].values())
    raw_embedding = sum(p['raw_final_index_once_native_tokens'] + sum(p['raw_query_embedding_native_tokens'].values())
                        for p in retrieval['patients'].values())
    raw_rerank = sum(p['raw_rerank_native_input_tokens'] for p in retrieval['patients'].values())
    reader = root / f'{prefix}_reader'
    assert base.read(reader / 'complete.json') == {'queries': len(config['episodes']), 'worker_exit': 0}
    feedback_embedding = embedding_usage(reader)
    feedback_judge = sum(base.read(p).get('judge_usage', {}).get('total_tokens', 0)
                         for p in (reader / 'answers').glob('*.json'))
    conditions, answers = {}, {}
    for name in ('empty', 'before', 'after'):
        run = root / f'{prefix}_{name}_full'
        summary = base.read(run / 'summary.json')
        rows = {p.stem: base.read(p) for p in (run / 'answers').glob('*.json')}
        assert set(rows) == expected and summary['queries'] == len(expected)
        assert sum(r['score']['correct'] for r in rows.values()) == summary['correct']
        final_embedding = embedding_usage(run)
        writer_tokens = feedback_tokens = source_judge = feedback_rerank = 0
        generation_embedding = {'tokens': 0, 'requests': 0}
        writes = rejected = noops = 0
        if name != 'empty':
            writer = root / f'{prefix}_writer_{name}'
            wconfig = base.read(writer / 'configuration.json')
            assert wconfig['episodes'] == config['episodes']
            assert wconfig['policy_frozen'] and wconfig['correctness_feedback'].startswith('withheld')
            assert base.read(writer / 'complete.json') == {'episodes': len(config['episodes']), 'worker_exit': 0}
            assert base.read(writer / 'worker_runtime.json')['step'] == (20 if name == 'before' else 26)
            assert base.read(writer / 'initial_graph.json') == base.read(cohort / 'initial_graph.json')
            assert base.read(writer / 'graph_state.json') == base.read(run / 'graph_snapshot.json')
            assert not list(writer.glob('step*.update.json'))
            records = [base.read(writer / f'step{i}.result.json') for i in range(len(config['episodes']))]
            for i, record in enumerate(records):
                observation = base.read(writer / f'step{i}.observation.json')
                observed = json.loads(observation[-1]['content'])
                assert observed['feedback']['reader_answer_correct'] is None
                assert 'reader_generated_answer' not in observed['feedback']
                assert record['feedback']['correct'] is None
            writer_tokens = sum(r['writer_tokens'] for r in records)
            source_judge = sum(r['support'].get('usage', {}).get('total_tokens', 0) for r in records)
            feedback_tokens = wconfig['feedback_model_tokens']
            feedback_rerank = wconfig['feedback_rerank_native_input_tokens']
            generation_embedding = embedding_usage(writer)
            writes = sum(r['graph_changed'] for r in records)
            rejected = sum(not r['support']['supported'] for r in records)
            noops = sum((r.get('action') or {}).get('operation') == 'NOOP' for r in records)
        else:
            assert base.read(run / 'graph_snapshot.json') == base.read(cohort / 'initial_graph.json')
        answer_tokens = sum(r['usage']['total_tokens'] for r in rows.values())
        qwen = answer_tokens + writer_tokens + feedback_tokens
        embeddings = source_index + final_embedding['tokens'] + generation_embedding['tokens']
        if name != 'empty':
            embeddings += feedback_embedding['tokens']
        rerank = summary['rerank_native_input_tokens'] + feedback_rerank
        conditions[name] = {
            'correct': summary['correct'], 'queries': len(rows),
            'patients': {str(p): {'queries': sum(e.startswith(f'p{p}_') for e in rows),
                                  'correct': sum(r['score']['correct'] for e, r in rows.items() if e.startswith(f'p{p}_'))}
                         for p in config['patients']},
            'answer_tokens': answer_tokens, 'writer_tokens': writer_tokens,
            'feedback_tokens': feedback_tokens, 'qwen_total_tokens': qwen,
            'source_judge_tokens': source_judge,
            'feedback_judge_tokens_charged': feedback_judge if name != 'empty' else 0,
            'all_llm_native_tokens': qwen + source_judge + (feedback_judge if name != 'empty' else 0),
            'embedding_tokens': embeddings, 'rerank_native_input_tokens': rerank,
            'embedding_records': {'final': final_embedding, 'writer': generation_embedding,
                                  'feedback': feedback_embedding if name != 'empty' else None},
            'supported_writes': writes, 'unsupported_or_invalid': rejected, 'noops': noops,
            'wins_vs_raw': [e for e, r in rows.items() if r['score']['correct'] and not raw[e]],
            'losses_vs_raw': [e for e, r in rows.items() if not r['score']['correct'] and raw[e]],
            'answers_above_individual_raw_tokens': [
                {'episode': e, 'excess_tokens': r['usage']['total_tokens'] - raw_usage[e.split('_', 1)[0][1:]][r['query_id']]['total_tokens']}
                for e, r in rows.items()
                if r['usage']['total_tokens'] > raw_usage[e.split('_', 1)[0][1:]][r['query_id']]['total_tokens']]
        }
        answers[name] = rows
    gained = [e for e in expected if answers['after'][e]['score']['correct'] and not answers['before'][e]['score']['correct']]
    lost = [e for e in expected if not answers['after'][e]['score']['correct'] and answers['before'][e]['score']['correct']]
    report = {'configuration': config, 'queries': len(expected), 'conditions': conditions,
              'raw': {**raw_summary, 'embedding_tokens': raw_embedding, 'rerank_native_input_tokens': raw_rerank},
              'gained_after_vs_before': sorted(gained), 'lost_after_vs_before': sorted(lost),
              'scope': 'Frozen checkpoints on patients 3 and 4, empty initial fact graphs, deterministic query triggers, no answer-correctness labels supplied to Writer; no parameter updates',
              'budget_scope': 'Source index, all selected Reader feedback acquisition, Writer generation and verification, final retrieval and answering. Reader answer scoring is conservatively charged although labels are withheld; final experimental accuracy scoring is separate.',
              'comparison_to_p12': 'Different cohort, empty initialization and withheld correctness feedback; this measures frozen transfer, not a repetition of the label-assisted patient-1/2 adaptation protocol.'}
    base.write_json(cohort / 'comparison.json', report)
    labels = [('RAW-RAG', report['raw']), ('空事实图', conditions['empty']),
              ('训练前 Writer', conditions['before']), ('训练后 Writer', conditions['after'])]
    lines = ['# 第 3、4 位患者的冻结模型迁移评估', '',
             f'全部 {len(expected)} 题；Writer checkpoint 固定在训练前 step 20 和训练后 step 26，回答器与 QFAD Reader 固定。', '',
             '| 条件 | 患者 3 | 患者 4 | 合计正确题数 | Qwen27B 累计 tokens |',
             '|---|---:|---:|---:|---:|']
    for label, row in labels:
        lines.append(f'| {label} | {row["patients"]["3"]["correct"]}/{row["patients"]["3"]["queries"]} | '
                     f'{row["patients"]["4"]["correct"]}/{row["patients"]["4"]["queries"]} | '
                     f'{row["correct"]}/{len(expected)} | {row.get("qwen_total_tokens", row.get("qwen_tokens")):,} |')
    lines += ['', f'训练后相对训练前，{len(gained)} 题改善、{len(lost)} 题退化，净增 {len(gained)-len(lost)} 题。', '',
              '两位新患者各自从空事实图开始。每个检查点按固定规则选择第一道时间定位题，缺少时选择第一道实体题，生成一条来源支持的图操作。问题选择在评分前固定，参考答案及正确性标签不进入 Writer，所有 189 题均纳入最终评价。', '',
              '## 范围', '',
              '该实验评估冻结权重在新患者上的迁移；未用新患者继续训练。与前两患者实验相比，初始图为空且不提供答题正确性反馈，不能把不同患者间的准确率差直接解释为训练增益。预算按累计部署原生 token 计数，最终实验评分另列；不代表逐题、GPU 时间或金额均更低。', '',
              '[逐题变化与完整成本](comparison.json)。']
    (cohort / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'queries': len(expected), 'raw': report['raw']['correct'],
                      'conditions': {k: v['correct'] for k, v in conditions.items()},
                      'gained': len(gained), 'lost': len(lost)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
