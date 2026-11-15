"""Audit source windows and report the frozen anchor-writer comparison."""
from collections import Counter
import csv
import json
import numpy as np
from qfad_anchor_writer import AnchorPatient, OUT, ENC, base
from fit_anchor_policy import choose, fit


def main():
    frozen = base.read(OUT / 'frozen_policy.json')
    policy = frozen['policy']
    replay = fit(policy['contextual'])
    assert np.allclose(replay['weights'], policy['weights'], atol=1e-12, rtol=0)
    assert policy['training_personas'] == [3, 4, 5] and policy['episodes'] == 70
    raw_usage = base.read(base.OUT / 'raw_usage.json')
    raw_rerank = base.read(base.OUT / 'raw_rerank_usage.json')
    rows = []
    cards_checked = 0
    for persona in (1, 2):
        patient = AnchorPatient(persona)
        baseline = {r['query_id']: r for p in (base.RAW / f'persona_{persona:02d}').glob('session_*_scores.json')
                    for r in base.read(p)}
        expected = {q['query_id'] for _, qq in patient.groups for q in qq}
        assert expected == set(baseline)
        for stage, (checkpoint, queries) in enumerate(patient.groups):
            view = patient.view(stage)
            action = choose(policy, len(view['anchors']))
            for q in queries:
                name = f'{q["query_id"]}_a{action}.json'
                path = OUT / f'evaluation/p{persona}/{name}'
                r = base.read(path)
                assert r['action'] == action and r['stage'] == stage and r['checkpoint'] == checkpoint
                assert r['answer'] == base.read(path.with_suffix('.answer.json'))['answer']
                assert r['judge_configuration']['reasoning_effort'] == 'high'
                assert r['context_tokens'] == len(ENC.encode(r['context'])) <= r['raw_context_tokens']
                retrieval = patient.retrieve(stage, q)
                assert len(retrieval['candidate_indices']) <= raw_rerank[str(persona)][q['query_id']]['candidates']
                assert retrieval['document_tokens'] <= raw_rerank[str(persona)][q['query_id']]['document_tokens']
                assert all(0 <= i < len(view['anchors']) for i in retrieval['candidate_indices'])
                assert len(r['used_sources']) <= 15
                context = ''
                for i, card in enumerate(r['used_sources']):
                    assert card == patient.payload(view, card['anchor_index'], base.SIZES[action])
                    assert 0 <= card['start_token'] < card['end_token'] <= len(view['tokens'])
                    part = f'Memory {i+1}:\n{card["text"]}\n\n'
                    context += ENC.decode(ENC.encode(part)[:r['raw_context_tokens']-len(ENC.encode(context))])
                    cards_checked += 1
                assert context == r['context']
                sem = base.read(OUT / f'semantic/evaluation/p{persona}/{name}')
                raw_sem = base.read(OUT / f'semantic/evaluation/p{persona}/{q["query_id"]}_raw.json')
                usage = raw_usage[str(persona)][q['query_id']]
                rows.append({'persona': persona, 'query_id': q['query_id'], 'action': action,
                             'primary': int(r['score']['is_correct']), 'raw_primary': int(baseline[q['query_id']]['is_correct']),
                             'semantic': int(sem['correct']), 'raw_semantic': int(raw_sem['correct']),
                             'prompt_tokens': r['usage']['prompt_tokens'], 'completion_tokens': r['usage']['completion_tokens'],
                             'raw_prompt_tokens': usage['prompt_tokens'], 'raw_completion_tokens': usage['completion_tokens'],
                             'truncated': r['finish_reason'] == 'length'})
    assert len(rows) == 197 and len({(r['persona'], r['query_id']) for r in rows}) == 197
    fields = ['primary', 'raw_primary', 'semantic', 'raw_semantic', 'prompt_tokens', 'completion_tokens',
              'raw_prompt_tokens', 'raw_completion_tokens', 'truncated']
    def aggregate(selected):
        return {'n': len(selected), **{k: sum(r[k] for r in selected) for k in fields}}
    summary = aggregate(rows)
    summary.update({'by_patient': {str(p): aggregate([r for r in rows if r['persona'] == p]) for p in (1, 2)},
                    'wins': sum(r['primary'] > r['raw_primary'] for r in rows),
                    'losses': sum(r['primary'] < r['raw_primary'] for r in rows),
                    'actions': dict(Counter(r['action'] for r in rows)), 'source_cards_checked': cards_checked,
                    'source_scope_and_context_verified': True, 'policy_replay_verified': True,
                    'rerank_budget_verified': True, 'validation': frozen['validation']})
    summary['answer_token_savings'] = 1 - (summary['prompt_tokens'] + summary['completion_tokens']) / (
        summary['raw_prompt_tokens'] + summary['raw_completion_tokens'])
    base.write_json(OUT / 'summary.json', summary)
    with (OUT / 'paired.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = ['# QFAD query 奖励训练原文窗口 writer', '',
            f'前两位患者全部 197 题中，冻结策略答对 {summary["primary"]} 题，RAW-RAG 答对 {summary["raw_primary"]} 题。Reader 实际输入与输出 token 总量减少 {100*summary["answer_token_savings"]:.1f}%。', '',
            '| 方法 | 原始基准正确 / 197 | 统一语义正确 / 197 | 输入 tokens | 输出 tokens |',
            '|---|---:|---:|---:|---:|',
            f'| RAW-RAG | {summary["raw_primary"]} | {summary["raw_semantic"]} | {summary["raw_prompt_tokens"]:,} | {summary["raw_completion_tokens"]:,} |',
            f'| QFAD RL writer | {summary["primary"]} | {summary["semantic"]} | {summary["prompt_tokens"]:,} | {summary["completion_tokens"]:,} |', '',
            'QFAD 实体图与 query-aware flow diffusion 用于检索。256-token 原文锚点使用既有向量，通过原文中的实体提及关联图节点。writer 从 256、512、1024-token 原文窗口中选择写入粒度，保留原始日期、主体和来源区间。回答模型和提示保持相同。', '',
            '训练使用患者 3–5 的 70 条查询奖励，均匀采样动作并记录行为概率；一次重要性加权 REINFORCE 更新，以 RAW-RAG 同题奖励为基线。奖励是原始基准正确性减去 0.02 倍的相对上下文成本。全局与记忆量条件策略仅在患者 6 上选择，正确数相同时选择回答 tokens 较少者，随后冻结。', '',
            f'逐题比较：RL writer 独有正确 {summary["wins"]} 题，RAW-RAG 独有正确 {summary["losses"]} 题。全部 {cards_checked} 个来源窗口、可见时间范围、实际上下文重建和重排预算通过核验；冻结参数可从训练奖励重放。', '',
            '模型推理使用 2A100 的 Qwen3.8-27B-FP8，temperature=0、max_tokens=16384、关闭 thinking；judge 在本地使用 gpt-6.1-sol high。每题最多 128 个候选、32,000 个重排文档 tokens、15 个证据窗口，Reader 上下文不超过同题 RAW-RAG。', '',
            '## 限制', '',
            '本实验学习窗口写入策略，未微调语言模型 writer。目标患者此前用于诊断实验，因此属于探索性比较。原始评分存在同义表述判错；统一语义评分也可能放宽细节要求，两套指标均完整保留。患者 6 的两题存在题面日期与参考日期冲突。预算结论覆盖在线回答与重排，图构建、嵌入、训练和验证成本另计。', '',
            '数据：[summary.json](summary.json)、[paired.csv](paired.csv)、[frozen_policy.json](frozen_policy.json)。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
