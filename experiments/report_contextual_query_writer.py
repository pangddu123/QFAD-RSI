"""Summarize the complete frozen contextual-writer evaluation."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/qfad_query_writer_rl_20261113'
RAW=ROOT/'results/rawrag_medmemorybench_en_mixed_qwen38_20261111'
def read(path):return json.loads(path.read_text())

def main():
    audit=read(OUT/'contextual_audit.json');assert audit['complete']
    policy=read(OUT/'frozen_contextual_policy.json');rows=[];selected=[];raw_semantic=[];selected_semantic=[]
    for persona in (1,2):
        raw={r['query_id']:r for p in (RAW/f'persona_{persona:02d}').glob('session_*_scores.json') for r in read(p)}
        manifest=read(OUT/f'contextual_evaluation_p{persona}.json')
        assert {r['query_id'] for r in manifest}==set(raw)
        for item in manifest:
            r=read(OUT/item['path']);b=raw[r['query_id']];selected.append(r)
            sem=read(OUT/'semantic'/item['path']);bs=read(OUT/f'semantic/evaluation/p{persona}/{r["query_id"]}_raw.json')
            selected_semantic.append(sem['correct']);raw_semantic.append(bs['correct'])
            rows.append({'persona':persona,'query_id':r['query_id'],'action':r['action'],
                         'raw_correct':int(b['is_correct']),'rl_correct':int(r['score']['is_correct']),
                         'raw_semantic':int(bs['correct']),'rl_semantic':int(sem['correct']),
                         'raw_answer':b['model_output'],'rl_answer':r['answer'],
                         'prompt_tokens':r['usage']['prompt_tokens'],'completion_tokens':r['usage']['completion_tokens'],
                         'source_result':item['path']})
    assert len(rows)==197
    with (OUT/'contextual_paired.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    n=len(rows);correct=sum(r['rl_correct'] for r in rows);baseline=sum(r['raw_correct'] for r in rows)
    usage={'prompt_tokens':sum(r['prompt_tokens'] for r in rows),'completion_tokens':sum(r['completion_tokens'] for r in rows)}
    raw_usage={'prompt_tokens':sum(p['raw_prompt_tokens'] for p in audit['patients'].values()),
               'completion_tokens':sum(p['raw_completion_tokens'] for p in audit['patients'].values())}
    savings=1-sum(usage.values())/sum(raw_usage.values())
    summary={'n':n,'primary_correct':correct,'raw_primary_correct':baseline,'delta_pp':100*(correct-baseline)/n,
             'semantic_correct':sum(selected_semantic),'raw_semantic_correct':sum(raw_semantic),
             'usage':usage,'raw_usage':raw_usage,'answer_token_savings':savings,
             'wins':sum(r['rl_correct']>r['raw_correct'] for r in rows),'losses':sum(r['rl_correct']<r['raw_correct'] for r in rows),
             'writer_action_counts':dict(Counter(r['action'] for r in rows)),
             'by_patient':{str(p):{'n':sum(r['persona']==p for r in rows),
                                 'primary_correct':sum(r['rl_correct'] for r in rows if r['persona']==p),
                                 'raw_primary_correct':sum(r['raw_correct'] for r in rows if r['persona']==p),
                                 'semantic_correct':sum(r['rl_semantic'] for r in rows if r['persona']==p),
                                 'raw_semantic_correct':sum(r['raw_semantic'] for r in rows if r['persona']==p)} for p in (1,2)},
             'validation':policy['validation_selection'],'complete':True}
    (OUT/'contextual_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    original=read(OUT/'summary.json')
    table=[('RAW-RAG',baseline,sum(raw_semantic),raw_usage['prompt_tokens'],raw_usage['completion_tokens'])]
    for action,label in [('2','QFAD 固定 1024-token writer'),('0','QFAD 全局 RL writer')]:
        group=original['evaluation'][action];sem=original['semantic_evaluation']['a'+action]
        assert group['n']==197 and sem['n']==197
        table.append((label,group['correct'],sem['correct'],group['answer_prompt_tokens'],group['answer_completion_tokens']))
    table.append(('QFAD 记忆量条件 RL writer',correct,sum(selected_semantic),usage['prompt_tokens'],usage['completion_tokens']))
    text=['# QFAD：query 奖励训练记忆量条件 writer','',
          f'在前两位患者全部 {n} 题上，冻结的 RL writer 答对 {correct} 题，RAW-RAG 答对 {baseline} 题，准确率差为 {summary["delta_pp"]:+.1f} 个百分点。Reader 实际输入与输出 token 总量减少 {100*savings:.1f}%。','',
          '| 方法 | 原始基准答对 / 197 | 统一语义评分 / 197 | 输入 tokens | 输出 tokens |',
          '|---|---:|---:|---:|---:|']
    for label,primary,semantic,prompt,completion in table:text.append(f'| {label} | {primary} | {semantic} | {prompt:,} | {completion:,} |')
    text+=['','| 患者 | RAW-RAG 原始指标 | RL writer 原始指标 | RAW-RAG 语义指标 | RL writer 语义指标 |','|---|---:|---:|---:|---:|']
    for p,r in summary['by_patient'].items():text.append(f'| {p} | {r["raw_primary_correct"]}/{r["n"]} | {r["primary_correct"]}/{r["n"]} | {r["raw_semantic_correct"]}/{r["n"]} | {r["semantic_correct"]}/{r["n"]} |')
    lexical=read(OUT/'lexical_validation.json')
    text+=['','## 验证集比较','',
           '| 方法（患者 6，41 题） | 原始基准答对 | 统一语义评分答对 |',
           '|---|---:|---:|',
           '| RAW-RAG | 14 | 22 |',
           '| 全局 writer，混合检索 | 10 | 18 |',
           '| 全局 writer，日期定位检索 | 13 | 21 |',
           f'| 全局 writer，日期定位与词项召回 | {lexical["primary_correct"]} | {lexical["semantic_correct"]} |',
           '| 记忆量条件 writer，日期定位检索 | 14 | 21 |']
    text+=['',f'逐题配对中，RL writer 独有正确 {summary["wins"]} 题，RAW-RAG 独有正确 {summary["losses"]} 题。','',
           '## query 如何训练 writer','',
           'QFAD 保留 passage–entity 图与 query-aware flow diffusion。writer 在问题到来之前，根据当前可见源段落数量选择 256、512 或 1024-token 原文片段。叶节点携带原始日期、主体和来源链接。六个策略参数通过 query 的回答正确性及上下文成本奖励学习；图抽取模型与回答模型固定。','',
           '训练使用患者 3–5 的 71 个采样问题。状态是截距和标准化后的 log(1 + 可见段落数)，策略为 softmax(Wx)。采用一次重要性加权 REINFORCE 更新，学习率 0.8，以同题 RAW-RAG 奖励作与动作独立的基线。原始行为采样概率被保留，最终奖励统一使用 gpt-6.1-sol high。','',
           '患者 6 的 41 题用于策略选择。记忆量条件策略的原始指标为 14/41，全局策略为 13/41，两者语义指标均为 21/41；RAW-RAG 对应为 14/41、22/41。条件策略据此冻结，目标患者的答案和奖励不进入训练或选择。','',
           '目标患者在相同模型、问题、writer 动作和检索配置下已有的答案直接复用；每题使用一个动作对应的单次回答。逐题动作、结果路径和 token 用量保存在配对表中。','',
           '## 预算和验证','',
           '每题最多 128 个重排候选、32,000 个候选文档 tokens 和 15 个证据叶节点，上下文不超过同题 RAW-RAG 的长度。实际 RAW-RAG 重排请求每题包含 128 个候选，文档 token 数至少为 32,655。上述上限使用 cl100k_base；结果表使用回答接口返回的模型原生 token 用量。','',
           f'已完成全部 {n} 题的来源、时间范围和预算核验，并从训练奖励重放冻结的六个策略参数。源片段逐一等于可见原文的相应 token 切片，评分过程未修改模型答案。模型计算使用 2A100，judge 在本地调用；共享模型服务保持运行。','',
           '离线训练、验证及探索成本另见 [REPORT.md](REPORT.md) 的成本表。','',
           '## 限制','',
           '本方法学习的是源片段写入粒度，尚非语言模型 writer 微调。两位目标患者此前用于诊断工作，本结果是探索性比较，不能据此推断跨患者泛化。原始基准与统一语义指标同时报告。训练中一题、验证中两题存在题面记录日期与参考日期冲突，详见 [DATA_QUALITY.md](DATA_QUALITY.md)。在线回答及重排预算不等于包括图抽取、嵌入和所有探索实验的生命周期总成本。','',
           '数据：[contextual_summary.json](contextual_summary.json)、[contextual_paired.csv](contextual_paired.csv)、[contextual_audit.json](contextual_audit.json)、[frozen_contextual_policy.json](frozen_contextual_policy.json)。','']
    (OUT/'CONTEXTUAL_REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
