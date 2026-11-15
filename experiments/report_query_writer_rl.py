"""Write the complete paired evaluation with measured inference costs."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[2]/'results/qfad_query_writer_rl_20261113'

def read(path):return json.loads(path.read_text())
def percent(n,d):return f'{100*n/d:.1f}%'

def main():
    s=read(OUT/'summary.json');audit=read(OUT/'audit.json');policy=s['policy']
    assert s['paired']['complete'] and audit['complete']
    action=str(policy['action']);main=s['evaluation'][action];coarse=s['evaluation']['2']
    raw=s['raw_answer_usage'];raw_in=sum(r['prompt_tokens'] for r in raw.values());raw_out=sum(r['completion_tokens'] for r in raw.values())
    n=s['paired']['n'];correct=s['paired']['rl_correct'];baseline=s['paired']['raw_correct']
    actual=main['answer_prompt_tokens']+main['answer_completion_tokens'];base_actual=raw_in+raw_out
    better=correct>baseline and actual<=base_actual
    headline=(f'QFAD writer 在两位患者的 {n} 题上答对 {correct} 题，RAW-RAG 答对 {baseline} 题；'
              f'准确率差为 {s["paired"]["delta_pp"]:+.1f} 个百分点，实际回答 token 用量减少 {100*(1-actual/base_actual):.1f}%。')
    text=['# QFAD query 奖励训练 writer：两位患者实验', '',headline,'',
          f'目标判定：{"达到本次两患者、在线回答预算的比较目标。" if better else "尚未达到高于 RAW-RAG 且在线回答预算不增加的目标。"}', '',
          '| 方法 | 答对 / 总题数 | 准确率 | Reader 输入 tokens | Reader 输出 tokens |',
          '|---|---:|---:|---:|---:|',
          f'| RAW-RAG | {baseline}/{n} | {percent(baseline,n)} | {raw_in:,} | {raw_out:,} |',
          f'| QFAD 固定 1024-token writer | {coarse["correct"]}/{coarse["n"]} | {percent(coarse["correct"],coarse["n"])} | {coarse["answer_prompt_tokens"]:,} | {coarse["answer_completion_tokens"]:,} |',
          f'| QFAD 学习的 {policy["writer_sizes"][policy["action"]]}-token writer | {correct}/{n} | {percent(correct,n)} | {main["answer_prompt_tokens"]:,} | {main["answer_completion_tokens"]:,} |','',
          '| 患者 | RAW-RAG | 学习 writer |', '|---|---:|---:|']
    for p,r in s['by_patient'].items():text.append(f'| {p} | {r["raw_correct"]}/{r["n"]} | {r["rl_correct"]}/{r["n"]} |')
    text+=['',f'配对比较：学习 writer 独有正确 {s["paired"]["wins"]} 题，RAW-RAG 独有正确 {s["paired"]["losses"]} 题。','',
           '## 学习机制','',
           '主体是 QFAD 的 passage–entity 图与 query-aware flow diffusion。writer 将原文写成带日期、主体及来源链接的叶节点，动作是选择 256、512 或 1024-token 粒度。回答正确性减去上下文成本构成奖励，经重要性加权 REINFORCE 更新三个策略参数。回答模型与图抽取模型保持固定。','',
           '训练来自患者 3–5 的 71 个采样问题，患者 6 用于检索验证。最终策略用训练奖励拟合一次并冻结；患者 1–2 的问题和评分不进入策略更新。显式日期检索在验证集选定，12 个带日期的训练问题使用同一检索重新收集奖励。','',
           f'最终动作概率：{", ".join(f"{size}: {prob:.4f}" for size,prob in zip(policy["writer_sizes"],policy["probabilities"]))}。','',
           '## 预算与证据核验','',
           f'学习 writer 的上下文上限违规数为 {main["context_budget_violations"]}，回答输出截断数为 {main["truncated_answers"]}。每题最多 128 个重排候选、32,000 个候选文档 tokens、15 个证据叶节点；上下文不超过同题 RAW-RAG 的长度。上下文与候选上限使用 cl100k_base，表中的 Reader 用量为模型接口返回的原生 tokens。','',
           f'已核对全部 {n} 题的来源与时间范围，检查 {sum(p["source_nodes_checked"] for p in audit["patients"].values()):,} 个引用片段。片段与当时可见原文的 token 切片完全一致，保存答案未被评分步骤修改，冻结策略可由训练记录重放。','',
           '已保存完整响应的离线训练、验证与探索成本如下；这些成本另列，不计入部署后每次查询的预算。已有 QFAD 抽取与嵌入的冷启动总成本未纳入这些回答 token 统计。','',
           '| 阶段 | 回答次数 | 输入 tokens | 输出 tokens |','|---|---:|---:|---:|']
    for key,label in [('training','训练行为采样'),('train_date','日期查询奖励'),('validation','writer 验证'),('budget_probe','上下文装填验证'),('date_probe','日期检索验证'),('lexical_probe','词项召回验证'),('contextual_probe','记忆量条件 writer 验证')]:
        rows=s.get(key,{}).values()
        text.append(f'| {label} | {sum(r["n"] for r in rows)} | {sum(r["answer_prompt_tokens"] for r in rows):,} | {sum(r["answer_completion_tokens"] for r in rows):,} |')
    sem=s.get('semantic_evaluation',{})
    if sem:
        assert sem['raw']['n']==n and sem[f'a{action}']['n']==n
        text+=['','## 统一语义评分','',
               '以下为相同盲评 rubric 的补充指标，接受同义表达，不替代上面的原始基准主指标。所有方法均由 gpt-6.1-sol、high 评分。','',
               '| 方法 | 语义评分答对 / 总题数 |','|---|---:|']
        for name,label in [('raw','RAW-RAG'),('a2','QFAD 固定 writer'),(f'a{action}','QFAD 学习 writer')]:
            r=sem[name];text.append(f'| {label} | {r["correct"]}/{r["n"]} |')
    text+=['','## 限制','',
           '这是三个动作的 writer 策略学习，尚不能代表语言模型 writer 微调。目标两位患者在此前诊断实验中已被检查，本次属于探索性比较。仅有两位患者，不能由问题级统计推断跨患者泛化。在线预算与模型冷启动、离线研发总成本是不同口径。固定 writer 对照与 RAW-RAG 比较也不能单独分离所有检索改动的贡献。','',
           '训练样本有一题、验证样本有两题存在题面记录日期与参考日期冲突，详见 [DATA_QUALITY.md](DATA_QUALITY.md)。当前策略与主指标沿用原始数据，因此这些冲突也影响奖励与验证结果。','',
           '详细协议见 [PLAN.md](PLAN.md)，逐题配对见 [paired.csv](paired.csv)，预算与汇总见 [summary.json](summary.json)，证据核验见 [audit.json](audit.json)。','']
    (OUT/'REPORT.md').write_text('\n'.join(text))

if __name__=='__main__':main()
