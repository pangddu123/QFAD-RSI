# Target Writer 开发交接与实验环境

记录日期 2026-11-15，核对的算法提交为 `57c9ccb`。本仓库默认开发分支是 `freeRL-RSI`。开发者可以在其他机器或自己的 Git 远端改代码，数据准备、完整测试、训练和配对评测继续使用原实验环境。

## 在新位置开发

```bash
git clone --branch freeRL-RSI https://github.com/pangddu123/QFAD-RSI.git
cd QFAD-RSI
git branch --show-current
python3 -m compileall -q experiments/qwen27b_rsi experiments/query_writer_rl.py
PYTHONPATH=experiments/qwen27b_rsi python3 -m unittest test_clinical_evidence
```

以上检查使用 Python 标准库，覆盖语法与临床证据奖励计算。完整入口会在导入时加载工作区的医学基准、提示模板、分词器或缓存。在仅有源码的开发机上，运行完整 CLI 的 `--help` 也可能因缺少这些资源而失败。

若开发远端与实验远端不同，提交修改后记录 `git rev-parse HEAD`，把该提交同步到实验仓库的 `freeRL-RSI`。在原实验机的对应检出目录执行 `git pull --ff-only origin freeRL-RSI`，再次核对提交号，再执行下文的完整检查。两个远端之间需要显式推送或合入提交。

| 要改的行为 | 主要文件（相对仓库根目录） |
|---|---|
| 医学抽取 Prompt、逐字段核验、逐字引文和原文行号 | `experiments/qwen27b_rsi/medical_navigation.py` |
| 检索证据评分、查询失败归因到哪一轮、反馈给教师的文本 | `experiments/qwen27b_rsi/clinical_evidence.py`、`targeted_guidance.py`（同目录） |
| 多轮采样、打分、反馈传递与参数更新 | `experiments/qwen27b_rsi/train_extractor.py`、`policy_worker.py`（同目录） |
| 冻结 / scalar / guided 三组与 RAW-RAG 的逐题比较 | `experiments/qwen27b_rsi/run_targeted_writer.py` |

## 原实验机与依赖边界

控制端是 Linux x86_64、Ubuntu 24.04.4 LTS。工作区为 `/mnt/mess/duxianghe/memos/medicalMemory`，其中 `QFAD-RSI/` 是本 Git 仓库。控制端使用 `MedMemoryBench/.venv/bin/python`（CPython 3.12.3）和工作区上一层的 `baselines/QAFD-RAG/.deps`。下表为该解释器加上此依赖路径后读取的版本。

| 环境 | 实验使用的版本 |
|---|---|
| 控制端 CPU | torch 2.12.1+cpu、transformers 5.22.2、numpy 1.26.4、igraph 0.11.9、pandas 2.2.3、pyarrow 19.0.1、tiktoken 0.14.0、openai 1.109.1、httpx 0.28.1 |
| GPU worker（已有实验记录） | 通过 `win → 2A100` 启动，在物理 GPU 1 上运行。记录的卡型为 NVIDIA A100 80GB PCIe。torch 2.13.0+cu129、transformers 5.16.1、accelerate 1.15.0、peft 0.21.3、bitsandbytes 0.50.2 |

根目录 `requirements.txt` 是上游 QFAD 通用依赖，其中 transformers 固定为 4.45.2。控制端版本和 GPU worker 版本来自两个不同环境。worker 的 Python 路径、基座权重路径、NF4、BF16、LoRA rank 8 / alpha 16 等配置见[现有 worker 环境记录](../research/qwen27b_rsi/env-spec.json)。仓库提供源码、Prompt、测试、方法说明和部分历史审计摘要。虚拟环境、SSH 配置、服务进程、凭据、完整实验数据、权重和 checkpoint 均由实验环境管理。

## 数据和服务实际在哪里

以下相对路径从 `medicalMemory/` 开始。尤其要区分工作区的 `results/` 与仓库里的 `QFAD-RSI/results/`。前者存已有实验输入，后者常作从仓库根目录启动时的运行输出，被 Git 忽略。部分代码还直接拼接目录名 `QFAD-RSI`，原实验机保留这个检出名称。

| 资源 | 位置或协议 | 用途 |
|---|---|---|
| 医学基准与适配脚本 | `MedMemoryBench/data/MedMemoryBench_EN_mixed/`、`scripts/qafd_medmemorybench_mixed.py`、`scripts/rawrag_medmemorybench_zh_mixed.py`、`scripts/medmemory_scoring.py` | 英文混合患者记录、时间可见性、问题与裁判调用 |
| 外部代码 | `MedMemoryBench` 基点 `d9c0d5a`，`../baselines/RAW-RAG` 基点 `9563919`，`../baselines/QAFD-RAG` 基点 `2a3a42c` | RAW-RAG 的 `methods/EverCore/evaluation/config/prompts.yaml` 提供回答模板。MedMemoryBench 与 RAW-RAG 检出含本地修改，基点 SHA 不能代表整个工作区 |
| 患者原文索引与初始检索轨迹 | `results/qafd_medmemorybench_en_mixed_gpt4omini_20261111/`、`results/qwen27b_rsi/initial_full/answers/` | 按患者读取原文块、向量、已交付的来源 ID |
| 校准与评测输入 | `results/medical_sage/clinical_calibration/summary.json`、`results/rawrag_medmemorybench_en_mixed_qwen38_20261111/`、`results/qfad_query_writer_rl_20261113/semantic/evaluation/`、`results/qwen27b_rsi/transfer_p34/raw_scores/` | 14 个临床校准用例、上下文预算与同题 RAW-RAG 分数。这里的 medical_sage 是历史数据目录名 |
| 分词与向量服务 | `results/qwen27b_rsi/tokenizers/`；控制端 `127.0.0.1:19085/v1` | Qwen3-Embedding-4B 生成向量，Qwen3-Reranker-4B 经 `/v1/rerank` 重排 |
| 冻结回答与裁判 | `127.0.0.1:19109/v1`，模型别名 `macaq`，同端口另有 `/tokenize`；裁判为 `gpt-6.1-sol` | 回答使用 Qwen3.8-27B-FP8。裁判从实验机环境读取 `EXPERIMENT_API_BASE_URL` 和 `EXPERIMENT_API_KEY`。历史基础 Reader 的 `QFAD_READER_URL` 默认指向 19107 |
| 策略与产物 | 2A100 的 `/home/administrator/du/memos/qwen27b_rsi/` | `pilot` 部署按内容哈希命名的 worker，checkpoint 写到 GPU 主机；控制端运行目录保存协议、轨迹、各组 checkpoint 路径与逐题结果 |

服务地址是代码约定，文档核对未进行在线健康检查。

## 在原实验机验证改动

在 `QFAD-RSI/` 下执行，使用已有控制端环境。下列命令只执行 CPU 测试。

```bash
export PYTHONPATH="$PWD/experiments:$PWD/src:$PWD/../../baselines/QAFD-RAG/.deps"
CUDA_VISIBLE_DEVICES='' ../MedMemoryBench/.venv/bin/python -m unittest discover -s experiments/qwen27b_rsi -p 'test_*.py'
```

完整实验顺序、GPU 阶段的副作用和命令见 [Target Writer 实验说明](../research/medical_sage/TARGETED_WRITER.md)。`prepare` 准备协议，`pilot` 训练 scalar 与 guided 并与冻结组配对评测，`full` 读取已有 checkpoint。旧联调协议（`prepare --smoke`）有 7 道训练题，pilot 有 14 道留出题。2026-11-16 起，默认清单按患者拆分为 482 道训练题、197 道验证题、189 道最终比较题。每个窗口的真实采样数由准备阶段的 `--group-size` 确定。验收分别看 `rl_gain`（guided 超过冻结）、`guidance_gain_over_scalar`（文字反馈的额外收益）和 `beats_raw`。当前交接已核验 CPU 测试与协议规模，尚无该协议超越 RAW-RAG 的已核验配对结果。
