#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=MedMemoryBench/.venv/bin/python
experiment_dir=QFAD-RSI/experiments/qwen27b_rsi
result_dir=results/qwen27b_rsi
"$python_bin" -u "$experiment_dir/generate_feedback_pair.py"
"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_feedback_fp8_original_full" \
  --state "$result_dir/writer_feedback_fp8_original/graph_state.json" \
  --paired-with "$result_dir/writer_factual_before_full" --concurrency 4
"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_feedback_fp8_answer_full" \
  --state "$result_dir/writer_feedback_fp8_answer/graph_state.json" \
  --paired-with "$result_dir/writer_feedback_fp8_original_full" --concurrency 4
"$python_bin" "$experiment_dir/compare_feedback_pair.py"
