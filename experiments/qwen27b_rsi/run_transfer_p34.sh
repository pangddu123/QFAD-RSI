#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=MedMemoryBench/.venv/bin/python
code=QFAD-RSI/experiments/qwen27b_rsi
results=results/qwen27b_rsi
cohort=$results/transfer_p34
"$python_bin" -u "$code/prepare_transfer.py" --out "$cohort" --patients 3 4
mapfile -t episodes < <("$python_bin" - <<'PY'
import json
from pathlib import Path
print('\n'.join(json.loads(Path('results/qwen27b_rsi/transfer_p34/configuration.json').read_text())['episodes']))
PY
)
"$python_bin" -u "$code/account_retrieval_budget.py" --patients 3 4 \
  --state "$cohort/initial_graph.json" --out "$cohort/retrieval_budget.json" > "$cohort/retrieval_budget.log" 2>&1 &
budget_pid=$!
"$python_bin" -u "$code/evaluate_graph.py" --patients 3 4 --out "$results/transfer_p34_empty_full" \
  --state "$cohort/initial_graph.json" --raw-scores "$cohort/raw_scores" --concurrency 6 \
  > "$cohort/empty_full.log" 2>&1 &
empty_pid=$!
"$python_bin" -u "$code/evaluate_reader.py" --patients 3 4 --out "$results/transfer_p34_reader" \
  --state "$cohort/initial_graph.json" --raw-scores "$cohort/raw_scores" --raw-usage "$cohort/raw_usage.json" \
  --resume /home/administrator/du/memos/qwen27b_rsi/joint_writer_selection/checkpoint.pt \
  --episodes "${episodes[@]}" --concurrency 4 > "$cohort/reader.log" 2>&1
"$python_bin" -u "$code/generate_writer_graph.py" --out "$results/transfer_p34_writer_before" \
  --reader-run "$results/transfer_p34_reader" --state "$cohort/initial_graph.json" --hide-answer-labels \
  --resume /home/administrator/du/memos/qwen27b_rsi/reader_factual/checkpoint.pt \
  --episodes "${episodes[@]}" > "$cohort/writer_before.log" 2>&1
"$python_bin" -u "$code/evaluate_graph.py" --patients 3 4 --out "$results/transfer_p34_before_full" \
  --state "$results/transfer_p34_writer_before/graph_state.json" --raw-scores "$cohort/raw_scores" \
  --paired-with "$results/transfer_p34_empty_full" --concurrency 4 > "$cohort/before_full.log" 2>&1 &
before_pid=$!
"$python_bin" -u "$code/generate_writer_graph.py" --out "$results/transfer_p34_writer_after" \
  --reader-run "$results/transfer_p34_reader" --state "$cohort/initial_graph.json" --hide-answer-labels \
  --resume /home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint_step26.pt \
  --episodes "${episodes[@]}" > "$cohort/writer_after.log" 2>&1
wait "$empty_pid"
"$python_bin" -u "$code/evaluate_graph.py" --patients 3 4 --out "$results/transfer_p34_after_full" \
  --state "$results/transfer_p34_writer_after/graph_state.json" --raw-scores "$cohort/raw_scores" \
  --paired-with "$results/transfer_p34_before_full" --concurrency 6 > "$cohort/after_full.log" 2>&1 &
after_pid=$!
wait "$before_pid"
wait "$after_pid"
wait "$budget_pid"
"$python_bin" "$code/compare_transfer.py" --cohort "$cohort"
