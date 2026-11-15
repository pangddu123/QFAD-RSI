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
remote_dir=/home/administrator/du/memos/qwen27b_rsi
initial_graph=$result_dir/writer_greedy_after_selection/graph_state.json
reader_run=$result_dir/reader_factual_before_full
writer_checkpoint=$remote_dir/reader_factual/checkpoint.pt
resume_index=${1:-0}
training_checkpoint=$writer_checkpoint
if (( resume_index > 0 )); then
  training_checkpoint=$remote_dir/writer_factual/checkpoint.pt
fi

"$python_bin" - "$resume_index" <<'PY'
import json
import sys
from pathlib import Path
root = Path('results/qwen27b_rsi')
assert json.loads((root / 'reader_factual_before_full/complete.json').read_text()) == {
    'queries': 197, 'worker_exit': 0}
assert json.loads((root / 'reader_factual/step1.update.json').read_text())['step'] == 20
resume_index = int(sys.argv[1])
if resume_index:
    assert len(list((root / 'writer_factual').glob('step*.update.json'))) == resume_index
else:
    assert not (root / 'writer_factual').exists(), 'Writer training already has an output directory'
PY

mapfile -t episodes < <("$python_bin" - <<'PY'
import json
with open('results/qwen27b_rsi/factual_query_curriculum.json') as handle:
    for row in json.load(handle)['episodes']:
        print(row['episode'])
PY
)

"$python_bin" "$experiment_dir/writer_feedback.py" --reader-run "$reader_run" \
  --out "$result_dir/reader_factual_writer_feedback.json"

if [[ ! -f "$result_dir/writer_factual_before/complete.json" ]]; then
  "$python_bin" -u "$experiment_dir/generate_writer_graph.py" \
    --out "$result_dir/writer_factual_before" --reader-run "$reader_run" \
    --state "$initial_graph" --resume "$writer_checkpoint" --episodes "${episodes[@]}"
fi

"$python_bin" "$experiment_dir/source_verdicts.py" --generation "$result_dir/writer_factual_before"

baseline_evaluation_pid=
if [[ ! -f "$result_dir/writer_factual_before_full/summary.json" ]]; then
  "$python_bin" -u "$experiment_dir/evaluate_graph.py" \
    --out "$result_dir/writer_factual_before_full" \
    --state "$result_dir/writer_factual_before/graph_state.json" \
    --paired-with "$result_dir/greedy_writer_full" --concurrency 4 &
  baseline_evaluation_pid=$!
fi

"$python_bin" -u "$experiment_dir/train_writer.py" \
  --out "$result_dir/writer_factual" --state "$initial_graph" \
  --resume "$training_checkpoint" --start-index "$resume_index" \
  --evidence-feedback "$result_dir/reader_factual_writer_feedback.json" \
  --replay-run "$result_dir/writer_factual_before" \
  --update-epochs 4 --episodes "${episodes[@]}"

"$python_bin" -u "$experiment_dir/generate_writer_graph.py" \
  --out "$result_dir/writer_factual_after" --reader-run "$reader_run" \
  --state "$initial_graph" --resume "$remote_dir/writer_factual/checkpoint.pt" \
  --episodes "${episodes[@]}"

if [[ -n "$baseline_evaluation_pid" ]]; then
  wait "$baseline_evaluation_pid"
fi

"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_factual_after_full" \
  --state "$result_dir/writer_factual_after/graph_state.json" \
  --paired-with "$result_dir/writer_factual_before_full" --concurrency 4

"$python_bin" "$experiment_dir/compare_writer_graphs.py" \
  --before "$result_dir/writer_factual_before_full" --after "$result_dir/writer_factual_after_full" \
  --writer-before "$result_dir/writer_factual_before" --writer-after "$result_dir/writer_factual_after" \
  --out "$result_dir/writer_factual_comparison.json"

"$python_bin" "$experiment_dir/summarize_writer_budget.py" \
  --comparison "$result_dir/writer_factual_comparison.json" \
  --out "$result_dir/writer_factual_budget.json"
