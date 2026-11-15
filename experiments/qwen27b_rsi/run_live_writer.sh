#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
checkpoint=${1:?Pass the remote initial Qwen27B policy checkpoint}
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=MedMemoryBench/.venv/bin/python
experiment_dir=QFAD-RSI/experiments/qwen27b_rsi
result_dir=results/qwen27b_rsi
initial_graph=$result_dir/writer_greedy_after_selection/graph_state.json
reader_run=$result_dir/reader_factual_before_full
test ! -e "$result_dir/writer_live"
mapfile -t episodes < <("$python_bin" - <<'PY'
import json
from pathlib import Path
config=json.loads(Path('results/qwen27b_rsi/writer_factual_before/configuration.json').read_text())
print('\n'.join(config['episodes']))
PY
)
"$python_bin" "$experiment_dir/writer_feedback.py" --reader-run "$reader_run" \
  --include-generated-answer --out "$result_dir/reader_answer_writer_feedback.json"
if [[ -e "$result_dir/writer_live_before" ]]; then
  "$python_bin" - "$checkpoint" <<'PY'
import json, sys
from pathlib import Path
p=Path('results/qwen27b_rsi/writer_live_before')
assert json.loads((p/'complete.json').read_text())=={'episodes':21,'worker_exit':0}
config=json.loads((p/'configuration.json').read_text())
assert config['resume']==sys.argv[1] and config['feedback_protocol']=='reader-generated-answer' and config['current_reader_feedback'] is True
assert config['episodes']==json.loads(Path('results/qwen27b_rsi/writer_factual_before/configuration.json').read_text())['episodes']
PY
else
  "$python_bin" -u "$experiment_dir/generate_writer_graph.py" \
    --out "$result_dir/writer_live_before" --reader-run "$reader_run" \
    --state "$initial_graph" --resume "$checkpoint" --include-reader-answer --current-reader-feedback --episodes "${episodes[@]}"
fi
"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_live_before_full" --state "$result_dir/writer_live_before/graph_state.json" \
  --paired-with "$result_dir/writer_factual_checkpoint6_full" --concurrency 4 &
baseline_pid=$!
"$python_bin" -u "$experiment_dir/train_writer.py" \
  --out "$result_dir/writer_live" --state "$initial_graph" --resume "$checkpoint" \
  --evidence-feedback "$result_dir/reader_answer_writer_feedback.json" \
  --replay-run "$result_dir/writer_live_before" --aligned-reader-feedback --current-reader-feedback \
  --rejected-action-reward no-change --retention-run "$result_dir/greedy_writer_full" \
  --update-epochs 1 --episodes "${episodes[@]}"
"$python_bin" -u "$experiment_dir/generate_writer_graph.py" \
  --out "$result_dir/writer_live_after" --reader-run "$reader_run" \
  --state "$initial_graph" --resume /home/administrator/du/memos/qwen27b_rsi/writer_live/checkpoint.pt \
  --include-reader-answer --current-reader-feedback --episodes "${episodes[@]}"
wait "$baseline_pid"
"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_live_after_full" --state "$result_dir/writer_live_after/graph_state.json" \
  --paired-with "$result_dir/writer_live_before_full" --concurrency 4
"$python_bin" "$experiment_dir/compare_writer_graphs.py" \
  --before "$result_dir/writer_live_before_full" --after "$result_dir/writer_live_after_full" \
  --writer-before "$result_dir/writer_live_before" --writer-after "$result_dir/writer_live_after" \
  --out "$result_dir/writer_live_comparison.json"
"$python_bin" "$experiment_dir/summarize_writer_budget.py" \
  --comparison "$result_dir/writer_live_comparison.json" --out "$result_dir/writer_live_budget.json"
