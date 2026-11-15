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
graph_state=$result_dir/writer_greedy_after_selection/graph_state.json
checkpoint=$remote_dir/joint_writer_selection/checkpoint.pt

if [[ ! -f "$result_dir/reader_factual_before_full/complete.json" ]]; then
  "$python_bin" -u "$experiment_dir/evaluate_reader.py" \
    --out "$result_dir/reader_factual_before_full" --state "$graph_state" \
    --resume "$checkpoint" --concurrency 4
fi

mapfile -t episodes < <("$python_bin" - <<'PY'
import json
with open('results/qwen27b_rsi/reader_training_curriculum.json') as handle:
    for row in json.load(handle)['episodes']:
        print(row['episode'])
PY
)

start_index=$("$python_bin" - <<'PY'
import json
from pathlib import Path
directory = Path('results/qwen27b_rsi/reader_factual')
index = 0
while (directory / f'step{index}.update.json').exists():
    assert json.loads((directory / f'step{index}.update.json').read_text())['event'] == 'updated'
    index += 1
print(index)
PY
)
if (( start_index > 0 )); then
  checkpoint=$remote_dir/reader_factual/checkpoint.pt
fi

"$python_bin" -u "$experiment_dir/train_reader.py" \
  --out "$result_dir/reader_factual" --state "$graph_state" \
  --resume "$checkpoint" --start-index "$start_index" --update-epochs 4 --episodes "${episodes[@]}"

"$python_bin" -u "$experiment_dir/evaluate_reader.py" \
  --out "$result_dir/reader_factual_after_full" --state "$graph_state" \
  --resume "$remote_dir/reader_factual/checkpoint.pt" --concurrency 4 \
  --paired-with "$result_dir/reader_factual_before_full"

"$python_bin" "$experiment_dir/compare_reader.py" \
  --before "$result_dir/reader_factual_before_full" \
  --after "$result_dir/reader_factual_after_full" \
  --out "$result_dir/reader_factual_comparison.json"
