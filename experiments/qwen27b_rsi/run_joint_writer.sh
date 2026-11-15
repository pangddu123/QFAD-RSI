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
reader_checkpoint=$remote_dir/reader_raw_reference/checkpoint.pt

"$python_bin" -u "$experiment_dir/evaluate_writer.py" \
  --out "$result_dir/writer_sampled_before_selection" \
  --training-run "$result_dir/evidence_feedback_writer" \
  --resume "$reader_checkpoint" --reader-checkpoint "$reader_checkpoint" \
  --samples 8 --seed 0

"$python_bin" -u "$experiment_dir/train_writer.py" \
  --out "$result_dir/joint_writer_selection" \
  --state "$result_dir/autonomous_feedback_writer/graph_state.json" \
  --resume "$reader_checkpoint" --reader-checkpoint "$reader_checkpoint" \
  --replay-run "$result_dir/evidence_feedback_writer" \
  --evidence-feedback "$result_dir/reader_raw_reference/reader_feedback.json" \
  --episodes p2_session_50_eem_1 p2_session_50_eem_1 p2_session_50_eem_1 p2_session_50_eem_1

"$python_bin" -u "$experiment_dir/evaluate_writer.py" \
  --out "$result_dir/writer_sampled_after_selection" \
  --training-run "$result_dir/evidence_feedback_writer" \
  --resume "$remote_dir/joint_writer_selection/checkpoint.pt" \
  --reader-checkpoint "$reader_checkpoint" \
  --paired-with "$result_dir/writer_sampled_before_selection" \
  --samples 8 --seed 0

"$python_bin" -u "$experiment_dir/evaluate_writer.py" \
  --out "$result_dir/writer_greedy_after_selection" \
  --training-run "$result_dir/evidence_feedback_writer" \
  --resume "$remote_dir/joint_writer_selection/checkpoint.pt" \
  --reader-checkpoint "$reader_checkpoint"

"$python_bin" "$experiment_dir/compare_writer.py" \
  --before "$result_dir/writer_sampled_before_selection" \
  --after "$result_dir/writer_sampled_after_selection" \
  --out "$result_dir/joint_selection_comparison.json"
