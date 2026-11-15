#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec MedMemoryBench/.venv/bin/python -u QFAD-RSI/experiments/qwen27b_rsi/probe_current_reader_feedback.py
