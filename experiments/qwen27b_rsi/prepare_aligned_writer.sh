#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
test ! -e results/qwen27b_rsi/writer_aligned_before
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
MedMemoryBench/.venv/bin/python - <<'PY'
import json, subprocess, sys
from pathlib import Path
root=Path('results/qwen27b_rsi')
assert json.loads((root/'writer_factual_checkpoint6/complete.json').read_text())=={'episodes':21,'worker_exit':0}
config=json.loads((root/'writer_factual_before/configuration.json').read_text())
subprocess.run([sys.executable,'-u','QFAD-RSI/experiments/qwen27b_rsi/generate_writer_graph.py',
    '--out',str(root/'writer_aligned_before'),
    '--reader-run',str(root/'reader_factual_before_full'),
    '--state',str(root/'writer_greedy_after_selection/graph_state.json'),
    '--resume','/home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint_step26.pt',
    '--include-reader-answer','--episodes',*config['episodes']],check=True)
PY
