#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mess/duxianghe/memos/medicalMemory
result_dir=results/qwen27b_rsi
while [[ ! -f "$result_dir/writer_factual/step5.update.json" ]]; do
  systemctl --user is-active --quiet qwen27b-factual-writer-resume.service
  sleep 5
done
python3 - <<'PY'
import json
from pathlib import Path
p=Path('results/qwen27b_rsi/writer_factual')
assert len(list(p.glob('step*.update.json')))==6
assert json.loads((p/'step5.update.json').read_text())['step']==26
PY
systemctl --user stop qwen27b-factual-writer-resume.service
ssh -T win 'ssh 2A100 bash -s' <<'REMOTE'
set -e
/home/administrator/du/memskill-determinacy-20261014/venv/bin/python - <<'PY'
import json, os, shutil, signal, subprocess, time
from pathlib import Path
pid=424387
proc=Path(f'/proc/{pid}')
if proc.exists():
    args=(proc/'cmdline').read_bytes().decode().split('\0')
    assert '--out' in args and args[args.index('--out')+1]=='/home/administrator/du/memos/qwen27b_rsi/writer_factual'
    os.kill(pid,signal.SIGTERM)
    for _ in range(40):
        if not proc.exists(): break
        time.sleep(.5)
    assert not proc.exists(), 'Writer worker has not exited'
import torch
source=Path('/home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint.pt')
checkpoint=torch.load(source,map_location='cpu',weights_only=False)
assert checkpoint['step']==26
target=source.with_name('checkpoint_step26.pt')
assert not target.exists()
shutil.copyfile(source,target)
result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(row.split(',')[0]):int(row.split(',')[1]) for row in result.splitlines()}
assert free[1]>=35000, free
print(json.dumps({'checkpoint':str(target),'step':26,'free_memory_mib':free}))
PY
REMOTE
python3 - <<'PY'
import json
from pathlib import Path
p=Path('results/qwen27b_rsi/writer_factual')
config=json.loads((p/'configuration.json').read_text())
(p/'stopped_at_checkpoint.json').write_text(json.dumps({
    'completed_groups':6,'optimizer_steps':24,'checkpoint_step':26,
    'completed_episodes':config['episodes'][:6],
    'checkpoint':'/home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint_step26.pt',
    'evaluation':'Both policies generate graphs from the same 21 feedback episodes; evaluate all 197 questions',
    'scope':'Checkpoint evaluation after six completed training groups; the remaining planned groups were not executed'},indent=2))
PY
set -a
source ~/.config/environment.d/60-experiment-judge.conf
set +a
export NO_PROXY=localhost,127.0.0.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin=MedMemoryBench/.venv/bin/python
experiment_dir=QFAD-RSI/experiments/qwen27b_rsi
mapfile -t episodes < <("$python_bin" - <<'PY'
import json
from pathlib import Path
print('\n'.join(json.loads(Path('results/qwen27b_rsi/writer_factual_before/configuration.json').read_text())['episodes']))
PY
)
"$python_bin" -u "$experiment_dir/generate_writer_graph.py" \
  --out "$result_dir/writer_factual_checkpoint6" --reader-run "$result_dir/reader_factual_before_full" \
  --state "$result_dir/writer_greedy_after_selection/graph_state.json" \
  --resume /home/administrator/du/memos/qwen27b_rsi/writer_factual/checkpoint_step26.pt \
  --episodes "${episodes[@]}"
"$python_bin" -u "$experiment_dir/evaluate_graph.py" \
  --out "$result_dir/writer_factual_checkpoint6_full" --state "$result_dir/writer_factual_checkpoint6/graph_state.json" \
  --paired-with "$result_dir/writer_factual_before_full" --concurrency 2
"$python_bin" "$experiment_dir/compare_writer_graphs.py" \
  --before "$result_dir/writer_factual_before_full" --after "$result_dir/writer_factual_checkpoint6_full" \
  --writer-before "$result_dir/writer_factual_before" --writer-after "$result_dir/writer_factual_checkpoint6" \
  --out "$result_dir/writer_factual_checkpoint6_comparison.json"
"$python_bin" "$experiment_dir/summarize_writer_budget.py" \
  --comparison "$result_dir/writer_factual_checkpoint6_comparison.json" \
  --out "$result_dir/writer_factual_checkpoint6_budget.json"
