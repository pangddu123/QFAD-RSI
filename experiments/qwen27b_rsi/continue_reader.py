"""Start the Reader phase after the current Writer process completes successfully."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--writer-pid', type=int, required=True)
    parser.add_argument('--writer-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', required=True)
    args = parser.parse_args()
    print('Waiting for Writer process', args.writer_pid, flush=True)
    while True:
        try:
            os.kill(args.writer_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)
    completion = json.loads((args.writer_run / 'complete.json').read_text())
    assert completion['worker_exit'] == 0, completion
    gpu = subprocess.run(['ssh', '-T', 'win', 'ssh 2A100 nvidia-smi --id=1 '
        '--query-gpu=memory.free --format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free_mib = int(gpu.stdout.strip())
    if free_mib < 40000:
        raise RuntimeError(f'Reader training needs 40000 MiB free on GPU1; available {free_mib}')
    print('Starting Reader with free GPU1 MiB', free_mib, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('train_reader.py')),
        '--out', str(args.out), '--state', str(args.writer_run / 'graph_state.json'),
        '--resume', f'/home/administrator/du/memos/qwen27b_rsi/{args.writer_run.name}/checkpoint.pt',
        '--episodes', *args.episodes]
    with (args.out / 'run.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (args.out / 'process.json').write_text(json.dumps({'pid': process.pid, 'command': command}, indent=2))
        print('Reader process', process.pid, flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f'Reader phase exited {status}; see {args.out / "run.log"}')


if __name__ == '__main__':
    main()
