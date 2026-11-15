"""Share identical source judgments across paired Writer conditions."""
import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import sqlite3
import threading

from calibrate_grounding import PROMPT

ROOT = Path(__file__).resolve().parents[3] / 'results/qwen27b_rsi'


def verification_key(action, state, model):
    existing = {action[key]: state['facts'][action[key]] for key in ('target_id', 'predecessor_id')
                if key in action and action['operation'] != 'ADD_FACT'}
    return json.dumps({'model': model, 'source_prompt': PROMPT, 'action': action,
                       'existing_facts': existing}, ensure_ascii=False, sort_keys=True)


class SourceVerdicts:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS verdicts (request TEXT PRIMARY KEY, verdict TEXT NOT NULL)')
        self.db.commit()
        self.lock = threading.Lock()
        self.pending = {}

    def seed(self, key, verdict):
        with self.lock:
            prior = self.db.execute('SELECT verdict FROM verdicts WHERE request = ?', (key,)).fetchone()
            if prior:
                assert json.loads(prior[0])['supported'] == verdict['supported'], 'Conflicting baseline source judgments'
            else:
                self.db.execute('INSERT INTO verdicts VALUES (?, ?)', (key, json.dumps(verdict, ensure_ascii=False)))
                self.db.commit()

    def check(self, key, evaluate):
        with self.lock:
            prior = self.db.execute('SELECT verdict FROM verdicts WHERE request = ?', (key,)).fetchone()
            if prior:
                return {**json.loads(prior[0]), 'paired_verdict_reuse': True}
            pending = self.pending.get(key)
            owner = pending is None
            if owner:
                pending = self.pending[key] = Future()
        if not owner:
            return {**pending.result(), 'paired_verdict_reuse': True}
        try:
            verdict = evaluate()
            self.seed(key, verdict)
            pending.set_result(verdict)
            return verdict
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self.lock:
                del self.pending[key]


VERDICTS = SourceVerdicts(ROOT / 'writer_source_verdicts.sqlite')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((args.generation / 'complete.json').read_text())['worker_exit'] == 0
    count = 0
    for path in sorted(args.generation.glob('step*.result.json')):
        row = json.loads(path.read_text())
        if row['action'] is None or 'usage' not in row['support']:
            continue
        state = json.loads(path.with_name(path.name.replace('.result.', '.before.')).read_text())
        VERDICTS.seed(verification_key(row['action'], state, 'gpt-6.1-sol'), row['support'])
        count += 1
    print(json.dumps({'seeded_source_verdicts': count, 'generation': str(args.generation)}))


if __name__ == '__main__':
    main()
