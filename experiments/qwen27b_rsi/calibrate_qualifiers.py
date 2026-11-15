"""Check preservation of firsthand qualifiers when clinician summaries differ."""
from copy import deepcopy
import json
from pathlib import Path

import calibrate_grounding as grounding

def main():
    root = grounding.ROOT
    out = root / 'results/qwen27b_rsi/grounding_qualifiers'
    rows = json.loads((root / 'results/qwen27b_rsi/atomic_writer_pilot/step0_sample2.reward.json').read_text())
    action = rows['action']
    docs = json.loads((root / 'results/qafd_medmemorybench_en_mixed_gpt4omini_20261111/persona_01/qafd/openie_results_ner_gpt-4o-mini.json').read_text())['docs']
    text = next(d['passage'] for d in docs if d['idx'] == action['source_id'])
    source = {'source_id': action['source_id'], 'text': text}
    wrong = {**action['fact'], 'citations': action['citations']}
    correct = deepcopy(wrong)
    correct.update(object='dry mouth and thirst relieved only briefly by water, then quickly recurring',
                   value='temporary partial relief', certainty='reported')
    correct['citations'] = [correct['citations'][0]]
    interpretation = {k: '' for k in wrong if k != 'citations'}
    interpretation.update(subject='doctor', relation='describes the patient symptoms as',
        object='persistent dry mouth and thirst not relieved by drinking', polarity='affirmed',
        certainty='reported', record_time='2024-03-01', citations=[wrong['citations'][-1]])
    cases = [('patient_qualifiers_erased', wrong, False), ('patient_qualifiers_preserved', correct, True),
             ('clinician_summary_attributed', interpretation, True)]
    grounding.write_json(out / 'configuration.json', {'prompt': grounding.PROMPT,
        'origin': 'Observed Writer action and controlled attribution/qualification variants',
        'clinical_expert_validation': False})
    judge = grounding.JudgeClient('gpt-6.1-sol')
    results = []
    for name, fact, expected in cases:
        verdict = grounding.check_source(fact, source, judge)
        row = {'case_id': name, 'fact': fact, 'source': source,
               'expected_supported': expected, 'verdict': verdict}
        grounding.write_json(out / f'{name}.json', row)
        results.append({'case_id': name, 'expected': expected, 'observed': verdict['supported']})
        print(name, expected, verdict, flush=True)
    grounding.write_json(out / 'summary.json', {'cases': results,
        'all_matched': all(r['expected'] == r['observed'] for r in results)})


if __name__ == '__main__':
    main()
