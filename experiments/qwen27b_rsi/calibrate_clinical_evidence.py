"""Calibrate clinical coverage on original evidence and controlled omissions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from clinical_evidence import score_evidence
from medical_navigation import NavigationGraph, base
from medmemory_scoring import JudgeClient

OUT = base.ROOT / 'results/medical_sage/clinical_calibration'


def cases():
    graph = NavigationGraph(base.Patient(1, store_names=('chunk',)))
    prior = {r['id']: r for r in base.read(base.ROOT / 'results/qwen27b_rsi/grounding_calibration_citations/cases.json')}
    records = base.read(base.ROOT / 'MedMemoryBench/data/MedMemoryBench_EN_mixed/persona_1/eval/generated_dialogues.json')['sessions']
    stop_session = next(s for s in records if s['session_id'] == 6)
    def locate(quote):
        source = next(s for s in graph.sources.values() if quote in s['text'])
        return {'source_id': source['source_id'], 'quote': quote}
    executed = locate(stop_session['messages'][2]['content'].split('\n\n')[0])
    requested = locate(stop_session['messages'][0]['content'].split('\n\n')[1])
    metformin = prior['metformin_supported']['source']
    metformin_quote = next(line for line in metformin['text'].splitlines() if line.startswith('Patient:'))
    metformin_anchor = locate(metformin['text'].splitlines()[0] + '\n' + metformin_quote)
    no_date = locate(metformin_quote)
    old_hba1c_line = next(line for line in metformin['text'].splitlines() if '9.2%' in line)
    old_hba1c = locate(old_hba1c_line)
    followup = locate(prior['hba1c_supported']['fact']['citations'][0]['quote'])
    unmeasured = locate(prior['not_measured_supported']['fact']['citations'][0]['quote'])
    question_only = locate(unmeasured['quote'].split('\nPatient:')[0])
    uncertainty = locate(prior['uncertainty_supported']['fact']['citations'][0]['quote'])
    body_only = locate('Your body' + uncertainty['quote'].split('Your body', 1)[1])
    rows = []
    def add(name, question, descriptions, evidence, expected):
        documents = [f'Source {x["source_id"]}:\n{x["quote"]}' for x in evidence]
        rows.append({'id': name, 'question': question,
            'requirements': {'bundles': [{'conditions': [{'description': d} for d in descriptions]}]},
            'evidence': {'documents': documents, 'source_ids': list(dict.fromkeys(x['source_id'] for x in evidence))},
            'original_fragments': evidence, 'expected_coverage': expected})
    stop = ['The patient has actually stopped empagliflozin, rather than only receiving a recommendation to stop.']
    add('stop_execution', 'What is the current empagliflozin status?', stop, [executed], 1.)
    add('stop_advice_only', 'What is the current empagliflozin status?', stop, [requested], 0.)
    add('stop_repeated', 'What is the current empagliflozin status?', stop, [executed, executed], 1.)
    regimen = ['The patient reports using metformin plus a DPP-4 inhibitor.',
               'The supplied record establishing this report is dated January 6, 2024.']
    add('regimen_dated', 'What regimen was reported in the January 6 record?', regimen, [metformin_anchor], 1.)
    add('regimen_missing_date', 'What regimen was reported in the January 6 record?', regimen, [no_date], .5)
    add('regimen_other_date', 'What regimen was reported in the January 6 record?', regimen, [followup], .5)
    trend = ['The earlier HbA1c measurement was 9.2%.',
             'The February follow-up HbA1c measurement is 8.1%, lower than the earlier measurement.']
    add('trend_complete', 'How did HbA1c change by the February follow-up?', trend, [followup], 1.)
    add('trend_baseline_only', 'How did HbA1c change by the February follow-up?', trend, [old_hba1c], .5)
    add('trend_repeated_baseline', 'How did HbA1c change by the February follow-up?', trend, [old_hba1c, old_hba1c], .5)
    observation = ['The patient explicitly reports not performing the planned fasting or post-meal glucose measurements over the last two days.']
    add('unmeasured_report', 'Were the planned measurements performed?', observation, [unmeasured], 1.)
    add('unmeasured_question_only', 'Were the planned measurements performed?', observation, [question_only], 0.)
    tentative = ['The clinician considers poor glucose control possible, requires tests for confirmation, and does not exclude other explanations.']
    add('uncertainty_preserved', 'How certain is the assessment of poor glucose control?', tentative, [uncertainty], 1.)
    add('uncertainty_context_missing', 'How certain is the assessment of poor glucose control?', tentative, [body_only], 0.)
    add('missing_evidence', 'What is the current empagliflozin status?', stop, [], 0.)
    for row in rows:
        for fragment in row['original_fragments']:
            assert fragment['quote'] in graph.sources[fragment['source_id']]['text']
    return rows


def main():
    rows = cases()
    if (OUT / 'cases.json').exists():
        assert base.read(OUT / 'cases.json') == rows, 'Use a separate run for changed calibration cases'
    base.write_json(OUT / 'cases.json', rows)
    base.write_json(OUT / 'configuration.json', {
        'judge': 'gpt-6.1-sol', 'protocol': 'responses', 'reasoning_effort': 'high', 'concurrency': 4,
        'labels': 'Manually specified expected coverage; original text is never rewritten',
        'scope': 'Controlled reward calibration, not benchmark QA or clinical validation',
    })
    judge = JudgeClient('gpt-6.1-sol')
    def evaluate(row):
        saved = OUT / 'verdicts' / f'{row["id"]}.json'
        if saved.exists():
            return base.read(saved)
        result = score_evidence(row['question'], row['requirements'], row['evidence'], judge)
        result.update(id=row['id'], expected_coverage=row['expected_coverage'])
        base.write_json(OUT / 'verdicts' / f'{row["id"]}.json', result)
        print(row['id'], result['coverage'], row['expected_coverage'], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, rows))
    failures = [r['id'] for r in results if r['coverage'] != r['expected_coverage']]
    base.write_json(OUT / 'summary.json', {'cases': len(results), 'failures': failures,
        'judge_tokens': sum(r['usage']['total_tokens'] for r in results),
        'scope': 'Controlled clinical evidence coverage calibration'})
    if failures:
        raise RuntimeError(f'Clinical calibration mismatches: {failures}')


if __name__ == '__main__':
    main()
