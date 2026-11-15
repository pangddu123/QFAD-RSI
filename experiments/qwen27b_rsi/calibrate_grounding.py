"""Calibrate the source gate on explicit facts and controlled field errors."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from medmemory_scoring import JudgeClient, write_json

OUT = ROOT / 'results/qwen27b_rsi/grounding_calibration_citations'
PROMPT = '''Check one proposed medical-memory fact against its cited original dialogue.
Return JSON with supported (boolean), error_fields (list of field names), and reason (brief).
Every nonempty factual field must be supported: subject, relation, object, value, unit,
frequency, polarity, certainty, status, event_time and record_time. A relative event_time
may be preserved verbatim. record_time is the dialogue header, not an inferred event date.
The patient speaker may be describing a relative: preserve the actual fact subject.
A suggestion is not a completed action; a suspected finding is not a confirmation;
a historical medication exposure does not establish current use. Preserve negation.
The cited quotes must substantiate the fact. Other source context can resolve speaker,
pronouns and dates, but cannot substitute for a missing supporting quote. Judge only
source support, not medical plausibility or whether a downstream answer would be correct.
Do not infer diagnoses, doses, event dates or status transitions not established here.
Preserve scope and degree: partial or temporary relief is not absence of relief.
For a patient-reported symptom, use the patient's qualified account; a clinician's stronger
summary cannot erase its qualifications or be attributed as the patient's own report.
A clinician's interpretation may be stored as that clinician's interpretation with its speaker retained.'''


def dataset():
    file = ROOT / 'results/qafd_medmemorybench_en_mixed_gpt4omini_20261111/persona_01/qafd/openie_results_ner_gpt-4o-mini.json'
    sources = {d['idx']: d['passage'] for d in json.loads(file.read_text())['docs']}
    cases = []

    def fact(case, source, needle, fields, errors):
        text = sources[source]
        quote = next(p for p in text.split('\n\n') if needle in p)
        found = re.match(r'\[Session time: (\d{4}-\d{2}-\d{2})\]', text)
        record = {k: '' for k in ('subject', 'relation', 'object', 'value', 'unit', 'frequency',
                                  'polarity', 'certainty', 'status', 'event_time', 'record_time')}
        record.update(polarity='affirmed', certainty='reported', record_time=found.group(1) if found else '')
        record.update(fields)
        record['citations'] = [{'source_id': source, 'quote': quote}]
        cases.append({'id': case + '_supported', 'expected_supported': True, 'error_type': None,
                      'source': {'source_id': source, 'text': text}, 'fact': record})
        for kind, replacement in errors.items():
            altered = deepcopy(record)
            altered.update(replacement)
            cases.append({'id': case + '_' + kind, 'expected_supported': False, 'error_type': kind,
                          'source': {'source_id': source, 'text': text}, 'fact': altered})
        return record

    dose_source = 'chunk-3b6b49301012484eb94a144d137bb130'
    dose = fact('metformin', dose_source, 'metformin 1500 mg',
        dict(subject='patient', relation='takes', object='metformin', value='1500', unit='mg',
             frequency='per day', status='current', event_time='today'),
        {'dose': {'value': '750'}, 'unit': {'unit': 'g'},
         'frequency': {'frequency': 'twice per day'}, 'status': {'status': 'stopped'}})
    allergy = fact('allergy', dose_source, 'cefuroxime',
        dict(subject='patient', relation='had an allergic reaction to', object='cefuroxime',
             status='history', event_time='before'),
        {'negation': {'polarity': 'negated'}, 'subject': {'subject': 'patient father'}})
    wrong_quote = deepcopy(dose)
    wrong_quote['citations'] = [{'source_id': dose_source, 'quote': 'Patient:'}]
    assert 'Patient:' in sources[dose_source]
    cases.append({'id': 'metformin_unsupported_citation', 'expected_supported': False,
        'error_type': 'citation_support', 'source': {'source_id': dose_source, 'text': sources[dose_source]},
        'fact': wrong_quote})
    broken_link = deepcopy(dose)
    broken_link['citations'][0]['source_id'] = 'absent-source'
    cases.append({'id': 'metformin_missing_source', 'expected_supported': False,
        'error_type': 'source_link', 'source': {'source_id': dose_source, 'text': sources[dose_source]},
        'fact': broken_link})
    fact('hba1c', 'chunk-159b311bbe7299e8b3feead6bbc2d611', '8.1%',
        dict(subject='patient', relation='has HbA1c measurement', object='HbA1c', value='8.1',
             unit='%', status='current', event_time='this month'),
        {'value': {'value': '9.2'}, 'event_time': {'event_time': '2024-03'},
         'record_time': {'record_time': '2024-03-05'}})
    fact('stopped_drug', 'chunk-5f7e01ce2f3948cd54430ee693845c38', 'stopped empagliflozin',
        dict(subject='patient', relation='stopped taking', object='empagliflozin', status='stopped', event_time='before'),
        {'status': {'relation': 'currently takes', 'status': 'current'},
         'negation': {'polarity': 'negated'}})
    fact('uncertainty', 'chunk-6b0e35c0466f53d54f7587de8319350c', 'possibly',
        dict(subject='doctor', relation='suspects', object='the patient blood glucose has not been well controlled',
             certainty='suspected', status='unconfirmed'),
        {'certainty': {'relation': 'confirms', 'certainty': 'confirmed', 'status': 'confirmed'}})
    fact('relative', 'chunk-e8f6970f3888aac17ee57ee0ac326b12', 'valsartan 80 mg',
        dict(subject='patient father', relation='takes', object='valsartan', value='80', unit='mg',
             frequency='in the morning', status='current'),
        {'subject': {'subject': 'patient'}, 'dose': {'value': '160'},
         'drug': {'object': 'amlodipine'}})
    fact('not_measured', 'chunk-888d8ec89ba68eb20e3f726c6583a1b0', 'I haven’t really measured',
        dict(subject='patient', relation='measured fasting and post-meal blood glucose according to plan',
             object='blood glucose', polarity='negated', status='not_completed', event_time='these past two days'),
        {'negation': {'polarity': 'affirmed', 'status': 'completed'}})
    recommendation = fact('recommendation', 'chunk-78dbdfb7342bf7d813775096280c36bf', 'taking one over‑the‑counter antihistamine tablet',
        dict(subject='doctor', relation='recommends', object='an over-the-counter antihistamine tablet for itching affecting sleep',
             certainty='recommended', status='recommendation'),
        {'completion': {'subject': 'patient', 'relation': 'took', 'certainty': 'reported', 'status': 'completed'}})
    source_id = recommendation['citations'][0]['source_id']
    partial = deepcopy(recommendation)
    partial['citations'][0]['quote'] = next(p for p in sources[source_id].split('\n\n')
                                          if 'short course of over‑the‑counter antihistamines' in p)
    cases.append({'id': 'recommendation_incomplete_citation', 'expected_supported': False,
                  'error_type': 'citation_support',
                  'source': {'source_id': source_id, 'text': sources[source_id]}, 'fact': partial})
    return cases


def check_source(fact, source, judge):
    if not fact['citations'] or any(c['source_id'] != source['source_id'] or
            not c['quote'] or c['quote'] not in source['text'] for c in fact['citations']):
        return {'supported': False, 'error_fields': ['citations'], 'reason': 'Missing source or non-verbatim citation',
                'validator': 'exact_source_link'}
    response = judge.chat(messages=[{'role': 'system', 'content': PROMPT},
        {'role': 'user', 'content': json.dumps({'source': source, 'fact': fact}, ensure_ascii=False)}],
        max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
    result = json.loads(response.content)
    assert type(result['supported']) is bool and isinstance(result['error_fields'], list)
    return {**result, 'validator': 'source_entailment', 'usage': response.usage}


def main():
    cases = dataset()
    write_json(OUT / 'cases.json', cases)
    write_json(OUT / 'configuration.json', {'judge': 'gpt-6.1-sol', 'reasoning_effort': 'high',
        'prompt': PROMPT, 'labels': 'Explicit dialogue facts and manually specified single-field changes',
        'raw_source_fields_used': ['idx', 'passage'], 'generated_openie_facts_used': False,
        'clinical_expert_validation': False})
    judge = JudgeClient('gpt-6.1-sol')

    def evaluate(case):
        path = OUT / 'verdicts' / f'{case["id"]}.json'
        if path.exists():
            return json.loads(path.read_text())
        result = check_source(case['fact'], case['source'], judge)
        result.update(id=case['id'], expected_supported=case['expected_supported'], error_type=case['error_type'])
        write_json(path, result)
        print(case['id'], result['supported'], case['expected_supported'], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=15) as pool:
        rows = list(pool.map(evaluate, cases))
    summary = {'n': len(rows), 'supported_cases': sum(r['expected_supported'] for r in rows),
        'corrupted_cases': sum(not r['expected_supported'] for r in rows),
        'false_accepts': [r['id'] for r in rows if r['supported'] and not r['expected_supported']],
        'false_rejects': [r['id'] for r in rows if not r['supported'] and r['expected_supported']],
        'reward_gate': 'Rejected edits are not applied and cannot receive positive QA reward',
        'scope': 'Finite source-grounding calibration; not a clinical correctness guarantee'}
    write_json(OUT / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
