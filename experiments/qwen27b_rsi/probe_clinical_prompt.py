"""Frozen prompt comparison on an authored fixture. No patient files or judge calls."""
import argparse
import json
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI

from medical_navigation import base, extraction_messages, parse_extraction
from targeted_guidance import CLINICAL_CUES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = {'source_id': 'fictional-lab-card', 'text': (
        'Invented unit-test record.\n'
        'Clinician: At the earlier visit, fictional marker X was 17.4 units after a meal.\n'
        'Clinician: Today fictional marker X is 12.6 units before breakfast.')}
    original = extraction_messages([source])
    pointer = 'Recheck fictional-lab-card, lines 1–2.'
    cue = pointer + ' ' + CLINICAL_CUES['followup_measurement']
    client = OpenAI(base_url='http://127.0.0.1:19109/v1', api_key='local', timeout=300,
                    http_client=DefaultHttpxClient(trust_env=False))
    report = {'fixture': 'Authored non-patient record', 'scope': 'Frozen FP8, mechanical value-to-line checks',
              'prompt': original, 'hint': cue, 'arms': {}}
    for arm, feedback in [('plain', None), ('lines_only', pointer), ('hinted', cue)]:
        messages = original if feedback is None else [*original[:-1], {**original[-1], 'content':
            original[-1]['content'] + '\nTraining feedback for this write:\n' + feedback}]
        response = client.chat.completions.create(model='macaq', messages=messages, temperature=0, seed=0,
            max_tokens=1536, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        choice = response.choices[0]
        row = {'finished': choice.finish_reason == 'stop', 'text': choice.message.content or ''}
        base.write_json(args.out / f'{arm}.sample.json', {**row, 'usage': response.usage.model_dump()})
        try:
            facts = parse_extraction(row, [source])['facts']
        except (ValueError, KeyError, TypeError) as exc:
            result = {'parsed': False, 'reason': str(exc), 'correct_value_line_pairs': 0}
        else:
            pairs = {value: any(fact['value'].strip() == value and fact['evidence_ids'] == [line]
                                 for fact in facts) for value, line in [('17.4', 1), ('12.6', 2)]}
            result = {'parsed': True, 'pairs': pairs, 'correct_value_line_pairs': sum(pairs.values())}
        report['arms'][arm] = result
    base.write_json(args.out / 'comparison.json', report)
    print(json.dumps(report['arms']))


if __name__ == '__main__':
    main()
