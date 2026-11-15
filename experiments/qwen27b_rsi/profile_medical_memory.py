"""Describe the training cohort's observable stream and evidence annotations."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'MedMemoryBench'))
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset


STATE_TARGETS = {
    1: {10: 'medication', 20: 'measurement', 30: 'behavior_or_preference',
        40: 'symptom_or_experience', 50: 'measurement', 60: 'behavior_or_preference',
        70: 'measurement', 80: 'symptom_or_experience', 90: 'measurement', 100: 'measurement'},
    2: {10: 'behavior_or_preference', 20: 'measurement', 30: 'symptom_or_experience',
        40: 'measurement', 50: 'behavior_or_preference', 60: 'behavior_or_preference',
        70: 'measurement', 80: 'measurement', 90: 'symptom_or_experience', 100: 'medication'},
}


def main():
    data = ROOT / 'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    dataset = MedMemoryBenchDataset(data, {
        'evaluation_mode': 'independent', 'persona_ids': [1, 2],
        'evaluation_interval': 10, 'inject_noise': True,
    })
    dataset.load()
    units = list(dataset.get_evaluation_units())
    report = {'data': str(data), 'patients': {},
              'scope': 'Patients 1 and 2 only. Annotation counts describe the release, not verified clinical evidence.'}
    examples = []
    for pid in (1, 2):
        directory = data / f'persona_{pid}/eval'
        sessions = json.loads((directory / 'generated_dialogues_with_noise.json').read_text())['sessions']
        queries = json.loads((directory / 'generated_queries.json').read_text())['queries']
        patient_units = [u for u in units if u.context_id == pid]
        visible = [s for u in patient_units for s in u.sessions_to_inject]
        counts = Counter(q['query_type'] for q in queries)
        points = [p for q in queries for p in q['source_key_points']]
        main_session_ids = {s['session_id'] for s in sessions if 'noise_type' not in s}
        widths = [len(q['source_key_points']) for q in queries]
        by_type = {}
        for kind in counts:
            selected = [q for q in queries if q['query_type'] == kind]
            by_type[kind] = {
                'queries': len(selected),
                'source_point_counts': [len(q['source_key_points']) for q in selected],
                'multiple_source_sessions': sum(len({p.get('session_id') for p in q['source_key_points']}) > 1 for q in selected),
            }
        report['patients'][str(pid)] = {
            'queries': len(queries), 'query_types': dict(counts),
            'file_session_types': dict(Counter(s.get('noise_type', 'main') for s in sessions)),
            'visible_session_types': dict(Counter(s.noise_type if s.is_noise else 'main' for s in visible)),
            'visible_messages': sum(len(s.metadata['messages']) for s in visible),
            'checkpoints': [u.metadata['eval_session_id'] for u in patient_units],
            'source_key_point_occurrences': len(points),
            'max_source_points_per_query': max(widths),
            'queries_with_over_10_source_points': sum(n > 10 for n in widths),
            'source_point_time_missing': sum(p.get('time') in (None, '', 'None') for p in points),
            'unresolved_source_session_ids': [
                {'query_id': q['query_id'], 'source_session': p.get('session_id')}
                for q in queries for p in q['source_key_points']
                if p.get('session_id') not in main_session_ids],
            'valid_source_session_after_query_checkpoint': [
                {'query_id': q['query_id'], 'source_session': p.get('session_id')}
                for q in queries for p in q['source_key_points']
                if p.get('session_id') in main_session_ids and p['session_id'] > q['session_id']],
            'by_type': by_type,
            'state_question_target_counts': dict(Counter(STATE_TARGETS[pid].values())),
            'state_target_annotation': 'Manual categorization of the question target, not clinical truth labels',
        }
        assert {q['session_id'] for q in queries if q['query_type'] == 'state_update'} == set(STATE_TARGETS[pid])
        for q in queries:
            if q['query_type'] == 'state_update':
                examples.append({'patient': pid, 'query_id': q['query_id'],
                                 'question': q['question'], 'annotation_count': len(q['source_key_points']),
                                 'target': STATE_TARGETS[pid][q['session_id']],
                                 'source_sessions': sorted({p['session_id'] for p in q['source_key_points']})})
    report['state_update_examples'] = examples
    out = ROOT / 'QFAD-RSI/research/medical_sage'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dataset_profile.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({pid: {k: row[k] for k in ('queries', 'query_types', 'visible_session_types',
          'visible_messages', 'max_source_points_per_query', 'queries_with_over_10_source_points',
          'source_point_time_missing', 'unresolved_source_session_ids',
          'valid_source_session_after_query_checkpoint')}
          for pid, row in report['patients'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
