"""Source-anchored patient-1/2 episodes for the initial extraction training run."""
import argparse
from pathlib import Path

from medical_navigation import NavigationGraph, base
from clinical_evidence import validate_requirements


def training_plan():
    graphs = {pid: NavigationGraph(base.Patient(pid, store_names=('chunk',))) for pid in (1, 2)}
    s0 = 'chunk-3b6b49301012484eb94a144d137bb130'
    s1 = 'chunk-5f7e01ce2f3948cd54430ee693845c38'
    s2 = 'chunk-cb72e33efafc9ad514367060b88e7c2c'
    followup = 'chunk-159b311bbe7299e8b3feead6bbc2d611'

    def query(pid, qid, conditions):
        stage = int(qid.split('_')[1]) // 10 - 1
        assert any(q['query_id'] == qid for q in graphs[pid].patient.groups[stage][1])
        requirements = {'bundles': [{'conditions': [
            {'description': description, 'anchors': [
                {'source_id': sid, 'span_id': line, 'quote': graphs[pid].sources[sid]['text'].splitlines()[line]}
                for sid, line in anchors], 'criterion': criterion}
                for criterion, description, anchors in conditions]}]}
        validate_requirements(requirements, graphs[pid], stage)
        return {'query_id': qid, 'stage': stage, 'requirements': requirements}

    allergy = query(1, 'session_10_eem_2', [
        ('patient_constraint', 'The patient reports a previous allergic reaction to cefuroxime and being told to avoid it.', [(s0, 1)])])
    regimen = query(1, 'session_30_tla_2', [
        ('treatment_status', 'The patient reports starting the prescribed metformin 1500 mg/day regimen on the day of the record, not merely receiving a recommendation.', [(s0, 1), (s0, 27)]),
        ('event_time', 'That record is dated January 6, 2024.', [(s0, 0)])])
    stopped = query(1, 'session_10_sua_1', [
        ('treatment_status', 'The dialogue explicitly records that empagliflozin has been stopped, not merely that stopping was recommended.', [(s1, 10)])])
    if '9.2%' not in graphs[1].sources[s0]['text'].splitlines()[26] or '8.1%' not in graphs[1].sources[followup]['text'].splitlines()[1]:
        raise ValueError('HbA1c curriculum anchors disagree with original records')
    baseline = query(1, 'session_70_eem_2', [
        ('baseline_measurement', 'The initial examination HbA1c was 9.2%.', [(s0, 26)])])
    latest = query(1, 'session_20_eem_1', [
        ('followup_measurement', 'The follow-up HbA1c was 8.1%; the previous 9.2% is the comparator.', [(followup, 1)])])
    continued = [
        ('treatment_status', 'The patient reports an instruction to continue perindopril and spironolactone at the existing doses.', [(s2, 1), (s2, 4)]),
        ('monitoring', 'The same medication discussion calls for electrolyte or potassium monitoring and warns about elevated potassium.', [(s2, 1), (s2, 8)])]
    order = query(2, 'session_10_tla_2', continued + [
        ('event_time', 'The record containing this medication instruction is dated January 18, 2024.', [(s2, 0)])])
    monitoring = query(2, 'session_10_mq_2', [
        ('measurement', 'The patient reports stable blood pressure around 128/76 at follow-up.', [(s2, 1)])] + continued)
    episodes = [
        {'id': 'p1_regimen', 'patient': 1, 'source_ids': [s0], 'queries': [regimen, allergy]},
        {'id': 'p1_stopped', 'patient': 1, 'source_ids': [s1], 'queries': [stopped, allergy]},
        {'id': 'p1_repeat_measure', 'patient': 1, 'source_ids': [followup], 'queries': [baseline, latest]},
        {'id': 'p2_continue_monitor', 'patient': 2, 'source_ids': [s2], 'queries': [order, monitoring]},
    ]
    return {'episodes': episodes, 'turns_per_rollout': 3,
            'requirements_provenance': 'Original source lines checked verbatim',
            'scope': 'Three chronological writes for patient 1 and one for patient 2'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=base.ROOT / 'results/medical_sage/extraction_training_plan.json')
    base.write_json(parser.parse_args().out, training_plan())
    print('Prepared four source blocks with seven distinct source-anchored benchmark queries')


if __name__ == '__main__':
    main()
