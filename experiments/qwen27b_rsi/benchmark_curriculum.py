"""Prepare Target Writer training from benchmark session annotations and original QFAD passages."""
from collections import Counter, defaultdict
import re

import pandas as pd
import tiktoken

from medical_navigation import base
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset


TRAIN_PATIENTS = (5, 6, 7, 8, 9, 10)
VALIDATION_PATIENTS = (1, 2)
TEST_PATIENTS = (3, 4)


CRITERIA = {'Test Results': 'measurement', 'Physiological Indicators': 'measurement',
            'Medication Records': 'treatment_status', 'Symptoms': 'assertion',
            'Symptom Fluctuations': 'longitudinal_change', 'Physiological Monitoring': 'monitoring'}


def terms(text):
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold()))


def patient_episodes(pid, data, encoder):
    dataset = MedMemoryBenchDataset(data, dict(evaluation_mode='independent', persona_ids=[pid],
                                               evaluation_interval=10, inject_noise=True))
    dataset.load()
    store = base.SOURCE / f'persona_{pid:02d}/qafd/gpt-4o-mini_qwen3-embedding-4b/chunk_embeddings/vdb_chunk.parquet'
    frame = pd.read_parquet(store, columns=['hash_id', 'content'])
    lookup = dict(zip(frame.content, frame.hash_id, strict=True))
    sessions, passages, queries, first_stage = {}, {}, {}, {}
    for stage, unit in enumerate(dataset.get_evaluation_units()):
        for session in unit.sessions_to_inject:
            if session.is_noise:
                continue
            if session.session_id in sessions:
                raise ValueError(f'Duplicate session ID for patient {pid}')
            ids = []
            for text in base.harness.session_passages(session, encoder):
                sid = lookup[text]
                if sid not in passages:
                    passages[sid] = (text, terms(text))
                    first_stage[sid] = stage
                ids.append(sid)
            sessions[session.session_id] = (stage, ids)
        for query in unit.queries_to_evaluate:
            if query.query_id in queries:
                raise ValueError(f'Duplicate query for patient {pid}')
            queries[query.query_id] = (stage, query)

    writes = defaultdict(dict)
    excluded = Counter()
    for qid, (stage, query) in queries.items():
        points = query.source_key_points
        if not any(a.get('is_correct') and str(a.get('content') or '').strip() for a in query.answers_data):
            excluded['no_reference_answer'] += 1
            continue
        if not points or any(not (p.get('content') or '').strip() for p in points):
            excluded['missing_key_point'] += 1
            continue
        if any(p.get('session_id') not in sessions or sessions[p['session_id']][0] > stage for p in points):
            excluded['missing_or_future_source_session'] += 1
            continue
        selected, conditions = set(), []
        for point in points:
            description = point['content'].strip()
            session_id = point['session_id']
            _, ids = sessions[session_id]
            # Lexical overlap chooses a write to sample. It is not evidence that the passage entails the point.
            words = terms(description)
            candidate = max(ids, key=lambda sid: len(words & passages[sid][1]))
            anchors = []
            for sid in ids:
                for i, line in enumerate(passages[sid][0].splitlines()):
                    if description in line:
                        anchors.append(dict(source_id=sid, span_id=i, quote=line))
            selected.update([a['source_id'] for a in anchors] or [candidate])
            conditions.append(dict(description=description, session_id=session_id,
                                   criterion=CRITERIA.get(point.get('category'), 'source_fidelity'),
                                   session_source_ids=ids, anchors=anchors))
        item = dict(query_id=qid, stage=stage, requirements={'bundles': [{'conditions': conditions}]})
        for sid in selected:
            writes[sid][qid] = item
    episodes = [dict(id=f'p{pid}_{sid}', patient=pid, source_stage=first_stage[sid], source_ids=[sid], queries=list(writes[sid].values()))
                for sid in passages if sid in writes]
    return episodes, dict(total=len(queries), excluded=dict(excluded))


def training_plan():
    data = base.ROOT / 'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    encoder = tiktoken.encoding_for_model('gpt-4o')
    episodes, audit = [], {}
    for pid in TRAIN_PATIENTS:
        rows, audit[pid] = patient_episodes(pid, data, encoder)
        if not rows:
            raise ValueError(f'No usable training queries for patient {pid}')
        episodes.extend(rows)
    return dict(episodes=episodes, turns_per_rollout=3, patient_audit=audit,
                patient_split=dict(train=list(TRAIN_PATIENTS), validation=list(VALIDATION_PATIENTS),
                                   test=list(TEST_PATIENTS)),
                requirements_provenance='Benchmark source key points and visible non-noise session IDs. '
                'Line hints require a verbatim match; candidate writes are weakly localized within the session.')
