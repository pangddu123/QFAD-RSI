"""Construct training-only feedback for the particular QFAD write that failed."""

CLINICAL_CUES = {
    'treatment_status': 'Separate the prescribed regimen from the patient reporting actual use, a missed dose, or stopping. Cite the speaker and the line establishing that state; leave earlier use retrievable.',
    'event_time': 'Keep the event time separate from the record date. Keep relative or uncertain timing at the precision used in the dialogue.',
    'measurement': 'Index each result with the analyte, original unit, collection time and setting when stated. Keep a prior value quoted in a follow-up separate from the new result. Do not silently convert units or supply a missing fasting status.',
    'assertion': 'Preserve the speaker and strength of the assertion. Rare, milder or transient symptoms are still symptoms; reserve a negative finding for an explicit denial. Distinguish an unknown value from a normal result.',
    'longitudinal_change': 'Keep the earlier and later state independently reachable, including the duration of any response. A temporal sequence alone does not establish causality.',
    'patient_constraint': 'Preserve whose constraint this is and what was actually reported. Separate an allergy history, a suspected adverse effect, and a precaution; cite the original reporter.',
    'monitoring': 'Distinguish the reason for follow-up testing from an observed abnormal result. Cite the monitoring instruction and its conditions.',
    'baseline_measurement': 'Make the original examination retrievable by its date and measurement setting. Keep a value repeated later as historical when that is how the speaker describes it.',
    'followup_measurement': 'Locate the measurement performed at this follow-up. When a sentence contains two values, give the comparison value and the new value separate time-qualified cues with their original units.',
    'subject': 'Identify the person who experienced the event separately from the speaker. Keep family history and general counseling separate from the patient’s observations.',
}


def anchor_lines(source, anchor):
    original = source['text']
    quote = anchor['quote']
    if 'span_id' in anchor:
        first = anchor['span_id']
        lines = original.splitlines()
        if type(first) is not int or not 0 <= first < len(lines) or lines[first] != quote:
            raise ValueError('Training anchor disagrees with its original line')
    else:
        start = original.find(quote)
        if start < 0 or not quote:
            raise ValueError('Training anchor no longer resolves to original dialogue')
        if original.find(quote, start + 1) >= 0:
            return None
        first = original.count('\n', 0, start)
    return first, first + quote.count('\n')


def window_lines(trace):
    return {w['source_id']: set(w['line_ids']) for w in trace['source_windows']}


def condition_status(prefix, bundle, condition):
    verdict, _ = prefix
    if not verdict['bundles']:
        return False, []
    row = verdict['bundles'][bundle]['conditions'][condition]
    return row['covered'], row.get('source_ids', [])


def attribute_writes(turns, queries, histories):
    """histories[q] contains (coverage, retrieval) before writing and after each turn."""
    events = [[] for _ in turns]
    owners = {sid: t for t, turn in enumerate(turns) for sid in turn['sources']}
    if len(owners) != sum(len(turn['sources']) for turn in turns):
        raise ValueError('A source has multiple owners in this rollout')
    for query, history in zip(queries, histories, strict=True):
        if len(history) != len(turns) + 1:
            raise ValueError('Expected an initial and one post-write retrieval per turn')
        verdict, trace = history[-1]
        if verdict['complete']:
            continue
        bundles = query['requirements']['bundles']
        # Target one acceptable bundle, chosen by its observed coverage. An empty
        # retrieval has no judge verdict and all of its conditions are missing.
        judged = verdict['bundles']
        best = max(range(len(bundles)), key=lambda i: sum(c['covered'] for c in
                   judged[i]['conditions']) / len(bundles[i]['conditions'])) if judged else 0
        delivered = window_lines(trace)
        for i, condition in enumerate(bundles[best]['conditions']):
            if judged and judged[best]['conditions'][i]['covered']:
                continue
            criterion = condition.get('criterion', 'source_fidelity')
            cue = CLINICAL_CUES.get(criterion, 'Preserve the stated subject, event and qualifiers in their original wording.')
            detail = {'query_id': query.get('query_id'), 'criterion': criterion, 'bundle': best, 'condition': i}
            # If coverage existed, attribute only the last observed covered -> missing transition.
            # A changed semantic verdict alone is insufficient; supporting original lines must leave.
            prior = [j for j in range(len(turns)) if condition_status(history[j], best, i)[0]]
            if prior:
                t = prior[-1]
                previous = history[t][1]
                following = history[t + 1][1]
                before, after = window_lines(previous), window_lines(following)
                relevant = condition_status(history[t], best, i)[1]
                lost = {}
                for sid in relevant:
                    missing = before.get(sid, set()) - after.get(sid, set())
                    if sid in after:
                        # A shorter window can still contain the supporting line.
                        # For partial loss, require a verified anchor among the removed lines.
                        anchored = set()
                        if sid in owners:
                            source = turns[owners[sid]]['sources'][sid]
                            for anchor in condition['anchors']:
                                if anchor['source_id'] == sid:
                                    span = anchor_lines(source, anchor)
                                    if span is not None:
                                        anchored.update(range(span[0], span[1] + 1))
                        missing &= anchored
                    if missing:
                        lost[sid] = sorted(missing)
                if lost and not set(lost).intersection(set(following.get('context_skipped_source_ids', ())) |
                                                      set(trace.get('context_skipped_source_ids', ()))):
                    entrants = [s for s in turns[t]['sources'] if s in after and s not in before]
                    pointer = (' Newly delivered from this write: ' + ', '.join(entrants) + '.'
                               if entrants else '')
                    events[t].append({**detail, 'cause': 'coverage_regression', 'lost_lines': lost,
                        'new_source_ids': entrants,
                        'text': 'After this write the same query lost previously delivered supporting lines. '
                                'Recheck the distinguishing qualifiers of this block and its links to older events. '
                                'Keep supported observations in both records.' + pointer + ' ' + cue})
                continue
            for anchor in condition['anchors']:
                sid = anchor['source_id']
                if sid not in owners:
                    continue
                t = owners[sid]
                span = anchor_lines(turns[t]['sources'][sid], anchor)
                if span is None:
                    continue
                first, last = span
                if set(range(first, last + 1)) <= delivered.get(sid, set()):
                    continue
                own_trace = history[t + 1][1]
                own_lines = window_lines(own_trace)
                if (set(range(first, last + 1)) <= own_lines.get(sid, set()) or
                    any(sid in tr.get('context_skipped_source_ids', ()) for tr in (trace, own_trace))):
                    continue
                if sid in own_lines:
                    failure = 'The source reached the reader, but its evidence window missed these lines.'
                elif sid in own_trace.get('rerank_source_ids', ()):
                    failure = 'QFAD shortlisted this source, but it did not enter the delivered evidence.'
                elif sid in own_trace.get('qfad_source_order', ()):
                    failure = 'QFAD ranked this source outside the reranking budget.'
                else:
                    failure = 'The source had no reachable index entry in this query trace.'
                if sid not in own_lines:
                    cited = {c['span_id'] for fact in (turns[t].get('admitted') or {}).get('facts', [])
                             for c in fact['citations'] if c['source_id'] == sid}
                    if set(range(first, last + 1)) <= cited:
                        continue  # Faithful pointer exists; ranking alone does not identify a bad write.
                events[t].append({**detail, 'cause': 'source_omission', 'source_id': sid, 'lines': span,
                                  'text': f'{failure} Recheck {sid}, lines {first}–{last}. {cue}'})
    for t, turn in enumerate(turns):
        if not turn['parsed']:
            events[t].insert(0, {'cause': 'format', 'text': 'Return complete facts and relations JSON with valid '
                'source and line IDs. ' + str(turn['support']['parse_error'])[:350]})
            continue
        for rejected in turn['admission']['rejected']:
            atom = rejected.get('id', rejected.get('index'))
            cue = CLINICAL_CUES.get(rejected.get('criterion'), 'Remove or qualify only what the cited lines do not support.')
            extraction = turn.get('extraction') or {}
            if rejected['kind'] == 'fact':
                proposed = next((f for f in extraction.get('facts', []) if f['id'] == atom), {})
            else:
                relations = extraction.get('relations', [])
                proposed = relations[atom] if isinstance(atom, int) and 0 <= atom < len(relations) else {}
            source = proposed.get('source_id')
            lines = proposed.get('evidence_ids', [])
            pointer = f' Recheck {source}, lines {lines}.' if source in turn['sources'] and lines else ''
            events[t].append({'cause': 'source_verification', 'atom': atom,
                'criterion': rejected.get('criterion'), 'source_id': source, 'lines': lines,
                'text': f"The verifier rejected {rejected['kind']} {atom}: {rejected['reason'][:350]}{pointer} {cue}"})
    return events


def render_hint(events):
    unique = list(dict.fromkeys(e['text'] for e in events))[:3]
    if unique:
        return ('Repair only the affected navigation cues using the original lines. '
                'Keep supported facts and their citations. Do not infer a diagnosis or treatment.\n'
                + '\n'.join(unique))
    return None


def extraction_hint(sources, queries, result, traces):
    empty = ({'complete': False, 'bundles': []}, {'source_windows': []})
    histories = [[empty, pair] for pair in zip(result['coverage'], traces, strict=True)]
    return render_hint(attribute_writes([{**result, 'sources': sources}], queries, histories)[0])
