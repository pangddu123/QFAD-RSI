"""Clinical evidence-bundle feedback for a source-indexing Writer."""
import json
import math
from concurrent.futures import Future
import threading


COVERAGE_PROMPT = '''Determine which required clinical evidence conditions are supported by the
DELIVERED ORIGINAL TEXT. Requirements were defined from verified historical sources; those
reference sources are not themselves delivered evidence. For every condition in every acceptable
bundle, return {"covered":true,"source_ids":["..."],"reason":"..."} in the same order, under
{"bundles":[{"conditions":[...]}]}. Use only delivered source text to decide coverage.
Check the requested time, medication execution status, measurement setting, uncertainty and
necessary before/after relationships. A recommendation does not establish execution. An older
measurement of the same indicator does not establish a later value. Absence of a statement is
not a negative result. Partial/transient relief does not establish sustained recovery. Necessary
negative evidence and patient constraints count when the requirement calls for them. No medical
inference beyond the explicitly required relationships. Ignore answer fluency and graph size.
Each covered condition must cite at least one actually delivered source ID; a relationship may
need several. Repeated copies of the same evidence do not cover additional conditions.'''


def validate_requirements(requirements, graph, stage):
    if not requirements['bundles']:
        raise ValueError('Clinical requirements must have an evidence bundle')
    for bundle in requirements['bundles']:
        if not bundle['conditions']:
            raise ValueError('Evidence bundle has no conditions')
        for condition in bundle['conditions']:
            if not condition['description'].strip() or not condition['anchors']:
                raise ValueError('Evidence condition needs source-anchored meaning')
            for anchor in condition['anchors']:
                source = graph.sources[anchor['source_id']]
                if source['stage'] > stage:
                    raise ValueError('Requirement source is not visible at the query checkpoint')
                if not anchor['quote'] or anchor['quote'] not in source['text']:
                    raise ValueError('Requirement anchor is not verbatim original evidence')


def score_evidence(question, requirements, evidence, judge):
    if not evidence['documents']:
        return {'coverage': 0., 'complete': False, 'bundles': [], 'usage': {'total_tokens': 0}}
    descriptions = [[c['description'] for c in bundle['conditions']] for bundle in requirements['bundles']]
    response = judge.responses(messages=[{'role': 'system', 'content': COVERAGE_PROMPT},
        {'role': 'user', 'content': json.dumps({'question': question,
            'acceptable_bundles': descriptions,
            'delivered_original_text': evidence['documents'],
            'delivered_source_ids': evidence['source_ids']}, ensure_ascii=False)}],
        max_tokens=4096, reasoning_effort='high')
    verdict = json.loads(response.content)
    assert len(verdict['bundles']) == len(descriptions)
    coverages = []
    for actual, expected in zip(verdict['bundles'], descriptions):
        assert len(actual['conditions']) == len(expected)
        for condition in actual['conditions']:
            assert type(condition['covered']) is bool
            assert isinstance(condition['source_ids'], list)
            assert set(condition['source_ids']) <= set(evidence['source_ids'])
            assert not condition['covered'] or condition['source_ids']
        coverages.append(sum(c['covered'] for c in actual['conditions']) / len(expected))
    coverage = max(coverages)
    return {**verdict, 'coverage': coverage, 'complete': coverage == 1., 'usage': response.usage}


class EvidenceScorer:
    """Identical evidence in paired candidates shares one judge realization."""
    def __init__(self, judge):
        self.judge = judge
        self.cache = {}
        self.lock = threading.Lock()

    def score(self, question, requirements, evidence):
        key = json.dumps([question,
            [[c['description'] for c in b['conditions']] for b in requirements['bundles']],
            evidence['documents'], evidence['source_ids']], ensure_ascii=False, sort_keys=True)
        with self.lock:
            pending = self.cache.get(key)
            owner = pending is None
            if owner:
                pending = self.cache[key] = Future()
        if owner:
            try:
                pending.set_result(score_evidence(question, requirements, evidence, self.judge))
            except BaseException as error:
                pending.set_exception(error)
                raise
        return {**pending.result(), 'cache_hit': not owner}


def task_reward(answers, coverage):
    """Query utility of the admitted graph; source errors are per-atom feedback."""
    if not answers or len(answers) != len(coverage):
        raise ValueError('Reward requires a fixed nonempty query set')
    success = sum(bool(a['score']['correct']) and c['complete'] for a, c in zip(answers, coverage))
    return success + .5 * sum(c['coverage'] for c in coverage) / len(coverage)


def extraction_reward(answers, coverage, admission):
    utility = task_reward(answers, coverage)
    if admission is None:
        error_rate = 1.
    else:
        proposed = admission['proposed_facts'] + admission['proposed_relations']
        accepted = admission['accepted_facts'] + admission['accepted_relations']
        error_rate = (proposed - accepted) / proposed if proposed else 0.
    return {'task_utility': utility, 'source_error_rate': error_rate,
            'reward': utility - .25 * error_rate}


def has_task_signal(candidates):
    utilities = [r['feedback']['task_utility'] for r in candidates if r['parsed']]
    return len(set(utilities)) > 1


def group_advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError('Group comparison requires at least two finite rewards')
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** .5
    return [(r - mean) / std for r in rewards] if std else [0.] * len(rewards)
