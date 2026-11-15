"""Query feedback shared by identical Reader prompts within a training episode."""
from concurrent.futures import Future
import json
import os
import re
import time
from threading import Lock
import query_writer_rl as base
from fact_query_likelihood import answer_likelihood
from semantic_query_writer_rl import RUBRIC


class QueryFeedback:
    def __init__(self, directory, scorer, dense_reward):
        self.directory = directory
        self.scorer = scorer
        self.dense_reward = dense_reward
        self.pending = {}
        self.lock = Lock()
        self.prior = {row['prompt']: row for path in directory.glob('*.json')
                      for row in [base.read(path)]}

    def __call__(self, question, prompt, request_id, retrieved_count):
        with self.lock:
            owner = prompt not in self.pending
            if owner:
                self.pending[prompt] = Future()
            future = self.pending[prompt]
        if owner:
            try:
                row = dict(self.prior.get(prompt, {'prompt': prompt, 'reader_request_id': request_id}))
                path = self.directory / f'{row["reader_request_id"]}.json'
                generated = 'answer' not in row
                if generated:
                    response = base.CLIENT.chat.completions.create(model='Qwen3.8-27B',
                        messages=[{'role': 'user', 'content': prompt}], temperature=0, seed=0,
                        max_tokens=16384, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                    content = response.choices[0].message.content or ''
                    matches = list(re.finditer(r'(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*', content, re.I))
                    row.update(answer=content[matches[-1].end():].strip() if matches else content.strip(),
                               response=content, usage=response.usage.model_dump(),
                               finish_reason=response.choices[0].finish_reason)
                    ledger = base.ROOT / 'results/reader_usage'
                    ledger.mkdir(parents=True, exist_ok=True)
                    with (ledger / f'{os.getpid()}.jsonl').open('a') as handle:
                        handle.write(json.dumps({'time': time.time(), 'model': response.model,
                            'request_id': request_id, 'cache_directory': str(self.directory),
                            'usage': row['usage']}) + '\n')
                    base.write_json(path, row)
                if 'score' not in row:
                    row['score'] = self.scorer({'question': {
                        'query_id': question['query_id'], 'query_type': question['query_type'],
                        'model_output': row['answer'], 'expected_answers': question['correct_answers'],
                        'question': question['question'], 'answers_data': question['answers'],
                        'metadata': question['metadata']}, 'seconds': 0., 'retrieved_count': retrieved_count})
                    base.write_json(path, row)
                if self.dense_reward and 'query_likelihood' not in row:
                    row['query_likelihood'] = answer_likelihood(str(base.CLIENT.base_url), prompt, question['correct_answers'])
                    base.write_json(path, row)
                if 'semantic_score' not in row:
                    payload = {'question': question['question'], 'question_type': question['query_type'],
                               'reference_answers': [a['content'] for a in question['answers'] if a.get('is_correct')],
                               'response': row['answer']}
                    verdict = self.scorer.client.chat(messages=[{'role': 'system', 'content': RUBRIC},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                        max_tokens=4096, reasoning_effort='high', response_format={'type': 'json_object'})
                    row['semantic_score'] = json.loads(verdict.content)
                    assert type(row['semantic_score']['correct']) is bool
                    row['semantic_rubric'] = RUBRIC
                    base.write_json(path, row)
                future.set_result((row, generated))
            except BaseException as error:
                future.set_exception(error)
                raise
        saved, generated = future.result()
        row = {k: v for k, v in saved.items() if k != 'prompt'}
        row['reader_cache_hit'] = not (owner and generated)
        return row


def group_rewards(samples):
    query = [r['query_accuracy'] + .2*r['query_likelihood'] if r['valid'] else None for r in samples]
    valid = [q for q in query if q is not None]
    span = max(valid)-min(valid) if valid else 0.
    active = span >= .002
    normalized = [(q-min(valid))/span if q is not None and active else 0. for q in query]
    rewards = [normalized[i]-.25*r['unsupported_fraction']-.01*r['write_cost'] if r['valid'] else -1.
               for i, r in enumerate(samples)]
    return {'rewards': rewards, 'query_reward_components': query,
            'normalized_query_components': normalized, 'query_signal_span': span,
            'query_signal_active': active, 'query_signal_min_span': .002}
