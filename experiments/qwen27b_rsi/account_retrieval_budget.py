"""Account for source indexing and RAW retrieval using service-native tokenizers."""
import argparse
from collections import Counter
import json
from pathlib import Path

from graph_environment import GraphEnvironment, base, fact_text
from retrieval_tokens import count_tokens, rerank_tokens


def final_raw_index_cost(persona):
    index = base.read(base.RAW / f'persona_{persona:02d}/rag_index.json')
    texts = [chunk['content'] for conversation in index['conversations'].values()
             for chunk in conversation['chunks']]
    text_set = set(texts)
    with (base.RAW / f'persona_{persona:02d}/logs/rag_requests.jsonl').open() as stream:
        for line in stream:
            if '"call_type": "embedding"' not in line[:600] or '"stage": "add"' not in line[:600]:
                continue
            record = json.loads(line)
            if record.get('error') or not record.get('response'):
                continue
            assert all(text in text_set for text in record['request']['input'])
            break
        else:
            raise ValueError('No successful index embedding request')
    return {'raw_final_index_chunks': len(texts),
            'raw_final_index_once_native_tokens': count_tokens(texts, 'Qwen3-Embedding-4B')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--patients', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()
    state = base.read(args.state)
    output = {'tokenizers': 'Copied from the running 2A100 retrieval service weights',
              'index_basis': 'One embedding per final index entry for both methods; actual RAW calls reported separately',
              'patients': {}, 'embedding_tokenizer_checks': []}
    for persona in args.patients:
        graph = GraphEnvironment(base.Patient(persona, store_names=('chunk',)), state.get(str(persona)))
        query_ids = {q['query_id'] for _, queries in graph.patient.groups for q in queries}
        stage_tokens, stage_calls = Counter(), Counter()
        queries, reranking = {}, {}
        checked_stages = set()
        with (base.RAW / f'persona_{persona:02d}/logs/rag_requests.jsonl').open() as stream:
            for line in stream:
                if not any(f'"call_type": "{kind}"' in line[:600] for kind in ('embedding', 'rerank')):
                    continue
                record = json.loads(line)
                if record.get('error') or not record.get('response'):
                    continue
                kind, stage = record['call_type'], record['stage']
                if kind == 'embedding':
                    tokens = record['response']['usage']['total_tokens']
                    stage_tokens[stage] += tokens
                    stage_calls[stage] += 1
                    if stage not in checked_stages:
                        texts = record['request']['input']
                        texts = [texts] if isinstance(texts, str) else texts
                        counted = count_tokens(texts, 'Qwen3-Embedding-4B')
                        assert counted == tokens, (persona, stage, counted, tokens)
                        output['embedding_tokenizer_checks'].append({
                            'patient': persona, 'stage': stage, 'counted': counted, 'reported': tokens})
                        checked_stages.add(stage)
                    if stage == 'search' and record['question_id'] in query_ids:
                        queries[record['question_id']] = tokens
                elif record['question_id'] in query_ids:
                    reranking[record['question_id']] = {
                        'native_input_tokens': rerank_tokens(record['request']),
                        'documents': len(record['request']['documents'])}
        assert set(queries) == query_ids and set(reranking) == query_ids
        dynamic_texts = [q['question'] for _, rows in graph.patient.groups for q in rows]
        dynamic_texts += [fact_text(f) for f in graph.state['facts'].values()]
        dynamic_texts += [f[k].lower().strip() for f in graph.state['facts'].values() for k in ('subject', 'object')]
        result = {
            'queries': len(query_ids), 'raw_embedding_native_tokens_by_stage': dict(stage_tokens),
            'raw_embedding_successful_calls_by_stage': dict(stage_calls),
            'raw_query_embedding_native_tokens': queries, 'raw_reranking': reranking,
            'raw_rerank_native_input_tokens': sum(r['native_input_tokens'] for r in reranking.values()),
            'graph_unique_sources': len(graph.sources),
            'graph_source_index_native_tokens': count_tokens([s['text'] for s in graph.sources.values()],
                                                            'Qwen3-Embedding-4B'),
            'graph_query_fact_entity_embedding_native_tokens_without_duplicate_requests': count_tokens(
                list(dict.fromkeys(dynamic_texts)), 'Qwen3-Embedding-4B')}
        result.update(final_raw_index_cost(persona))
        output['patients'][str(persona)] = result
        base.write_json(args.out, output)
        print('patient', persona, 'raw index', stage_tokens['add'],
              'graph index', result['graph_source_index_native_tokens'],
              'RAW rerank', result['raw_rerank_native_input_tokens'], flush=True)


if __name__ == '__main__':
    main()
