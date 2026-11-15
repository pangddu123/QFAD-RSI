"""Summarize completed fact-writer updates and recorded model token usage."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]/'results/qfad_fact_writer_rl_20261114'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = ROOT/args.run
    count = read(root/'pilot_complete.json')['steps']
    runtime = read(root/'worker_runtime.json')
    writer_prompt = writer_completion = replay_tokens = 0
    sampled = valid = accuracy_signal = query_signal = 0
    groups = []
    for step in range(count):
        packet = read(root/f'rollouts_{step}.json')
        update = read(root/f'update_{step}.json')
        rewards = read(root/f'rewards_{step}.json')
        assert update['episode_id'] == packet['episode_id'] and update['rewards'] == rewards['rewards']
        rows = list(packet['rows'])
        sampled += len(rows)
        replay_tokens += sum(packet['prompt_tokens']+len(r['token_ids']) for r in rows)
        for greedy in (packet['initial_greedy'], update['post_update_greedy']):
            if greedy is not None:
                rows.append(greedy)
        writer_prompt += packet['prompt_tokens']*len(rows)
        writer_completion += sum(len(r['token_ids']) for r in rows)
        accepted = [r for r in rewards['samples'] if r['valid']]
        correct = [r['query_accuracy'] for r in accepted]
        has_accuracy_signal = bool(correct) and max(correct) != min(correct)
        valid += len(accepted)
        accuracy_signal += has_accuracy_signal
        query_signal += bool(update['query_signal_active'])
        groups.append({'step': step, 'episode_id': packet['episode_id'], 'valid_candidates': len(accepted),
            'query_accuracies': correct, 'query_signal_active': update['query_signal_active'],
            'query_signal_span': update['query_signal_span'], 'accuracy_signal_active': has_accuracy_signal,
            'adapter_delta_l2': update['adapter_delta_l2'], 'gradient_norm': update['gradient_norm']})
    contexts = [read(p) for p in (root/'reader_cache').glob('*/*.json')]
    reader_prompt = sum(r['usage']['prompt_tokens'] for r in contexts)
    reader_completion = sum(r['usage']['completion_tokens'] for r in contexts)
    likelihood_prompt = sum(r['query_likelihood']['usage']['prompt_tokens'] for r in contexts)
    likelihood_completion = sum(r['query_likelihood']['usage']['completion_tokens'] for r in contexts)
    summary = {'run': args.run, 'completed_updates': count, 'sampled_candidates': sampled,
        'valid_candidates': valid, 'updates_with_query_signal': query_signal,
        'updates_with_accuracy_signal': accuracy_signal,
        'updates_with_likelihood_only_signal': query_signal-accuracy_signal,
        'base_parameters_frozen': runtime['base_parameters_frozen'],
        'trainable_parameters': runtime['trainable_parameters'],
        'writer_generation': {'prompt_tokens': writer_prompt, 'completion_tokens': writer_completion},
        'optimizer_replay_tokens': replay_tokens,
        'reader_feedback': {'unique_contexts': len(contexts), 'prompt_tokens': reader_prompt,
                            'completion_tokens': reader_completion},
        'likelihood_feedback': {'prompt_tokens': likelihood_prompt, 'completion_tokens': likelihood_completion},
        'source_judge_verdicts': len(list((root/'pilot').glob('*/source_support.json'))),
        'judge_token_usage_recorded': False, 'groups': groups}
    (root/'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print({k: summary[k] for k in ('completed_updates', 'sampled_candidates', 'valid_candidates',
           'updates_with_query_signal', 'updates_with_accuracy_signal', 'updates_with_likelihood_only_signal')})


if __name__ == '__main__':
    main()
