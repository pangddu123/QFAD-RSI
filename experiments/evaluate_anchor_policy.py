"""Validate learned writer policies, freeze one, and evaluate target patients."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil
from qfad_anchor_writer import AnchorPatient, OUT, base, episode
from fit_anchor_policy import choose


def run(split):
    validation = split == 'validation'
    names = ['global', 'contextual'] if validation else ['frozen']
    policies = {name: base.read(OUT / f'{name}_policy_candidate.json') for name in names} if validation else {
        'frozen': base.read(OUT / 'frozen_policy.json')['policy']}
    patients = {p: AnchorPatient(p) for p in ([6] if validation else [1, 2])}
    jobs = {}
    selections = {name: [] for name in names}
    for p, patient in patients.items():
        for stage, (checkpoint, queries) in enumerate(patient.groups):
            if validation and checkpoint not in (20, 40, 60, 80):
                continue
            anchors = len(patient.view(stage)['anchors'])
            for name, policy in policies.items():
                action = choose(policy, anchors)
                for q in queries:
                    key = (p, q['query_id'], action)
                    jobs[key] = (patient, stage, checkpoint, q, action)
                    selections[name].append({'persona': p, 'query_id': q['query_id'], 'action': action,
                                             'checkpoint': checkpoint})
    base.write_json(OUT / f'{split}_selections.json', selections)
    for (p, qid, action), job in jobs.items():
        if validation and action == 1:
            for suffix in ('.json', '.answer.json'):
                source = OUT / f'validation_fixed/p{p}/{qid}_a{action}{suffix}'
                target = OUT / f'{split}/p{p}/{qid}_a{action}{suffix}'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: job[0].retrieve(job[1], job[3]), jobs.values()))
    print(f'Prepared {len(jobs)} {split} query/action pairs', flush=True)
    client = base.OpenAI(api_key='local', base_url='http://127.0.0.1:19106/v1', timeout=900)
    scorer = base.Scorer('gpt-6.1-sol', 'en')
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(episode, *job, split, scorer, client=client)
                   for job in jobs.values()]
        for future in futures:
            future.result()
    if validation:
        summaries = {}
        for name, selection in selections.items():
            rows = [base.read(OUT / f'{split}/p{r["persona"]}/{r["query_id"]}_a{r["action"]}.json')
                    for r in selection]
            summaries[name] = {'n': len(rows), 'primary_correct': sum(r['score']['is_correct'] for r in rows),
                               'answer_tokens': sum(r['usage']['total_tokens'] for r in rows)}
        winner = max(names, key=lambda name: (summaries[name]['primary_correct'],
                                             -summaries[name]['answer_tokens']))
        frozen = {'selection_rule': 'Highest validation primary accuracy; ties use fewer answer tokens',
                  'selected_name': winner, 'validation': summaries, 'policy': policies[winner],
                  'training_personas': [3, 4, 5], 'validation_persona': 6, 'target_personas': [1, 2]}
        base.write_json(OUT / 'frozen_policy.json', frozen)
        print(f'Frozen {winner}: {summaries}', flush=True)
    else:
        print(f'Completed {len(jobs)} target answers', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['validation', 'evaluation'])
    run(parser.parse_args().split)
