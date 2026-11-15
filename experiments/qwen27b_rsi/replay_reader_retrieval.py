"""Replay recorded Reader actions to identify changed retrieval observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from graph_environment import GraphEnvironment, base
from reader_environment import ReaderEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = base.read(args.run / 'graph_snapshot.json')
    paths = sorted((args.run / 'trajectories').glob('*.json'))
    pending = [p for p in paths if not (args.out / 'episodes' / p.name).exists()]
    personas = {int(p.stem.split('_', 1)[0][1:]) for p in pending}
    graphs = {p: GraphEnvironment(base.Patient(p, store_names=('chunk',)), state[str(p)]) for p in personas}
    questions = {f'p{p}_{q["query_id"]}': (p, stage, q)
                 for p, g in graphs.items()
                 for stage, (_, queries) in enumerate(g.patient.groups) for q in queries}

    def replay(path):
        output = args.out / 'episodes' / path.name
        if output.exists():
            return base.read(output)
        saved = base.read(path)
        name = saved['episode_id']
        p, stage, question = questions[name]
        graph = graphs[p]
        if not graph.visible_facts(stage):
            result = {'episode_id': name, 'identical': True,
                      'reason': 'No visible facts; diffusion is not invoked'}
        else:
            initial = json.loads(saved['messages'][1]['content'])
            episode = ReaderEpisode(graph, question, stage, initial['evidence_token_budget'],
                                    native_budget=saved['native_budget'])
            consumed = 0
            differences = []
            for i, packet in enumerate(saved['policy_packets']):
                if 2 + 2 * i >= len(saved['messages']):
                    break
                row = packet['rows'][0]
                consumed += packet['prompt_tokens'] + len(row['token_ids'])
                try:
                    episode.step(row, policy_tokens=consumed)
                except (ValueError, KeyError, TypeError) as error:
                    differences.append({'action_index': i, 'error': str(error)})
                    break
                old = json.loads(saved['messages'][3 + 2 * i]['content'])
                new = json.loads(episode.messages[-1]['content'])
                if old != new:
                    differences.append({'action_index': i, 'old': old, 'new': new})
                    break
            evidence = episode.evidence()
            changed_fields = [k for k in evidence if evidence[k] != saved['evidence'].get(k)]
            result = {'episode_id': name, 'identical': not differences and not changed_fields,
                      'observation_differences': differences, 'evidence_changed_fields': changed_fields}
        base.write_json(output, result)
        print(name, result['identical'], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(replay, paths))
    base.write_json(args.out / 'summary.json', {
        'run': str(args.run), 'checked': len(rows),
        'identical': sum(r['identical'] for r in rows),
        'changed': [r['episode_id'] for r in rows if not r['identical']],
        'comparison': 'All policy observations and final evidence, including retrieval inputs and costs'})


if __name__ == '__main__':
    main()
