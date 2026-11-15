"""Retain rewarded graph-evidence actions for repeated queries."""
import argparse
from pathlib import Path
from qfad_fact_graph import base
from rsi_reader_policy import ReaderPolicy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--groups',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    policy=ReaderPolicy();policy.load(args.policy)
    retained=[]
    for directory in args.groups:
        group=base.read(directory/'group.json')
        semantic={}
        if (directory/'semantic_feedback.json').exists():
            for row in base.read(directory/'semantic_feedback.json')['groups']:
                semantic.update({name:row['verdict']['correct'] for name in row['names']})
        successful=[]
        for name in ('baseline','sample0','sample1','sample2','sample3','updated'):
            row=base.read(directory/f'{name}.json')
            if row['score']['is_correct'] and semantic.get(name,True):
                quality=1.+.2*row['query_likelihood']['token_probability']
                successful.append((quality,name,row))
        if not successful:
            continue
        quality,name,row=max(successful,key=lambda item:item[0])
        documents=[group['retrieval']['candidate_documents'][i] for i in row['candidate_indices']]
        policy.remember(group['episode_id'],documents,quality)
        retained.append({'episode_id':group['episode_id'],'feedback_file':str(directory/f'{name}.json'),
                         'facts':len(documents),'reward':quality,
                         'policy_memory_contains':'graph facts and cited source evidence'})
    policy.save(args.output)
    base.write_json(args.output.with_suffix('.json'),{'retained_paths':retained,
                    'path_count':len(policy.paths),'answer_text_stored_in_policy':False})
    print('Retained evidence paths',len(policy.paths),flush=True)


if __name__=='__main__':main()
