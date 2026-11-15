"""Query-answer rewards for a source-only writer running on 2A100."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess
from qfad_fact_graph import FactGraph, base
from fact_writer_protocol import parse_output, fact_text
from medmemory_scoring import JudgeClient
from query_reward import QueryFeedback, group_rewards

OUT=base.ROOT/'results/qfad_fact_writer_rl_20261114'
REMOTE='/home/administrator/du/memos/fact_writer_rl_20261114'
SUPPORT_PROMPT='''Check whether each proposed graph fact is supported by its cited source text. Judge the subject, relation, object, negation, certainty, status and time together. An exact quotation alone does not prove support for the interpretation. A recommendation must not become a completed action; a suspected diagnosis must not become confirmed. Use only the supplied source. Return JSON with supported: a boolean list in fact order, and reason: a brief explanation of any unsupported facts. Do not answer any medical question.'''


class RewardController:
    def __init__(self, plan, writer_input, patient, dense_reward=False, reader_policy_path=None, graph_state=None):
        self.plan=plan
        self.sources=writer_input['sources']
        self.patient=patient
        self.dense_reward=dense_reward
        self.queries=[q for q in self.patient.groups[plan['stage']][1] if q['query_id'] in plan['query_ids']]
        self.embedding_cache={}
        visible={text for stage,text in patient.history if stage<=plan['stage']}
        visible_ids={source for source,text in zip(patient.stores['chunk']['ids'],patient.stores['chunk']['texts']) if text in visible}
        self.base_writes=[r for r in (graph_state or {}).get(str(plan['persona']),[]) if r['source_id'] in visible_ids]
        self.scorer=base.Scorer('gpt-6.1-sol','en')
        self.judge=JudgeClient('gpt-6.1-sol')
        self.query_feedback=QueryFeedback(OUT/'reader_cache'/plan['episode_id'],self.scorer,dense_reward)
        self.reader_policy=None
        if reader_policy_path:
            from rsi_reader_policy import ReaderPolicy
            self.reader_policy=ReaderPolicy()
            self.reader_policy.load(reader_policy_path)

    def retrieve(self, graph, q, cap):
        if self.reader_policy is not None:
            path=self.reader_policy.paths.get(self.plan['episode_id'])
            indices=graph.resolve_path(path['documents']) if path else None
            if indices is not None:
                documents=[graph.fact_document(i) for i in indices]
                context='\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(documents))
                if len(documents)<=15 and len(base.ENC.encode(context))<=cap:
                    return {'documents':documents,'fact_indices':indices,'method':'learned_evidence_path',
                            'reader_policy_weights':self.reader_policy.weights.detach().tolist()}
        retrieved=graph.retrieve(q['question'],anchor_source=self.sources[0]['source_id'])
        if self.reader_policy is not None:
            from rsi_reader_group import candidate_features
            features,costs=candidate_features(graph,retrieved,q['question'],q['query_type'])
            selection=self.reader_policy.replay(self.plan['episode_id'],retrieved['candidate_documents'],costs,cap) or self.reader_policy.select(features,costs,cap)
            retrieved['documents']=[retrieved['candidate_documents'][i] for i in selection.indices]
            retrieved['fact_indices']=[retrieved['candidate_fact_indices'][i] for i in selection.indices]
            retrieved['reader_policy_weights']=self.reader_policy.weights.detach().tolist()
        return retrieved

    def baseline(self):
        path=OUT/'graph_baselines'/f'{self.plan["episode_id"]}.json'
        if path.exists():return base.read(path)
        graph=FactGraph(self.patient,self.plan['stage'],self.base_writes,self.embedding_cache)
        answers=[]
        for q in self.queries:
            raw=next(r for r in base.read(base.RAW/f'persona_{self.plan["persona"]:02d}/session_{self.plan["checkpoint"]:03d}_answers.json') if r['query_id']==q['query_id'])
            cap=len(base.ENC.encode(raw['formatted_context']))
            retrieved=self.retrieve(graph,q,cap)
            context='\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(retrieved['documents']))
            context=base.ENC.decode(base.ENC.encode(context)[:cap])
            prompt=base.ANSWER_PROMPT.format(context=context,question=base.PROMPT.format_query(q['question'],q['query_type']))
            result=self.query_feedback(q,prompt,f'{self.plan["episode_id"]}_graph_baseline',len(retrieved['documents']))
            answers.append({'query_id':q['query_id'],'context':context,'retrieval':retrieved,**result})
        saved={'written_sources':[r['source_id'] for r in self.base_writes],
               'query_accuracy':sum(r['semantic_score']['correct'] for r in answers)/len(answers),
               'reward_metric':'fixed_semantic_rubric',
               'query_likelihood':sum(r['query_likelihood']['token_probability'] for r in answers)/len(answers) if self.dense_reward else 0.,
               'answers':answers}
        base.write_json(path,saved)
        return saved

    def evaluate(self, row, name):
        dest=OUT/'pilot'/name
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/'writer_output.json').exists():
            assert base.read(dest/'writer_output.json') == row
        if (dest/'reward.json').exists():
            return base.read(dest/'reward.json')
        base.write_json(dest/'writer_output.json',row)
        try:
            if not row['finished']:
                raise ValueError('Writer output truncated')
            replacement=parse_output(row['text'],self.sources)
        except (ValueError,KeyError,TypeError) as error:
            result={'reward_before_group_normalization':-1.,'query_accuracy':None,'valid':False,'reason':str(error)}
            base.write_json(dest/'reward.json',result)
            return result
        writes={r['source_id']:r for r in self.base_writes}
        writes.update({r['source_id']:r for r in replacement})
        graph=FactGraph(self.patient,self.plan['stage'],list(writes.values()),self.embedding_cache)
        base.write_json(dest/'graph.json',graph.manifest())
        proposed=[{'source_id':s['source_id'],**f} for s in replacement for f in s['facts']]
        payload={'sources':[{'source_id':s['source_id'],'text':s['text']} for s in self.sources],'facts':proposed}
        if (dest/'source_support.json').exists():
            supported=base.read(dest/'source_support.json')
        else:
            supported=json.loads(self.judge.chat(messages=[{'role':'system','content':SUPPORT_PROMPT},
                {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_tokens=4096,
                reasoning_effort='high',response_format={'type':'json_object'}).content)
        assert len(supported['supported'])==len(proposed) and all(type(v) is bool for v in supported['supported'])
        base.write_json(dest/'source_support.json',supported)
        answers=[]
        for q in self.queries:
            raw=next(r for r in base.read(base.RAW/f'persona_{self.plan["persona"]:02d}/session_{self.plan["checkpoint"]:03d}_answers.json') if r['query_id']==q['query_id'])
            cap=len(base.ENC.encode(raw['formatted_context']))
            retrieved=self.retrieve(graph,q,cap)
            context='\n\n'.join(f'Memory {i+1}:\n{doc}' for i,doc in enumerate(retrieved['documents']))
            context=base.ENC.decode(base.ENC.encode(context)[:cap])
            prompt=base.ANSWER_PROMPT.format(context=context,question=base.PROMPT.format_query(q['question'],q['query_type']))
            feedback=self.query_feedback(q,prompt,f'{name}__{q["query_id"]}',len(retrieved['documents']))
            saved={'query_id':q['query_id'],'context':context,'retrieval':retrieved,
                   'retrieved_facts':[graph.facts[i] for i in retrieved['fact_indices']],
                   'context_tokens':len(base.ENC.encode(context)),'raw_context_tokens':cap,**feedback}
            base.write_json(dest/f'{q["query_id"]}.answer.json',saved)
            base.write_json(dest/f'{q["query_id"]}.json',saved)
            answers.append(saved)
        accuracy=sum(r['semantic_score']['correct'] for r in answers)/len(answers)
        unsupported=sum(not value for value in supported['supported'])/max(1,len(proposed))
        write_cost=sum(len(base.ENC.encode(fact_text(f))) for f in proposed)/3072
        likelihood=sum(r['query_likelihood']['token_probability'] for r in answers)/len(answers) if self.dense_reward else 0.
        result={'reward_before_group_normalization':accuracy+.2*likelihood-.25*unsupported-.01*write_cost,'query_accuracy':accuracy,
                'query_likelihood':likelihood,'dense_query_reward':self.dense_reward,
                'unsupported_fraction':unsupported,'write_cost':write_cost,'valid':True,
                'written_facts':len(proposed),'query_correct':[r['semantic_score']['correct'] for r in answers],
                'reward_metric':'fixed_semantic_rubric'}
        base.write_json(dest/'reward.json',result)
        print(name,result,flush=True)
        return result


def main():
    global OUT,REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='')
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--greedy-every',type=int,default=1)
    parser.add_argument('--dense-query-reward',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--writer-gpu',type=int,default=0)
    parser.add_argument('--graph-baseline',action='store_true')
    parser.add_argument('--reader-policy')
    parser.add_argument('--graph-state',type=base.Path)
    args=parser.parse_args()
    if args.run:
        OUT=OUT/args.run
        REMOTE=REMOTE+'/'+args.run
    plans={p['episode_id']:p for p in base.read(OUT/'training_plan.json')}
    inputs={p['episode_id']:p for p in base.read(OUT/'writer_inputs.json')}
    patients={p:base.Patient(p) for p in {plan['persona'] for plan in plans.values()}}
    graph_state=base.read(args.graph_state) if args.graph_state else {}
    controllers={key:RewardController(plan,inputs[key],patients[plan['persona']],args.dense_query_reward,args.reader_policy,graph_state) for key,plan in plans.items()}
    resume=' --resume' if args.resume else ''
    command=['ssh','-T','win',f'ssh 2A100 env CUDA_VISIBLE_DEVICES={args.writer_gpu} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/administrator/du/memskill-determinacy-20261014/venv/bin/python -u {REMOTE}/fact_writer_worker.py --inputs {REMOTE}/writer_inputs.json --out {REMOTE}/pilot --steps {args.steps} --greedy-every {args.greedy_every}{resume}']
    with (OUT/'worker.log').open('a' if args.resume else 'w') as errors:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
        for line in process.stdout:
            packet=json.loads(line)
            event=packet['event']
            print('writer',event,packet.get('step'),flush=True)
            if event=='ready':
                base.write_json(OUT/('worker_resume_runtime.json' if args.resume else 'worker_runtime.json'),packet)
            elif event=='rollouts':
                base.write_json(OUT/f'rollouts_{packet["step"]}.json',packet)
                controller=controllers[packet['episode_id']]
                baseline=controller.baseline() if args.graph_baseline else None
                jobs=[(row,f'step{packet["step"]}_sample{i}') for i,row in enumerate(packet['rows'])]
                if packet['initial_greedy'] is not None:
                    jobs.append((packet['initial_greedy'],f'step{packet["step"]}_initial'))
                with ThreadPoolExecutor(max_workers=5) as pool:
                    values=list(pool.map(lambda job:controller.evaluate(*job),jobs))
                samples=values[:len(packet['rows'])]
                if baseline is None:
                    grouped=group_rewards(samples)
                    rewards=grouped['rewards']
                    grouped['advantages']=[(len(rewards)*r-sum(rewards))/(len(rewards)-1) for r in rewards]
                else:
                    components=[];rewards=[]
                    for sample in samples:
                        if not sample['valid']:
                            components.append(None);rewards.append(-1.);continue
                        delta=sample['query_accuracy']-baseline['query_accuracy']+.2*(sample['query_likelihood']-baseline['query_likelihood'])
                        if abs(delta)<.002:delta=0.
                        components.append(delta)
                        rewards.append(delta-.25*sample['unsupported_fraction']-.01*sample['write_cost'])
                    valid=[v for v in components if v is not None]
                    grouped={'rewards':rewards,'advantages':rewards,'query_reward_components':components,
                             'normalized_query_components':None,'query_signal_active':any(v!=0 for v in valid),
                             'query_signal_span':max(valid)-min(valid) if valid else 0.,
                             'baseline':{k:baseline[k] for k in ('query_accuracy','query_likelihood')},
                             'credit_assignment':'improvement_over_current_graph'}
                reply={'step':packet['step'],'episode_id':packet['episode_id'],**grouped}
                base.write_json(OUT/f'rewards_{packet["step"]}.json',{'initial':values[-1] if packet['initial_greedy'] is not None else None,'samples':samples,**grouped})
                process.stdin.write(json.dumps(reply)+'\n');process.stdin.flush()
            elif event=='update':
                base.write_json(OUT/f'update_{packet["step"]}.json',packet)
                if packet['post_update_greedy'] is not None:
                    controllers[packet['episode_id']].evaluate(packet['post_update_greedy'],f'step{packet["step"]}_updated')
            elif event=='complete':
                base.write_json(OUT/'pilot_complete.json',packet)
        status=process.wait()
        if status:
            raise RuntimeError(f'Writer worker exited with status {status}; see worker.log')


if __name__=='__main__':main()
