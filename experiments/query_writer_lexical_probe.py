"""Validate lexical source-leaf recall under the frozen QFAD budgets."""
from concurrent.futures import ThreadPoolExecutor
import re
from query_writer_rl import Patient, Scorer, episode

def main():
    patient=Patient(6);scorer=Scorer('gpt-6.1-sol','en')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[]
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in (20,40,60,80):continue
            patient.view(stage)
            for q in queries:
                if re.search(r'\b\d{4}-\d{2}-\d{2}\b',q['question']):continue
                patient.candidates(stage,q)
                pending.append(pool.submit(episode,patient,stage,checkpoint,q,0,'lexical_probe',scorer,
                    max_evidence_nodes=15,query_method='lexical_leaf'))
        for future in pending:future.result()

if __name__=='__main__':main()
