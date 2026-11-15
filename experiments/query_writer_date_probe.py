"""Validate source-date anchors for all explicitly dated validation queries."""
from concurrent.futures import ThreadPoolExecutor
import re
from query_writer_rl import Patient, Scorer, episode


def main():
    patient=Patient(6)
    scorer=Scorer('gpt-6.1-sol','en')
    for stage,(checkpoint,queries) in enumerate(patient.groups):
        if checkpoint not in (20,40,60,80):continue
        dated=[q for q in queries if re.search(r'\b\d{4}-\d{2}-\d{2}\b',q['question'])]
        patient.view(stage)
        print('Date-anchor validation',checkpoint,len(dated),flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda q:episode(patient,stage,checkpoint,q,0,'date_probe',scorer,query_method='date_anchored'),dated))

if __name__=='__main__':main()
