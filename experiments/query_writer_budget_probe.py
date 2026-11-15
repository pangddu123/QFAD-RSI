"""Validation of evidence packing under the existing RAW-RAG token cap."""
from concurrent.futures import ThreadPoolExecutor
from query_writer_rl import Patient, Scorer, episode


def main():
    patient=Patient(6)
    stage=1
    checkpoint,queries=patient.groups[stage]
    assert checkpoint==20
    patient.view(stage)
    scorer=Scorer('gpt-6.1-sol','en')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda q:episode(patient,stage,checkpoint,q,0,'budget_probe',scorer,max_evidence_nodes=128),queries))

if __name__=='__main__':main()
