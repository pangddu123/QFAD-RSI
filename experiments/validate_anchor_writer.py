"""Run the patient-6 anchor-index validation with fixed source-window size."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from qfad_anchor_writer import AnchorPatient, episode, OUT, base

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoints',nargs='+',type=int,default=[20,40,60,80])
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    patient=AnchorPatient(6);jobs=[]
    for stage,(checkpoint,queries) in enumerate(patient.groups):
        if checkpoint not in args.checkpoints:continue
        patient.view(stage)
        jobs.extend((stage,checkpoint,q) for q in queries)
    if args.prepare:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda job:patient.retrieve(job[0],job[2]),jobs))
        print(f'Prepared {len(jobs)} validation queries',flush=True)
    else:
        scorer=base.Scorer('gpt-6.1-sol','en')
        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(lambda job:episode(patient,job[0],job[1],job[2],1,'validation_fixed',scorer),jobs))
        print(f'Completed {len(jobs)} validation answers',flush=True)

if __name__=='__main__':main()
