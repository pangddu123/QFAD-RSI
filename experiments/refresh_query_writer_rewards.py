"""Re-evaluate sampled writer actions affected by date-anchored querying."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re
from query_writer_rl import OUT, Patient, Scorer, episode, read


def main():
    groups=defaultdict(list)
    for p in (OUT/'train').glob('p*/*.json'):
        if p.name.endswith('.answer.json'):continue
        r=read(p)
        if re.search(r'\b\d{4}-\d{2}-\d{2}\b',r['question']):groups[r['persona']].append(r)
    scorer=Scorer('gpt-6.1-sol','en')
    for persona,rows in sorted(groups.items()):
        patient=Patient(persona)
        by_checkpoint=defaultdict(list)
        for row in rows:by_checkpoint[row['checkpoint']].append(row)
        for stage,(checkpoint,queries) in enumerate(patient.groups):
            if checkpoint not in by_checkpoint:continue
            qmap={q['query_id']:q for q in queries}
            patient.view(stage)
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda r:episode(patient,stage,checkpoint,qmap[r['query_id']],r['action'],'train_date',scorer,query_method='date_anchored'),by_checkpoint[checkpoint]))

if __name__=='__main__':main()
