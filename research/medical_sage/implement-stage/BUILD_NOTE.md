# Build Note

| Rung | Feature | Acceptance | Status |
|---|---|---|---|
| F0 | Source-only Qwen27B extraction to graph artifact | Remote generation with exact source pointers | Generation complete; individual admission verified by saved-output replay |
| F1 | QFAD navigation returning original evidence | Real retrieval and source/visibility reconciliation | Three replay queries passed |
| F2 | Clinical query feedback | Source-verified requirements and controlled reward ordering | 14 semantic calibration cases and five reward tests passed |
| F3 | Extraction policy learning | Extraction-token update with frozen readout | Runner active; awaiting sampled task variation and update artifact |
| F4 | Full patient comparison | Complete patient-1/2 before/after generation and QA | Pending |

## Verification

- `test_medical_navigation.py`: six regression tests passed.
- `test_clinical_evidence.py`: five reward and paired-scoring tests passed.
- `test_policy_logps.py` on 2A100: causal action alignment and direct loss positions passed, parameters unchanged.
- `results/medical_sage/admission_replay/summary.json`: 34 facts, 7 relations, three nonempty original-evidence retrievals. Saved generations were not regenerated or edited.

## Deferred

Reader parameter updates follow the isolated Writer comparison and require separate evaluation.

## Blockers

No external blocker identified. The active training run must demonstrate task-relevant candidate variation and a real adapter update. QFAD convergence at the configured iteration bound is not established.

Active unit: `medical-sage-extractor-paired.service`; artifacts: `results/medical_sage/extractor_training_paired/`. Initial worker policy step: 0.
