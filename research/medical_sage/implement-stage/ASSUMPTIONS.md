# Assumption Ledger

ASK mode: never

| ID | Decision | Chosen | Class | Source |
|---|---|---|---|---|
| A-001 | Writer role | Source-only batch extraction of facts and relations; queries and labels excluded from its observation | semantic | user |
| A-002 | Memory purpose | Facts navigate to original dialogue; answer context contains original evidence | semantic | user |
| A-003 | Model and compute | Qwen3.8-27B, 2A100, preserve Qwen32B; fixed Qwen27B answerer | interface | user |
| A-004 | Training population | Patients 1/2 are reusable training environments; patient 3/4 old-policy transfer remains separate | semantic | user |
| A-005 | Extraction unit | Existing fixed source chunks; no learned chunk-size action | interface | default |
| A-006 | Graph state | Retain source-scoped facts and cited relations, preserve earlier clinical events; rebuilding a chunk replaces its own extraction | interface | default |
| A-007 | Retrieval | QFAD over entity, fact and source nodes; relation edges participate in propagation; return cited source windows | interface | default |
| A-008 | Initial extraction policy | Fresh Qwen27B base adapter, rather than the previously trained patch policy | semantic | default |
| A-009 | Learning objective | SAGE-style group comparison with fixed readout; clinical evidence requirements and supported answer success as specified in METHOD.md | semantic | default |
| A-010 | Scope of early probes | Real-inference and reward-calibration probes establish execution only; complete patient evaluation is still required | semantic | default |
| A-011 | Admission unit | Admit source-supported facts individually; admit supported relations only when both endpoints survive. Retain rejected items and reasons separately. No batch-wide rejection for a local semantic error | semantic | user |
| A-012 | Saved-output replay | Replay saved generations and their saved source verdicts to verify admission and navigation; this is not regenerated-policy performance or a training result | semantic | default |
| A-013 | Feedback granularity | Compute query utility on the admitted graph; retain source errors per atom. Source-error credit assignment must be calibrated before parameter training; an entire batch is not assigned a constant negative reward | semantic | default |
| A-014 | First extraction objective | Task utility minus 0.25 times the proportion of proposed facts/relations not admitted. This bounded auxiliary penalty cannot outweigh one additional fully supported correct answer; its coefficient remains an experimental choice | semantic | default |
| A-015 | Update evidence | Require task-utility variation among at least two parsed candidates from the same source and pre-write graph. Formatting or source-error variation alone does not authorize an update in this experiment | semantic | default |
| A-016 | Initial training run | Three existing patient-1/2 source blocks establish the batch-extraction training path. Query sets and evidence requirements are fixed before sampling; complete-patient evaluation remains required | semantic | default |
| A-017 | Paired scoring | Identical answer prompts and identical delivered-evidence scoring inputs share a single answer/judge realization across candidate comparisons | semantic | default |
| A-018 | Full-memory inference | Use Qwen27B FP8 serving for both untrained and trained Writer evaluation, with the trained LoRA loaded for the latter. Keep NF4 sampling and optimization together during training. Process each patient's sources in arrival order | interface | default |
| A-019 | Source input budget | Keep up to eight recent earlier facts; remove the oldest supplied facts if the native prompt exceeds 12,000 tokens. Source chunk boundaries and the 3,072-token generation budget remain fixed | interface | default |
