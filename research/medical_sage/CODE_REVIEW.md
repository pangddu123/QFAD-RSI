# Clinical navigation code review

Initial training runner is active on 2A100 GPU1. The recorded policy step is 0; no optimizer update is yet established by an update artifact.

## Verified behavior

- Writer observations contain original spans and earlier facts, with no query or reference answer.
- Source-supported facts are admitted individually. Relations require source support and available endpoints. A rejected relation leaves its supported endpoints intact.
- Re-extracting a source replaces its facts and source-owned relations, including relations between facts from other sources.
- A relation cannot reinterpret a replaced fact identifier as a newly generated fact.
- QFAD includes relation evidence sources, returns exact original windows, and excludes sources beyond the query checkpoint.
- Query utility is computed on the admitted graph. Correct answers without sufficient evidence receive no success credit. Source errors remain per-atom feedback rather than a constant batch penalty.
- The policy loss uses causally aligned generated-token targets. Prompt-only and final unused logit positions receive no direct loss gradient.

## Evidence

Six navigation regression cases pass, including local rejection, dependency removal, replacement identity, source-owned relation replacement, relation-source retrieval, and future-source exclusion. Five reward tests pass for evidence-dependent success, equal-reward groups, local error aggregation, task-signal qualification, and shared judgements for identical evidence.

A CUDA check on 2A100 exercised the actual `policy_worker.logps` function with controlled logits: action alignment and gradient positions passed, and parameters remained unchanged. The existing Qwen27B NF4 forward/backward environment witness remains applicable; this check made no optimizer update.

Saved Qwen27B extraction outputs contain 37 facts and 12 relations. Source verdicts support 34 facts and 7 relations. Individual admission preserves those 34 facts and 7 relations. Three real embedding/QFAD/reranking queries return nonempty original windows, reconciled against source text and checkpoint visibility. This is saved-output replay, not fresh policy generation or an accuracy gain.

Artifacts: `results/medical_sage/extraction_probe/`, `results/medical_sage/admission_replay/`.

## Training path

`train_extractor.py` samples four complete extractions from the same source-only observation, admits facts and relations individually, evaluates a fixed query set through QFAD and the frozen answerer, and calculates clinical task utility with a bounded rejected-atom penalty. A group updates only when task utility differs among parsed candidates. It checks source arrival order and source-anchor visibility before launching the worker. The next training state uses the first sampled candidate independently of reward.

The semantic evidence-coverage calibration completed all 14 predeclared cases with zero mismatches, using gpt-6.1-sol through the service's Responses API. `results/medical_sage/clinical_calibration/summary.json` records the result. A JSON result is requested in the prompt and parsed locally.

Active run: `results/medical_sage/extractor_training_paired/`. Its saved first observation contains only sources and existing facts. The reviewed worker file exactly matches the deployed worker. GPU1 had 53,742 MiB free before launch; the protected Qwen32B service remains on GPU0.

Required runtime evidence: sampled task-reward variation, a finite real gradient/update with a changed adapter, and generation/evaluation from the resulting policy. Full patient memory formation and all 197 questions remain outstanding.

## Limitations

The bounded QFAD solver reached 500 iterations in the real replay. Source reachability and verbatim window delivery passed; convergence and retrieval quality at full graph scale are not established. Three source blocks do not represent complete patient memory. Source support is an LLM judgement, not clinical validation.
