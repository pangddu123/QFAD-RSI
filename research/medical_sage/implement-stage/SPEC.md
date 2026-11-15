# Clinical navigation Writer

Target: a Qwen3.8-27B extraction policy that forms reusable fact and relation indices from observed dialogue. QFAD navigates these indices and returns original evidence. Query feedback trains extraction outputs; the answer model stays fixed.

Inputs: the existing patient-1/2 source chunks, chronological visibility, and previously extracted facts. Queries and benchmark labels enter evaluation only.

Outputs: source-scoped extraction JSON, a graph with verbatim citations, original-evidence retrieval records, candidate rewards, actual adapter updates, and before/after evaluation. Artifacts live under results/medical_sage.

Success commands: the extraction probe, navigation probe, reward calibration, training runner and complete-cohort evaluator recorded in BUILD_NOTE.md.

Base commit: none (the workspace is not a Git repository).

Completion requires real inference, extraction-to-source navigation, calibrated medical evidence feedback, parameter updates, and evaluation at the requested patient scope. A working prompt or a small inference probe alone is insufficient.
