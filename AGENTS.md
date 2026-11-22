# Maintaining the BUD reproducibility package

- Preserve the complete-cohort KM–RMST target, the locked one-pass BUD design, the paired rescaled bootstrap, and the three published methods. Build partitions from phase-I covariates and fixed seeds. Use outcome data only where the protocol permits it, for the evaluation or a separate pilot.
- Keep the main entry point in `reproduce.py`. Add a file only when it owns a distinct stage or shared responsibility. Reuse the C and Python interfaces and retain the fallback.
- Explain scientific choices close to the code that implements them. Comment on allocation, tie handling, reference targets, random-number streams, and pilot/evaluation separation. Use docstrings for public inputs and outputs.
- Retain original inputs and raw outputs in the research archive. Keep downloads pinned and checked, the GitHub tree source-focused, and result claims reproducible from a fresh checkout. Never hand-edit generated results or discard unfavorable draws.
- Update the protocol, the CLI output map, and the result verification inventory together when the paper changes. Compare rebuilt outputs against the manuscript, update the expected values from that checked run, and run the full target and its verification before tagging a release.
- Compare C and Python results on relevant representative inputs after changes to numerical kernels or design operations. Record compiler and runtime separately from the scientific outputs.
- Keep the code repository separate from the journal supplementary document. Before packaging, remove local paths, private review files, generated caches, binaries, and irrelevant historical experiments.
