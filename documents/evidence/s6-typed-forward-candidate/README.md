# Retained S6 evidence

The [acceptance record](../../s6-typed-forward-closure.md) owns the bounded
stage decision; the archive is supporting evidence, not a release decision.

`index.json` hashes `native-evidence.zip` and its 255 members: four selected
DX12/WGL offscreen and Surface attempts, exact source snapshots, binaries,
commands, targeted test logs, raw inputs/expected/readback files, native
diagnostics, and the independent review. Its original ledger manifests may
also mention unselected attempts; the index identifies the four directories
actually retained here. Earlier failed attempts remain in the local ledger.

`baseline-index.json` hashes `baseline-evidence.zip` and its 800 members:
two five-sample release captures, raw process outputs, build/runtime logs,
source snapshots and release binaries. `source-reconciliation.json` and
`staged-source-check.json` document the subsequent publication check.

The visual index identifies Codex assistant inspection of all 22 offscreen
images, four native captures, and their four companion readbacks. It does not
claim human approval. Surface captures observe the live HWND through GDI;
companion readbacks are separate submissions with the same snapshot/input.

ZIP entries retain their original bytes. Check the archive and every member
hash before interpreting their contents. Nested source snapshots also retain
their complete per-file inventory. Absolute paths are original provenance;
they are not commands or a requirement to recreate those machine directories.
The index maps original paths to archive members.

The reviewer ran against the captured worktree. Replaying it in another
checkout requires restoring that source and mapping the retained paths.
Subsequent publication documentation must be reconciled separately; it must
never rewrite these artifacts or justify unverified product changes.

This record does not establish other native/browser targets, general GPU
Feature products, arbitrary cross-pass extensions, a performance improvement,
or the final full regression. Those remain separate roadmap requirements.
