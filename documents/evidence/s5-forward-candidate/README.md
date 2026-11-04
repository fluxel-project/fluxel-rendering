# Retained S5 evidence

The [closure record](../../s5-forward-closure.md) defines the supported scope.

- `index.json` hashes `native-evidence.zip` and each of its 289 members: six
  selected native attempts, raw workload manifests and independent review.
- `baseline-index.json` hashes `baseline-evidence.zip` and its 728 members:
  two five-sample release captures, raw measurements, source archives and binaries.
- `focused-checks/index.json` hashes scoped validation logs and the separate
  failed runner bootstrap log.
- `source-reconciliation.json` relates the two captures and lists subsequent
  documentation-only differences. `staged-source-check.json` records the
  publication check against Git's index, including line-ending normalization.

ZIP members retain original bytes, including nested source snapshots. Original
absolute paths in manifests are provenance, not executable instructions. The
index maps each original path to its archive member for an independent reviewer.
Verify archive/member hashes before interpreting the raw inputs, expected output,
readbacks, diagnostics and review report. Source snapshots contain their own
`snapshot-inputs.json` inventory and per-file hashes.

The recorded reviewer command used the original workspace paths and exact
captured source. Replaying that command in another checkout requires restoring
that snapshot and mapping the archived paths to the review location. Running
it directly against later source deliberately fails its current-source check;
the documented README follow-up does not rewrite the original attempt.

These records cover the selected S5 candidate and named machine/backend routes.
They do not establish a full historical regression, S6 composition, a performance
improvement, or support for untested targets. Assistant screenshot inspection is
identified explicitly in `visual-review.json`; its legacy `human_verdict` key
does not assert human approval.
