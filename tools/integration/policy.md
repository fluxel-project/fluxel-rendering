# Integration evidence policy

This policy describes current runner obligations. It is not a support matrix,
an acceptance record, or permission to reuse a historical diagnostic waiver.

## Scope and identity

The independent native consumer lives in
`crates/renderer/examples/rendering-integration`. Its standalone manifest and
lockfile are separate from the library workspace and the external public-API
consumer. An attempt records the actual toolchain, executable, source inputs,
all relevant locks, selected Git dependency checkouts, backend and device facts.
Dirty source inputs need an exact frozen snapshot; a commit name alone cannot
identify them. Changes during capture invalidate the attempt.

Each attempt uses a new directory. A failed attempt retains its inputs, expected
facts, raw output and diagnostics rather than replacing an earlier result.
Runtime assets and independently derived expectations are saved before running
the oracle. Source archives exclude credentials, local drafts and build output.
Unknown or unclosed build inputs are refused by the runner instead of attributed
to a supposedly reproducible source snapshot.

## CPU and native execution

Library, compile-fail, consumer and Python tests prove their exercised CPU/API
contracts. Merely compiling a backend feature does not establish device support
or execute a graphics workload. Zero discovered tests add no behavior coverage.

Native workload acceptance additionally requires the actual selected backend,
complete nested workload inventory, raw readbacks, input-derived oracles and
native diagnostics. Device/Surface support comes from the selected RHI contract
and actual device facts. Do not infer it from a backend name or WGSL validity.
Completion failures retain pending owners; successful GPU completion remains
retirement evidence even if a later CPU map/readback operation fails.

Tests create any required host windows hidden. Visible demos use a real Surface,
appear behind existing windows without activation, and remain interactive when
the user selects them. Window capture must not activate, raise or temporarily
make a window topmost. Independent per-window capture and RHI readback are
different evidence sources. A locked or unavailable desktop leaves presentation
and screenshot acceptance unverified; offscreen pixels cannot replace it.

## Diagnostic admission

With DX12 validation enabled, the runner requires its device-activation marker
and retains all diagnostics. Driver/API diagnostics fail the workload; old
clear-value warnings are not current exemptions.

The WGL observer must prove activation on the actual RHI-owned context, retain
its raw log, and preserve observation across replacement contexts. Application
markers are separated only using source/type/severity metadata. Every other
message remains a native diagnostic and fails the attempt. Unknown decoding,
full queues, lost records, missing context observations and failed restoration
leave diagnostics unverified. Message text alone never grants a waiver.

## Verification and retained artifacts

The machine-readable manifest contract is [manifest.schema.json](manifest.schema.json).
Reviewers check hashes, recursive runtime artifacts, exact commands, nonzero
test counts, oracles, failure cases and visual verdicts as applicable. An
external verdict cannot approve unrelated files or replace raw records.
SHA-256 records detect accidental changes and mismatched inputs; they are not
signatures or protection against rewriting both an artifact and its manifest.

Performance claims additionally require representative inputs, independent
baseline/candidate runs, raw samples and counters, declared thresholds and small
workload guards. CPU/API elapsed time is not GPU duration, VRAM use or whole-frame
speedup. Preserve original artifact bytes and source identities when archiving;
an old result cannot certify current source merely because its pixels should
remain unchanged.
