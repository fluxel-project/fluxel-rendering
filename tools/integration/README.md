# Native integration evidence tooling

The independent `rendering-integration` executable has been rewritten for the
Copy, offscreen Raster, supported Compute, fixed Surface presentation, direct
Shader, and Material fixtures. The S2 direct-Shader raster and texture paths
retain their own source, reflection, route, expected pixels, and readbacks. The
S3 Material path separately exercises the original and replacement static
contributions, including a visible channel reordering, with independent
expected pixels and readbacks. Fixed S0/S1 fixture schemas still do not define
the future Geometry, Feature, or FramePipeline model.

The `--workload s5-forward` route invokes the executable with `--s5-only` and
writes a separate S5 closure. It hashes the root and consumer locks, freezes
the complete source-input set, records every nested `*workload.json` beneath
the run directory, and rejects a successful process that omitted required S5
input, expected-pixel, readback, cache, or replacement artifacts. Failed
attempts retain the same source snapshot, command logs and recursive artifact
inventory for diagnosis.

The `--workload s5-device-isolation` route invokes `--s5-device-only`. It
requires two simultaneously live devices to consume the same logical asset
snapshots, retains both device readbacks plus an old-device cache-reuse readback,
and rejects an attempt unless the new device records fresh static uploads and
zero completed-product hits. It has its own closure and never shares an attempt
directory with the four-layout scene workload.

The `--workload s5-surface` route invokes `--s5-surface-only`. It retains the
independent triangle expectation and inputs, each native client capture and
its verdict, plus the actual Host resize event and both Surface configurations.
The runner rejects a successful process unless the two captured frames cover
the initial and resized client extents and both visual verdicts passed.

`s5_review.py` reviews exactly six selected attempts: Forward, device isolation
and Surface for each backend. Pass `--attempt CLOSURE ID BACKEND WORKLOAD` once
per selection, `--human-review VISUAL_REVIEW_JSON` and `--output REVIEW_JSON`.
It checks frozen/current source, commands and nonzero focused tests, diagnostics,
raw inputs/readbacks, the independent pixel oracle, cache/device facts, measurement
facts and hash-bound visual review. The retained command is included in the
[S5 archive](../../documents/evidence/s5-forward-candidate/index.json).
`--human-review` and `human_verdict` are legacy schema names; the review record
must identify the actual reviewer. The selected S5 record explicitly identifies
Codex assistant image inspection and does not claim human approval.
The report verifies evidence; the [S5 closure record](../../documents/s5-forward-closure.md)
makes the bounded stage decision. Release timing uses the separate
[baseline tool](../../documents/s5-measurement-baseline.md).

The historical [S0/S1 support decision](../../documents/s0-s1-support-decision.md)
is retained and hashed only as a historical reference. It cannot approve S2/S3
support, native output, or any current stage decision.

Copy, Raster, and Compute submit three immutable generations before
their first completion wait. The executable checks all readback bytes and structured
empty/capacity/foreign/stale refusals, Raster image pixels, all transformed
Compute words, and a presented Surface frame after verified HWND resize/restore.
Original inputs, WGSL and readbacks are retained before oracles. The read-set
fixture adds uniform-window changes and non-white sampled texels over the same
Graph/cache/completion path. S2 texture and S3 Material workloads retain
separate inputs, expected output, and readback evidence; device replacement
exercises both native routes.

Run one selected backend from the repository root:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
python tools/integration/smoke.py --backend dx12 --workload s5-forward
python tools/integration/smoke.py --backend native-gl-wgl --workload s5-forward
python tools/integration/smoke.py --backend dx12 --workload s5-surface
python tools/integration/smoke.py --backend native-gl-wgl --workload s5-surface
python tools/integration/smoke.py --backend dx12 --workload s5-device-isolation
python tools/integration/smoke.py --backend native-gl-wgl --workload s5-device-isolation
```

Use `RUSTUP_TOOLCHAIN=1.87.0` in the process environment for MSRV runs. Each
attempt builds with the independent lock and explicit features, checks Cargo's
actual source identities, hashes source inputs and the executable, and retains
raw output under `target/integration/<workload>/<revision>/attempts/`, where
`<workload>` is `s0-s3`, `s5-forward`, `s5-surface`, or `s5-device-isolation`. The index links all attempts,
including a missing consumer manifest before any build starts.
Concurrent source edits invalidate an attempt. Diagnostic classification records
the hash of this current policy; the historical ledger cannot grant waivers.
Each new attempt also freezes its exact worktree inputs in `source-inputs.zip`,
including new files and both locks. The archive manifest and every member hash
are checked independently by the reviewer. A capture-time edit is retained as a
failed attempt rather than attributed to the binary. The archive identifies a
source snapshot; a dirty Git revision alone is not a frozen input.
The source closure includes package build scripts and the applicable workspace
and consumer `.cargo/config` or `config.toml` files, while excluding `target`,
Git metadata, local `draft` and `AGENTS.md` material, and all credentials files.
Build-affecting Rust/Cargo override variables are recorded only as safe hashes;
an override, configured explicit `CARGO_HOME`, or configured default Cargo home
is rejected rather than silently attributed to the archived source. Probable
secret files are never archived and instead fail the attempt without exposing
their paths or contents.
This runner does not claim a complete process-environment sandbox for Cargo
build scripts: any selected `build.rs`, dynamic `include_*` expression, or
root-escaping link/include is retained as a refusal rather than attributed to a
source snapshot. Transitive locked Cargo dependencies may also contain build
scripts; this local runner does not presently prove their complete environment
closure, so its source archive is not a reproducibility claim for those scripts.
Build and focused-test children receive a Windows/MSVC/Rust/Cargo toolchain
allowlist, not the parent environment; recorded values are hashes only. Runtime
children add only runner-owned diagnostics/evidence variables. For attended
capture, request the hold explicitly with `--frame-hold-ms`; inherited
`FLUXEL_*` variables are discarded.
The executed binary is copied into its own attempt directory and checked before
and after execution. A later backend build cannot overwrite that retained native
artifact; Cargo's original output path is recorded separately.

Every D3D12 diagnostic fails these workloads; historical clear-warning
waivers do not apply. With validation enabled, the runner requests
`FLUXEL_DX12_DEBUG` and requires a marker written after the selected DX12
device opens. The WGL route queries the exact RHI-owned context's KHR_debug
message log through a scoped native-interop lease and retains the result as
`wgl-debug-observer.json`. A missing or malformed marker is unverified and a
driver diagnostic still fails the attempt. WGL replacement-context observers
are checked recursively; a clean original-context log cannot cover another
context. Actual Cargo Git checkout revisions and cleanliness are verified in
addition to package-ID strings (only Cargo's `.cargo-ok` marker is ignored). Neither an empty diagnostic log nor
a successful workload oracle closes any stage. The report is integration
evidence only; stage-support and closure decisions require their own current
review.
Presented-window inspection, Graph binding checks and fault evidence must be
linked explicitly in the integration-evidence review. Physical device-removal recovery and a
general public Graph/Material API are not implied by these fixtures.
External review JSON is fail-closed: visual review must pair every retained
surface-frame BMP with its capture/verdict metadata and an explicit visual verdict;
test review must link a nonempty retained log to its exact command, nonzero
execution count, and zero exit status. Supply those runner-owned test records
with repeatable `--test-command-json '["python","-m","unittest",...]'` arguments;
the external review only names the recorded command index and log, never a
self-reported command result. Fault and read-set review must each link
the fixed mandatory raw scenario artifacts for every selected attempt. A file
that merely says `reviewed` or hashes unrelated artifacts cannot approve evidence.
These local hashes detect accidental edits and mismatched records. They are not a
signature, WORM store, or protection against an actor able to rewrite both the
attempt directory and its manifest.

For an attended presented-window capture, pass a bounded hold explicitly, for
example `--frame-hold-ms 30000`, to the smoke runner. It supplies
`FLUXEL_INTEGRATION_FRAME_HOLD_MS` only to the runtime child; arbitrary parent
`FLUXEL_*` variables are not inherited. After each completed Surface frame the
executable writes `surface-frame-<n>.ready.json`, continues to pump Host
events, and waits for the named `.release` file. Capture the HWND while it is
stable, then create that release file to advance through the initial, resized,
and restored frames. These capture files are review inputs; they do not replace
the raw Surface lifecycle or diagnostic evidence.

Development validation follows the local batch policy: finish a coherent
implementation batch, then select only the necessary affected crate/test checks.
Do not run a workspace or complete dual-backend matrix for each small change or
routine batch. Full regression is reserved for a completed major refactor or
complete goal implementation and its closure. The runner commands above are
available procedures, not an instruction to run a full matrix during this
implementation batch awaiting native verification.

KHR_debug messages identified by `DEBUG_SOURCE_APPLICATION`, notification severity,
and marker/push-group/pop-group type are retained in `applicationMarkers`. They
are application instrumentation, including RHI pass labels, rather than driver
diagnostics. Every other returned entry remains in `nativeDiagnostics` and fails
the attempt. Classification uses OpenGL metadata decoded strictly from the pinned glow
0.18.0 Debug representation (the entry type has no public field accessors).
Unknown representations remain fatal native diagnostics. Message text and
historical driver-warning exemptions never grant an exception. The observer's own activation marker is
also checked by source, type, severity, ID and exact message. Early queue entries
are preserved alongside the final drain, including when a workload fails.

WGL application push/pop-group notifications are disabled at log admission to
avoid consuming the bounded driver queue with pass labels. Other sources/types,
including driver/API errors, warnings and performance messages, remain enabled.
Every drain checks `MAX_DEBUG_LOGGED_MESSAGES`; a full, unknown-capacity or
non-progressing log records saturation and cannot pass validation. Saturation is
sticky across the old and final observations of each context.
