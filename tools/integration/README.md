# Initial foundation integration tooling

The independent `rendering-integration` executable has been rewritten for the
Copy, offscreen Raster, supported Compute, and fixed Surface presentation
fixtures. Their fixed input schemas do not define the future Geometry,
Feature, FramePipeline or Material model. Current native artifacts and logical
regression results are linked by the
[foundation closure review](../../documents/s0-s1-closure.md), with source snapshots
kept distinct from historical attempts.

Copy, Raster, and Compute submit three immutable generations before
their first completion wait. The executable checks all readback bytes and structured
empty/capacity/foreign/stale refusals, Raster image pixels, all transformed
Compute words, and a presented Surface frame after verified HWND resize/restore.
Original inputs, WGSL and readbacks are retained before oracles. The read-set
fixture adds uniform-window changes and non-white sampled texels over the same
Graph/cache/completion path; device replacement exercises both native routes.

Run one selected backend from the repository root:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
```

Use `RUSTUP_TOOLCHAIN=1.87.0` in the process environment for MSRV runs. Each
attempt builds with the independent lock and explicit features, checks Cargo's
actual source identities, hashes source inputs and the executable, and retains
raw output under `target/integration/s0-s1/<revision>/attempts/`. The index links
all attempts, including a missing consumer manifest before any build starts.
Concurrent source edits invalidate an attempt. Diagnostic classification records
the hash of this current policy; the historical ledger cannot grant waivers.
Each new attempt also freezes its exact worktree inputs in `source-inputs.zip`,
including new files and both locks. The archive manifest and every member hash
are checked independently by the reviewer. A capture-time edit is retained as a
failed attempt rather than attributed to the binary. The archive identifies a
source snapshot; a dirty Git revision alone is not a frozen input.
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
a successful workload oracle closes S0/S1.
Presented-window inspection, Graph binding checks and fault evidence must be
linked explicitly in the closure review. Physical device-removal recovery and a
general public Graph/Material API are not implied by these fixtures.

For an attended presented-window capture, set
`FLUXEL_INTEGRATION_FRAME_HOLD_MS` to a bounded timeout. After each completed
Surface frame the executable writes `surface-frame-<n>.ready.json`, continues
to pump Host events, and waits for the named `.release` file. Capture the HWND
while it is stable, then create that release file to advance through the
initial, resized, and restored frames. These capture files are review inputs;
they do not replace the raw Surface lifecycle or diagnostic evidence.

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
