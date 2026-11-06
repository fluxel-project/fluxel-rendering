# Native integration tooling

This directory owns the runnable native-consumer procedures and their validation
policy. The consumers live under `crates/renderer/examples/` with independent
Cargo manifests and lockfiles; they exercise public Renderer, Shader, Material,
RHI, Host, and Bases interfaces without adding workspace-private access.

`manifest.schema.json` defines retained-attempt records. `policy.md` defines
what an attempt may claim and the diagnostic, source-closure, capture, and
review rules. The tools retain inputs and observations for review; they do not
by themselves make a product-support or release decision.

## Consumer locations

| Consumer | Manifest | Purpose |
| --- | --- | --- |
| Native integration | `crates/renderer/examples/rendering-integration/Cargo.toml` | Native DX12 or desktop WGL workload, hidden/offscreen checks, and bounded visible demos. |
| External reference | `crates/renderer/examples/external-renderer/Cargo.toml` | Public CPU facade and private-API compile-fail contracts. |

Run CPU contracts without opening a demo window:

```powershell
cargo +1.87.0 test --manifest-path crates/renderer/examples/rendering-integration/Cargo.toml --features dx12 --locked
cargo +1.87.0 test --manifest-path crates/renderer/examples/rendering-integration/Cargo.toml --features native-gl-wgl --locked
cargo +1.87.0 test --manifest-path crates/renderer/examples/external-renderer/Cargo.toml --locked
python -m unittest discover -s tools/integration -p "test_*.py"
```

These commands check executable contracts and tooling. They do not create a
native device merely because a backend feature is enabled in a test target.

## Visible demos and hidden workloads

Run a persistent real-window demo on Windows with `--demo-forward` or
`--demo-deferred`:

```powershell
cargo +1.87.0 run --manifest-path crates/renderer/examples/rendering-integration/Cargo.toml --features dx12 -- --demo-forward --output target/demo-forward
```

Select `native-gl-wgl` for desktop WGL. A visible demo is placed behind existing
windows without activation; the user may select it normally. Offscreen workloads
create required WGL context windows hidden from their first creation. WGL visible
routes map the background drawable before their first render; DX12 can render
before reveal.

For deterministic demo evidence, set `FLUXEL_DEMO_EVIDENCE=1`. The demo writes
its inputs and readbacks, captures window frames 0, 5, and 10, then exits.
`demo_review.py DEMO_OUTPUT --report REVIEW_JSON` independently checks the
retained geometry, texture, transform, and tint against raw and window pixels.
Only the bottom two 16x16 DWM decoration corners are excluded from a schema-2
window comparison; raw readbacks have no decoration exclusion. Always inspect
the retained window images as well. A successful readback is not a substitute
for visible-window validation.

## Native attempt runner

Run one selected workload from the repository root:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
python tools/integration/smoke.py --backend dx12 --workload s5-forward
python tools/integration/smoke.py --backend dx12 --workload s5-surface
python tools/integration/smoke.py --backend dx12 --workload s5-device-isolation
python tools/integration/smoke.py --backend dx12 --workload s6-composed
python tools/integration/smoke.py --backend dx12 --workload s6-surface
```

Repeat a selected workload with `native-gl-wgl` where that backend is in scope.
The formal Deferred runner is separate:

```powershell
python tools/integration/s7_smoke.py --backend dx12 --workload s7-deferred
```

Use `RUSTUP_TOOLCHAIN=1.87.0` for MSRV runs. Every attempt has its own unique
directory under `target/integration/<workload>/<revision>/attempts/`; it records
the exact source input set, consumer and workspace locks, executable hash,
commands, raw diagnostics, and output inventory. An attempt retains a failed
snapshot and logs before any build or oracle failure. Source edits during an
attempt invalidate it instead of being attributed to the built binary.

The source snapshot includes source packages, this README, the manifest schema,
the policy, and relevant Cargo configuration. It excludes `target`, Git metadata,
local `draft`, local agent instructions, and credential-like files. Build-affecting
overrides, unsafe links/includes, build scripts, or Cargo configuration outside
the source closure reject the attempt rather than silently expanding its claim.

The executed binary is copied into its own attempt directory and re-hashed after
execution, so a later backend build cannot overwrite the retained artifact.

## Diagnostics, capture, and review

Every D3D12 diagnostic fails an attempt. With validation enabled, the runner
requests `FLUXEL_DX12_DEBUG` and requires an activation marker after device open.
The WGL route reads the selected RHI context's KHR_debug log through a scoped
native-interop lease and retains it as `wgl-debug-observer.json`. Missing or
malformed activation evidence remains unverified; a clean log alone does not
prove that observation was active.

Application debug markers and driver/API diagnostics remain separate. The former
are retained as instrumentation; every other native message is a failure.
Unknown or saturated WGL log representations fail closed. Window capture never
raises, activates, or makes a window topmost. A held frame must be requested
explicitly with `--frame-hold-ms`; arbitrary parent `FLUXEL_*` variables are not
inherited by runtime children.

Review tools consume retained facts, not self-reported success. A visual verdict
must link each image to its capture metadata and oracle; a test verdict must link
to a nonempty retained log, an executed command, and a nonzero test count. Raw
inputs and readbacks must exist before an oracle is evaluated. See `policy.md`
for the detailed fail-closed contract and `manifest.schema.json` for the record
format.
