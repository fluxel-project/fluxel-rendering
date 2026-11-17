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
local planning/evidence directories, local agent instructions, and credential-like files. Build-affecting
overrides, unsafe links/includes, build scripts, or Cargo configuration outside
the source closure reject the attempt rather than silently expanding its claim.

The executed binary is copied into its own attempt directory and re-hashed after
execution, so a later backend build cannot overwrite the retained artifact.

## Diagnostics, capture, and review

### Material development attempts

`m0_sampled_attempt.py` retains a source ZIP, verifies its actual contents against
the live source inventory before execution, and rejects source changes during an
attempt. Select the material workload explicitly. The E0 example uses its own
independent workspace lock and requires a local, hash-checked texture:

```powershell
python tools/integration/m0_sampled_attempt.py --backend dx12 --workload e0-texture-material-forward-deferred --asset ../fluxel-rhi/crates/rhi/examples/assets/textures/metalplate01_rgba.ktx --attempt-dir target/e0-texture-material-source-bound-01
python tools/integration/m0_sampled_attempt.py --backend dx12 --workload e0-capture-material-forward --dragon ../fluxel-rhi/crates/rhi/examples/assets/models/chinesedragon.gltf --plane ../fluxel-rhi/crates/rhi/examples/assets/models/plane.gltf --attempt-dir target/e0-capture-material-source-bound-01
```

Both examples accept `dx12` and `native-gl-wgl`. The capture workload
`e0-capture-material-surface` selects bounded visible mode with real window
capture and a same-input offscreen companion. Capture acceptance uses two producer-color controls,
each submitted once through the same Graph producer/receiver path. Model files
remain external, hash-checked inputs. The 04 dynamic-uniform consumer uses
`e0-dynamic-uniform-material-forward` or `e0-dynamic-uniform-material-surface`
with its own lock and no external asset arguments; the Surface selector requests
`--visible --exit-after-capture`. Neither example proves complete PBR or
refraction support.

The selected specular IBL plane controls use
`m1-specular-ibl-forward-deferred` or `m1-specular-ibl-surface`, resolving the
`pbr-environment-material` workspace and its actual lock. The corresponding
ORM/IBL sphere consumer selectors are `m1-orm-ibl-sphere-forward-deferred`
and `m1-orm-ibl-sphere-surface`, resolving `pbr-sphere-material`. These selectors
take no external asset arguments; each Surface selector bounds execution with
`--exit-after-capture`. A selector does not itself claim consumer acceptance.
`m1-orm-ibl-multi-object-forward-deferred` and
`m1-orm-ibl-multi-object-surface` select the same sphere workspace with
`--multi-records`, retaining per-object sources and transforms instead of
claiming a combined mesh is a multiple-record test.

`m1-srgb-filter-forward-deferred` and `m1-srgb-filter-surface` select the 02
workspace with `--srgb-filter` and its own lock. They use retained synthetic
sRGB bytes, require no external asset, and test decoding before filtering with
an independent numeric oracle. The Surface selector also requests
`--exit-after-capture` for bounded real-window capture and same-input companion
comparison. This input-color test does not prove HDR output or tone mapping.
`m1-material-bindings-forward-deferred` and `m1-material-bindings-surface` use
the same synthetic input with `--permuted-bindings`, selecting texture zero,
sampler one, and dynamic tint two. They prove location resolution independently
of the subsequent multiple-texture source model.
`m1-four-texture-forward-deferred` and `m1-four-texture-surface` select the
`pbr-four-texture-material` workspace and its actual lock. They require no
external files and retain each nonconstant image and logical-to-physical source
mapping. The Surface selector bounds real-window execution with
`--exit-after-capture`; both pipelines use independent numerical predictions.

These development records retain and re-hash the actual executable before and
after its direct run. `--validate` additionally requests DX12 debug activation
before device creation or requires the WGL context's returned KHR_debug marker,
retains the activation record and its hash, and rejects all native diagnostics.
Without this flag diagnostic status remains unverified. The bounded Surface
selectors still require inspection of the independently captured window images;
these records do not replace the complete formal smoke contract above.
A source-stable offscreen attempt proves only its named
numerical workload, not milestone closure. Never reuse a nonempty attempt directory.

### Formal diagnostics

Every D3D12 diagnostic fails an attempt. With validation enabled, the runner
requests `FLUXEL_DX12_DEBUG` and requires an activation marker after device open.
The WGL route reads the selected RHI context's KHR_debug log through a scoped
native-interop lease and retains it as `wgl-debug-observer.json`. Missing or
malformed activation evidence remains unverified; a clean log alone does not
prove that observation was active.

Application debug markers and driver/API diagnostics remain separate. The former
are retained as instrumentation; every other native message is a failure.
Unknown or saturated WGL log representations fail closed. Window capture never
raises, activates, or makes a window topmost. Windows Graphics Capture disables
cursor capture, queries the session setting, and records that fact. It never
moves the pointer or masks its pixels to repair a comparison. A held frame must be requested
explicitly with `--frame-hold-ms`; arbitrary parent `FLUXEL_*` variables are not
inherited by runtime children.

Review tools consume retained facts, not self-reported success. A visual verdict
must link each image to its capture metadata and oracle; a test verdict must link
to a nonempty retained log, an executed command, and a nonzero test count. Raw
inputs and readbacks must exist before an oracle is evaluated. See `policy.md`
for the detailed fail-closed contract and `manifest.schema.json` for the record
format.

`m1-mixed-capture-forward` and `m1-mixed-capture-surface` select the same
four-texture workspace with `--mixed-capture`. The Forward-only matrix combines
three immutable receiver images with the actual same-Graph producer target as
emissive. Producer-only and receiver-only source controls have separate asset
identities; CPU reference images remain oracle artifacts and are never uploaded.
Both selectors retain the actual standalone lock and executable. The surface
selector adds bounded real-window capture and an independently checked companion.

`m1-material-mips-forward-deferred` and `m1-material-mips-surface` select the
four-texture workspace's `--mip-generation` mode. Five source/explicit-LOD controls
exercise three generated levels from one base payload, with an independent
numerical oracle and both Forward/Deferred routes. The surface selector adds
bounded real-window captures and same-input companions. It establishes the
initial generated-chain boundary, not the complete moving-tunnel/filter matrix.

`m1-mipmap-tunnel-forward-deferred` and `m1-mipmap-tunnel-surface` select the actual
RHI-09 geometry/base-image adaptation. Supply `--model` and `--asset`; the standalone
consumer uses its own lock, implicit `textureSampleBias`, named mirror/trilinear
sampler and CPU perspective-derivative oracle. The bounded Surface selector adds
actual WGC images and independent same-input companions. It does not certify the
original Phong lighting/animation, aniso-16 quality, or all filtering controls.
