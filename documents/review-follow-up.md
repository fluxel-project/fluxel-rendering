# Rendering Review Follow-up Disposition

> Status: follow-up implementation, focused CPU checks, and fresh named-route
> evidence are reviewed. The resulting bounded closure is recorded in
> [S2/S3 named-Windows closure](s2-s3-closure.md). It is not a GPU-general,
> full-regression, or later-stage acceptance report.

## Scope and retained baseline

The reviewed S2/S3 implementation was committed in product
`8ddf2da072e7d48331e423a6921f4deaffadf279`. Two retained native attempts
exercised the then-exact archived working-tree inputs that became that product
implementation:

| Route | Attempt | Retained scope |
| --- | --- | --- |
| Windows DX12 | `dx12-20261103T035749288553Z-14744` | S2 direct shader and texture, S3 material and replacement, read-set, device replacement, capability snapshots, diagnostics, raw outputs, and reviewed presented frames |
| Desktop OpenGL through WGL | `native-gl-wgl-20261103T035908660151Z-24048` | The equivalent bounded workload and review inputs on the WGL route |

`target/integration/reviews/20261103-current/review.json` verified the
selected attempts' recursive artifacts, source/lock records, raw workload
facts, capability snapshots, native diagnostics, visual records, focused test
records, fault records, and read-set records. It deliberately makes no stage
support or closure decision. The attempts remain useful historical baseline
evidence for the named `8ddf2da` implementation only.

## Disposition

### S2/S3 native-evidence finding: fresh named-route closure

The earlier statement that S2/S3 had no current executable evidence is no
longer accurate for product `8ddf2da`: the two retained attempts above cover
the bounded S2/S3 consumer on both selected Windows routes. The independent
consumer is intentionally outside the root Cargo workspace, so root workspace
tests do not replace those attempts.

The follow-up changed the Bases revision and the Material source identity and
admission behavior, so historical output could not certify it. Fresh DX12/WGL
attempts now bind the dirty source, locks, executable, readbacks, diagnostics,
and reviewed archive. They close only the named S2/S3 fixture matrix described
in [the closure record](s2-s3-closure.md).

### Bases Stable-Clippy finding: accepted and upgraded

Bases commit `80ffaa8` introduced the waiter/file-loading work. The reported
`fetch_update` deprecation does not reproduce on Stable 1.98 or the Rust 1.87
MSRV line. The actual observed Stable-Clippy blocker was
`waiter_registration`'s `manual_noop_waker` finding. Bases
`ee640d57a8d7c32d251252d0e8b32950636a770c` addresses that waiter issue while
retaining the Rust 1.87 MSRV line. Rendering now pins that revision consistently
in the root and independent consumer manifests and locks.

This was an intentional dependency-input change. Fresh CPU and native validation
uses the new pin; attempts pinned to Bases
`c5fdaa993c13feb6b55b23fc766c22c9733836f2` remain historical only.

### Material namespace and source identity findings: implemented and reviewed

The old material compiler accepted some collisions among emitted WGSL
top-level names. It now has a single admission check spanning generated
`evaluate_material`, static options, resource/parameter-visible declarations,
and top-level declarations scanned from all fragment source. It returns a
structured error before composition when a name conflicts or safe scanning
cannot complete.

The material contribution source ID now uses a SHA-256 digest of the canonical
variant key rather than embedding complete material source text. The complete
canonical key remains the material cache identity, while the Shader composition
request continues to compare full source contents; a short source ID therefore
does not turn a digest collision into cache aliasing. Focused tests cover
generated-name, function, type, variable, override and resource collisions,
plus forced duplicate source-ID behavior.

These changes are source-visible to S3 shader composition. They invalidate the
historical S3 GPU claim even when ordinary fixture pixels are expected to be
unchanged.

### Shader capability-cost finding: accepted as measurement work

Capability inference still uses repeated Naga module validation to remove
unneeded capability bits safely. The recorded cold-composition baseline remains
measurement-only; it does not claim an optimization, speedup, or regression
threshold. Its original source/harness identity is retained rather than rewritten
to include later lint-only code.

### Pending snapshot type-erasure observation: retained deliberately

The fixture renderer retains submitted asset snapshots as
`Box<dyn Any + Send + Sync>` until completion evidence permits release. This is
not a Feature registry, frame-policy dispatch mechanism, or public extension
API: Feature composition remains static and monomorphized. Replacing this
private heterogeneous ownership holder without a demonstrated consumer-owned
retention model would broaden the current fixture work. Keep it under review
for the production submission ownership design; do not treat it as resolved or
as permission for trait-object extensibility.

## Required affected verification

Focused verification against the new Bases lock passed: 18 RenderScene tests,
one feature test, two external-consumer tests, seven Material library and seven
Material contract tests, 14 targeted Shader tests, and their applicable Clippy
checks. Test discovery was recorded; no zero-test exit is treated as coverage.

The minimal native verification used one fresh retained attempt per selected
route, using the existing independent runner:

```powershell
python tools/integration/smoke.py --backend dx12
python tools/integration/smoke.py --backend native-gl-wgl
```

Each attempt archives the updated source inputs and lockfile, records the
actual Bases `ee640d5` checkout, and retain the S2 shader/texture and S3
original/replacement material workload inputs, independent expected output,
raw readback, route/capability facts, replacement-device facts, and diagnostics
before its oracle. Re-review both attempts, including all presented-frame
captures and the focused/fault/read-set review inputs.

The S2 direct shader code itself was not changed by the Material fixes, so the
old attempts remain useful S2 scope evidence. They are nevertheless
insufficient for a new S2 decision because the independent executable and its
Bases dependency are rebuilt at a new exact input. No smaller GPU selection is
safe with the current runner: one route invocation deliberately exercises the
complete coupled S0--S3 workload and is the smallest available real-consumer
path. These focused native runs are not a workspace-wide regression; the
project-wide regression remains pending until all planned S stages are tested.

## Explicitly pending

- Generalization beyond the named DX12/WGL fixture matrix, full S-stage
  regression, and all later S4 GPU/S5/S6 delivery work remain pending.
- S4 remains a separately bounded CPU closure; this document adds no S4 GPU,
  Forward, FramePipeline, or general extension claim.
