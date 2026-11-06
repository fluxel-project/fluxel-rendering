# Rendering Review Follow-up

## Current maintenance review

This review starts from Rendering `ec05ae4` and Bases `6c80772`. Stage names
retain the bounded meanings in [the implementation plan](version-plan.md).
Historical native and performance archives retain their original source and
dependency identities; they do not certify this follow-up's changed sources.

### Correctness and engineering gates

The reported texture access defect is valid. The mip partitioner included
endpoints from declarations outside the requested interval. A pass declaring
`0..1` and `3..4` could therefore reject a valid request for `0..1`. Only
nonempty intersections may partition the requested mip interval. Layer and
aspect coverage must still hold independently in every resulting partition;
separate declarations cannot fill a missing mip or layer by implication.
Regression tests exercise the real declared-access validator, including legal
disjoint declarations and illegal gaps, rather than only its interval helper.

The new [CI workflow](../.github/workflows/ci.yml) runs on pushes, pull requests
and manual dispatch. Windows jobs run workspace behavior and documentation
tests, the independent external consumer (including diagnostic-checked private
API compile failures), and DX12/WGL integration CPU contracts on Stable and
Rust 1.87. Stable additionally checks formatting and Clippy. Python discovery
checks the integration and benchmark tooling. These jobs create no GPU device
or visible demo and are not hardware acceptance. Branch-protection policy and
native hardware certification remain separate from installing the workflow.

Enabling these checks also exposed preexisting formatting differences and two
`cloned_ref_to_slice_refs` test warnings. The affected files are formatted and
the two test inputs now borrow their original descriptor without cloning.
The independent consumer's complex scene return type also has a local alias;
its behavior and public API are unchanged.

The Bases warning is real on GitHub's Stable 1.99.1, although local Stable
1.98 does not reproduce it. Run `42009617744` failed on `fetch_update` at
`6c80772`; the earlier S2/S3 review's toolchain-specific result was not a
current green-CI claim. Bases `124e825f82e5cd0eb485c95fe1281b11c8dbc844`
uses a relaxed compare-exchange loop instead, retaining unique claimed IDs,
contention retry and exhaustion rejection without requiring a post-MSRV API.
Its [CI run](https://github.com/fluxel-project/fluxel-bases/actions/runs/42018516642)
passes both quality and Rust 1.87 jobs.

Rendering selects that exact Bases revision in all four consumer manifests and
all five Cargo lockfiles, updated with targeted Cargo resolution. No unrelated
package versions change. RHI remains
`b5d569feace548a6ed498901deaf31a5efcaca61`; the native executable still selects
Host `8d101624c603a6c1e0badf729768c6b26e70fba8`. The runnable cross-repository
artifact is `examples/rendering-integration` (owned by Rendering); its DX12/WGL
CPU contracts and the independent external consumer validate the changed pin.
This review changes no backend recording, presentation or shader behavior and
does not claim refreshed GPU/performance certification for the new pin.

The Python aggregate failure was also valid: discovery imports `smoke_tests` as
a top-level package, where its relative parent import cannot resolve. The
support module now selects its import mode explicitly, preserving package
execution and making the aggregate command itself usable. Historical S8 logs
remain unchanged, including their originally recorded aggregate failure.

The first workflow run also exposed nine corresponding deprecated atomic
updates in Rendering itself, plus a clock-resolution collision in attempt
names. Rendering now uses one private MSRV-compatible relaxed CAS helper at
all nine identity/plan allocation sites, retaining each caller's exhaustion
error or panic. Concurrent uniqueness and unchanged-counter overflow tests
cover the helper. Attempt names include a process-local sequence in addition
to backend, UTC timestamp and PID; a frozen-clock regression verifies that
separate attempts cannot share a directory. Directory creation still refuses
an existing path, and failed raw output remains independently retained.

### Current maintenance validation

Local Rust 1.87 validation on the updated pins passes 312 workspace behavior
tests (one diagnostic remains intentionally ignored), 45 DX12 integration CPU
tests and 48 WGL integration CPU tests. Stable Clippy passes for all workspace
targets and the independent consumer; workspace formatting passes. Stable's
independent consumer passes ten behavior tests, including both private-API
compile failures. Rust 1.87 workspace doctests also exit successfully but
discover zero cases, so they add no behavioral coverage. Benchmark-tooling
discovery passes 25 tests. CI repeats the defined matrix on the pushed source;
its status must be read from the commit's workflow rather than inferred from
these local results.

The repaired aggregate integration command discovers 118 tests: 117 pass and
one Windows symlink-privilege case is skipped. Its successful aggregate run
replaces the need to combine separate package logs for this maintenance check.

### Review judgments and future work

| Finding | Judgment and next evidence |
| --- | --- |
| Fixed Feature/pass structure | Accurate limitation, already outside the bounded S6/S7 closure. Before expanding the facade, choose a real external producer and downstream pass consumer; prove declarations, current-frame cache-hit validation, failure cleanup and completion ownership through the same Graph. No new registry, scheduler, public Graph or speculative generic framework is justified by this review. |
| Large evidence archives | Git-tree bytes are dominated by retained evidence. Keep existing archives immutable. A future storage migration must first prove durable retrieval, source/lock identity and SHA-256 verification from a clean checkout. No archive or Git history is removed here. |
| Shader composition cost | Repeated Naga capability validation and frame-local composers are visible in source, but do not establish a performance regression. Measure representative variants, cold/warm composition and composer hits before choosing Renderer-level caching. No new performance claim or cache is introduced. |
| Forward/Deferred duplication | Similar preparation/cache paths alone do not establish a correctness problem or justify merging ownership protocols. Future reduction needs a concrete duplicated invariant and tests preserving admission, actual-owner facts, failure cleanup and retirement. |
| S8 measurement | Retain the selected CPU/API validation-index optimization and its workload-specific historical result. No GPU-duration, VRAM or whole-frame speedup is inferred. |

The original S0–S8 archives remain evidence for their named subsets, including
the fixed-light, one-mesh Deferred and selected position-deformation paths.
General pass chains, general GPU products, complete PBR, LOD, streaming and GPU
culling remain unimplemented or unaccepted as described in the plan. This
maintenance review does not authorize those as new implementation scope.

## Historical S2/S3 disposition

> Status: follow-up implementation, focused CPU checks, and fresh named-route
> evidence are reviewed. The resulting bounded closure is recorded in
> [S2/S3 named-Windows closure](s2-s3-closure.md). It is not a GPU-general,
> full-regression, or later-stage acceptance report.

### Scope and retained baseline

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

### Disposition

#### S2/S3 native-evidence finding: fresh named-route closure

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

#### Bases Stable-Clippy finding: accepted and upgraded

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

#### Material namespace and source identity findings: implemented and reviewed

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

#### Shader capability-cost finding: accepted as measurement work

Capability inference still uses repeated Naga module validation to remove
unneeded capability bits safely. The recorded cold-composition baseline remains
measurement-only; it does not claim an optimization, speedup, or regression
threshold. Its original source/harness identity is retained rather than rewritten
to include later lint-only code.

#### Pending snapshot type-erasure observation: retained deliberately

The fixture renderer retains submitted asset snapshots as
`Box<dyn Any + Send + Sync>` until completion evidence permits release. This is
not a Feature registry, frame-policy dispatch mechanism, or public extension
API: Feature composition remains static and monomorphized. Replacing this
private heterogeneous ownership holder without a demonstrated consumer-owned
retention model would broaden the current fixture work. Keep it under review
for the production submission ownership design; do not treat it as resolved or
as permission for trait-object extensibility.

### Required affected verification

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

### Explicitly pending

- Generalization beyond the named DX12/WGL fixture matrix, full S-stage
  regression, and all later S4 GPU/S5/S6 delivery work remain pending.
- S4 remains a separately bounded CPU closure; this document adds no S4 GPU,
  Forward, FramePipeline, or general extension claim.
