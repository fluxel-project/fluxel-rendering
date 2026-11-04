# S6 Typed Forward Acceptance Record

**Decision: accepted for the named Windows DX12 and desktop OpenGL/WGL S6
subset.** This closes the statically composed external Feature/product MVP
described below. It does not close S7/S8, final full regression, or untested
targets; it is not a release promise.

## Scope and source identity

S6 demonstrates a deliberately constrained, statically composed Forward
extension: an independently built Rust consumer supplies a typed `QuadFeature`,
a normal composed material, and an external composed pipeline. In one opaque
Forward Graph pass it is combined with the built-in mesh path. The Feature
prepares typed quad draws, the external pipeline applies the global ordering
policy, and its contribution uses the same pass and declared resource scopes as
the mesh contribution. It does not introduce a second scheduler, submission
path, GPU-device owner, trait-object registry, or runtime type erasure.

The Git HEAD locator at capture was
`6e160488d0e3e4d6663f7a866e2b445235b1cbd8`; that commit alone does not contain
this S6 implementation. Each attempt archives the exact dirty worktree inputs,
including new files, and hashes those bytes. The selected dependency revisions are:

| Component | Revision |
| --- | --- |
| `fluxel-rhi` | `b5d569feace548a6ed498901deaf31a5efcaca61` |
| `fluxel-bases` | `ee640d57a8d7c32d251252d0e8b32950636a770c` |
| Host | `33a6bbb3fa15500e928b4192ca8f280b735034d9` |

The candidate index records identical exact source-input lists for all four
selected attempts (412 inputs). The retained archive is
`documents/evidence/s6-typed-forward-candidate/native-evidence.zip`: 255
members, 23,231,202 bytes, SHA-256
`3ff2ec6b9b96abfb5b98aa65d86b948685928781c757e4bb8ca03216eba354c0`.

## Native evidence matrix

| Backend | Workload | Selected attempt | Evidence exercised |
| --- | --- | --- | --- |
| DX12 | `s6-composed` | `dx12-20261104T140459760842Z-1896` | Typed quad and mesh composition, readback/oracle, ordering, cache and pending-resource cases |
| DX12 | `s6-surface` | `dx12-20261104T140907035547Z-9408` | Native Surface presentation and resize, client capture, same-snapshot companion readback |
| native GL/WGL | `s6-composed` | `native-gl-wgl-20261104T140748170004Z-13340` | Typed quad and mesh composition, readback/oracle, ordering, cache and pending-resource cases |
| native GL/WGL | `s6-surface` | `native-gl-wgl-20261104T141059455126Z-18696` | Native Surface presentation and resize, client capture, same-snapshot companion readback |

`target/integration/s6-development/visual-review-b/review.json` is schema
version 1, has `evidence_status: "verified"`, and records all ten named
identity, command/diagnostic, matrix, and input/output/lifecycle/visual checks
as passed. Its `stage_closure_decision` is explicitly
`"not_made_by_this_tool"`.

## Functional evidence

The offscreen workload executes eleven frames per backend over four cases:

| Case | Observed behavior covered |
| --- | --- |
| `original-three-frame-cache` | First submission, warm cached plan, order flip, and continued Feature prepare/contribute callbacks |
| `enabled-disabled-enabled` | Runtime Feature enablement changes its contribution without changing resource ownership |
| `independent-providers` | Feature-local marker visibility and the pipeline's mesh visibility act independently on their own data before the external global ordering rule |
| `three-pending-replacements` | Geometry, material, and texture replacement remains retained across three pending submissions; application-side old references are released and pending ownership drains `3 -> 2 -> 1 -> 0` |

Each frame stores the actual mesh bounds and transform, marker center,
half-extent, tint, order, enablement and visibility, typed snapshot version, and
asset slot, slot generation, and content generation. The independent oracle
compares the retained readback with the frame-specific expected image; actual
and expected raw RGBA/PPM artifacts are retained. The material shader visibly
uses its texture binding and changes the normal material colour path rather than
merely declaring an unused binding.

The Surface workload runs two frames at 320x180 then 256x144. It records host
resize/configuration before acquisition, native client captures, and a
same-snapshot offscreen companion readback. Surface images are therefore a
presentation check paired with, rather than substituted for, readback evidence.

## Failure-boundary evidence

The focused checks recorded for this source cover the S6 additions and their
protected operations:

- Typed feature snapshots are scene-, issuer-, and version-bound; the consuming
  binding/context independently validates the selected Feature handle.
- The public phase/scope boundary is compile-fail tested: construction of a
  restricted phase context and direct access to the feature-pass scope or
  composition phase are rejected as private.
- Incompatible supplied texture payloads and material schemas are rejected
  before allocation, mapping, recording, or submission, even where an asset
  reference itself is present.
- The pending replacement case observes renderer-held ownership until the final
  consumer completion; a fourth submission at backpressure does not invoke the
  Feature callbacks.
- The composition path verifies the global draw order is a permutation of the
  supplied mesh and Feature draws, rather than independently sorted append-only
  lists.

The frozen targeted results are 67 Forward tests, four external
Feature/material tests, one phase/scope privacy compile-fail test, two CLI
tests, and 15 S6 evidence-tool tests. Eight existing shared-path reviewer
regressions also passed. These numbers are evidence inventory, not a claim of a
full-workspace regression.

## Shared-path boundary

S5 evidence is not used as proof of the new `RenderFeature` product boundary.
The S6 matrix independently proves the external Feature, custom material, and
single-pass composition in a real consumer. Existing shared Forward, residency,
completion, diagnostic, and review mechanisms are reused only where their
implementation and applicable coverage are unchanged; their earlier coverage
does not replace S6-specific evidence.

## Stabilized public subset and exclusions

This evidence supports only the demonstrated opaque Forward subset: a static
typed Feature binding, concrete generic composition with the built-in mesh
path, the selected Forward resources and material/binding schema, and runtime
enablement of those concrete implementations. Typed payload reuse by an
application within the same issuer remains data reuse; it is not a claim that a
payload is permanently affine to the Feature that originally produced it.

It does not stabilize arbitrary GPU products, cross-pass Features, transparent
composition, arbitrary runtime-loaded Feature types, a binary plugin ABI,
general material frameworks, terrain, water, particles, or skinning. Those are
outside this S6 acceptance record and require a separately selected consumer and
evidence before any public contract is widened.

## Measurement and publication evidence

The [S6 baseline](s6-measurement-baseline.md) retains five release samples per
backend, with clean/warm build time, complete process wall time, executable
bytes, and preparation/Graph/allocation/upload/cache observations. Every
sample passed the independent S6 reviewer. The five focused baseline-tool
checks also passed, including the actual review path and raw-readback tampering.
The [baseline archive index](evidence/s6-typed-forward-candidate/baseline-index.json)
hashes 800 original members. This is a baseline, not an optimization claim.

The public call site binds the concrete Feature, issuer and typed input with
`FeatureBinding::new`, then uses the same renderer-owned submission and
completion handles as Forward. CPU preparation does not require an artificial
static lifetime; actual assets and copied frame data remain completion-owned.
This exercised API is sufficient for the bounded consumer without adding a
registry or a general product framework. Broader authority remains consumer-gated.

The [native index](evidence/s6-typed-forward-candidate/index.json) identifies
all four reviewed native attempts. The
[source reconciliation](evidence/s6-typed-forward-candidate/source-reconciliation.json)
and [staged source check](evidence/s6-typed-forward-candidate/staged-source-check.json)
bind captured source/locks and retained evidence to the committed
candidate, allowing only explicitly reviewed documentation and line-ending
normalization differences. Original archives are never rewritten.

S7 must next prove a real Deferred pipeline over shared scene/material inputs
and a materially different external Feature. S8 requires its own representative
measurements and selected strategy evidence. Full regression remains deferred
until the planned stages are complete; no S6 green check substitutes for it.
