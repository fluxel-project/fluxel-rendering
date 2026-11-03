# S2/S3 Named-Windows Closure

> Status: closed only for the named S2/S3 fixture matrix below. It is not a
> release, full-workspace regression, general Renderer API, S4 GPU result, S5
> Forward result, or support claim for untested targets.

## Reviewed product and archive

The reviewed source had HEAD baseline
`2ce58d599ee7bdfdea46b4ebfac487da94667aae` and was dirty. That baseline is not
the exact source identity: the attempt manifest plus each `source-inputs.zip`
member hash define the exact source and lock inputs, so later working-tree
contents do not replace them. Both attempts use the pushed Fluxel Bases revision
`ee640d57a8d7c32d251252d0e8b32950636a770c`.

The complete reviewed archive is
`documents/evidence/s2-s3-review-follow-up/native-evidence.zip` (9,695,531
bytes; 423 members; SHA-256
`0ffdcf326bd33a6eeedbf798e5a1466f6427fbc65fd636117ba69f9f46daf310`). Its
[`index.json`](evidence/s2-s3-review-follow-up/index.json) maps every archived
member to the original absolute path and records its SHA-256. Independent review
verified all 208 DX12 and 209 WGL artifact hashes, 110 archived source-input
members, route/readback records, Bases checkout, and diagnostics.

| Route | Reviewed attempt |
| --- | --- |
| Windows DX12 | `dx12-20261103T081700807832Z-25088` |
| Desktop OpenGL 4+ through WGL | `native-gl-wgl-20261103T081901164338Z-5752` |

The archive contains the original attempts under the `closure/` prefix. Their
unarchived roots were
`target/integration/s0-s3/2ce58d599ee7bdfdea46b4ebfac487da94667aae/attempts/`.

## Closed S2/S3 scope

The independent fixture exercised direct S2 shader/texture work and S3 material
work, including material replacement, on both named routes. It retained input,
independent expectation, readback, route/capability facts, diagnostics,
replacement-device observations, and reviewed live-surface captures before its
oracles. This closes the selected S2/S3 fixture acceptance for those exact
archives and routes.

Material now rejects collisions in the complete emitted WGSL top-level namespace:
generated `evaluate_material`, static options, shader-visible resources and
parameters, and scanned declarations from every fragment. It structurally refuses
duplicate function, type, variable, override, generated, resource, parameter, or
unparseable-top-level cases before composition. Its `source_id` is
`material:` plus the SHA-256 of the full canonical variant key; the full key and
complete Shader composition source remain collision-safe cache identities.

Focused CPU verification on the new Bases combination passed: 18 RenderScene
tests, one feature test, two external-consumer tests, seven Material library and
seven Material contract tests, and 14 targeted Shader tests with their applicable
Clippy checks. The independent native attempts are additional real-target
evidence, not a substitute for the deferred complete S-stage regression.

The Shader cold-compile document remains a historical same-machine baseline at
its recorded source and harness identity. The later lint-only transport cleanup
does not rewrite that measurement into evidence for newer code; it still makes no
speedup, threshold, or optimization claim.

Private heterogeneous snapshot retention remains justified only for the fixture's
completion-safe ownership. It is not a public plugin/Feature dispatch mechanism,
does not relax static extension composition, and is not a stabilized production
retention model.

## Explicit limits

This closure does not add Vulkan, browser WebGPU/WebGL, Metal, or any unnamed
Windows configuration. It does not close S4 GPU work, FramePipeline, Forward,
external same-frame composition, or the planned full regression.

The S4 CPU closure remains bound to its earlier source, lock, and evidence
record. The new Bases pin and this dirty-source S2/S3 archive do not automatically
refresh, broaden, or supersede that S4 record.
