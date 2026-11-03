# Fluxel Shader

`fluxel-shader` implements portable WGSL composition, Naga validation and
reflection, source provenance, logical interface checks, requirements, artifacts,
and cache identity.

Shader consumes generic source/interface contributions. It does not depend on
Material or Renderer. The independent RHI owns device routes, backend ABI and
lowering, native compilation, and executable objects. Logical input interfaces
do not prescribe Geometry stream packing or a fixed vertex struct. Concrete
consumer extensions use the static composition rule in
[ADR-0026](../../documents/adr/0026-static-extension-composition.md); source and
interface descriptions remain runtime data.

`ShaderCompositionRequest` combines ordered source units, selected entry points,
binding declarations, vertex semantics, callable declarations, static options,
and portable requirements. `ShaderComposer` returns a `ShaderArtifact` only when
the composed WGSL and declared interfaces agree. The artifact carries composed
source, source spans, reflected bindings/inputs, inferred Naga capabilities, and
a logical cache key. Invalid WGSL, missing entries, conflicting contributors, and
interface mismatches return structured errors.

This crate does not create device objects or decide whether an opened device can
execute an artifact. A consumer queries the selected RHI route and capabilities,
maps any expressible requirements, and creates the RHI shader/pipeline. The
repository's DX12 and desktop WGL integration fixture exercises that consumer
path for its selected shaders in the reviewed named-Windows closure; it is not
evidence for untested backends. Its cold-composition benchmark is a same-machine
baseline, not an optimization or performance claim.

The implemented subset is intentionally smaller than the target described in the
[Shader design](../../documents/design-shader.md). Geometry packing, Renderer
preparation, RenderScene, and FramePipeline remain outside this crate.
