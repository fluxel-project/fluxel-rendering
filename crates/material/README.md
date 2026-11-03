# Fluxel Material

`fluxel-material` implements Rust-defined fragments, typed parameters and
resources, a validated call graph, private normalized composition, static
variants, instances, and generic WGSL contributions.

Material contributes source and interfaces in Shader-owned vocabulary. It does
not depend on Renderer or choose a rendering pipeline. Renderer combines those
contributions with pass templates before Shader assembles a complete program.
Geometry, visibility, deformation and phase policy remain with the consuming
Features/pipeline. Their concrete extensions are assembled statically under
[ADR-0026](../../documents/adr/0026-static-extension-composition.md); Material
runtime data does not require a Rust type per material or trait-object plugins.

`MaterialCompiler` validates a `MaterialDefinition` and produces a
`CompiledMaterialVariant`. Parameters explicitly map to declared resources;
instances retain asset identity and content generation while dynamic values reuse
the compiled static variant. Invalid fragments, missing calls, cycles, type
mismatches, duplicate bindings, unsupported outputs, and invalid instance values
return structured errors.

Material emits source units and metadata in Shader vocabulary. It neither chooses
a render pipeline nor creates RHI objects. The repository's S3 fixture combines a
contribution with a pass template, validates reflected bindings against concrete
resources, and executes it through the S2 and Graph paths on the observed DX12
and desktop WGL routes. This limited fixture does not establish authoring graphs,
PBR, FramePipeline, or RenderScene APIs.
