# Fluxel Material

`fluxel-material` implements Rust-defined fragments, WGSL Code Nodes, typed parameters and
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
mismatches, duplicate bindings, unsupported outputs, invalid instance values, and
emitted WGSL top-level namespace collisions return structured errors. The compact
`source_id` is a SHA-256 digest of the full canonical variant key; the full key
and complete Shader composition source remain collision-safe cache identities.

`MaterialDefinition::with_code_node` adds a `MaterialCodeNode` to the same typed
call graph. A node carries its callable name, input/result types, call edges,
resource declarations, and an author-owned Shader `SourceUnit`. Shared identical
resource declarations merge; conflicting resources and duplicate source identities
are rejected. Shader composition validates the actual WGSL callable and bindings.
The implemented surface outputs are `Color` and
`MetallicRoughness { properties }`. Both use a zero-argument `Vec4` base-color
root. Metallic-roughness additionally requires a named zero-argument `Vec2`
callable returning metallic then perceptual roughness, and emits
`evaluate_metallic_roughness()` beside `evaluate_material()`. Invalid property
roots and reserved generated evaluator names fail before Shader composition. A
typed node with inputs is reached through a material function that supplies them.

Pass consumers use `MaterialShaderContribution::source_units()` to preserve node
identities and source locations in diagnostics. `source()` derives aggregate WGSL,
and `source_unit()` provides the legacy single-unit representation; aggregation
does not preserve each node's diagnostic identity. The generated unit identity is
available through `source_id()`. Source units are the only stored source authority.

Material emits source units and metadata in Shader vocabulary. It neither chooses
a render pipeline nor creates RHI objects. The repository's S3 fixture combines a
contribution with a pass template, validates reflected bindings against concrete
resources, and executes it through the S2 and Graph paths on the reviewed DX12
and desktop WGL fixture routes. The S7 consumer extends this path with a selected
Code Node shared by Forward and Deferred. C2 additionally proves Material/Shader
composition for the bounded metallic-roughness evaluator pair. Separate native
DX12/WGL consumers validate its opaque direct-light output through both routes.
These do not establish an editor graph, arbitrary surface domains, or a complete
PBR product. The selected private Renderer PBR consumer is documented
in [the Renderer design](../../documents/design-renderer.md).
The later S8 validation-index measurement is Renderer-private: it does not alter
Material source authority, variant identity, or the requirement that each frame
declare and validate its actual bindings.
