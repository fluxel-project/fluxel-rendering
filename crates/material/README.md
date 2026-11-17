# Fluxel Material

`fluxel-material` models Rust-defined material functions, WGSL Code Nodes,
typed parameters and resources, static variants, dynamic instances, and generic
Shader contributions. It owns authoring semantics and lowering; it does not own
a render pipeline, Graph, GPU resource, or device object.

`MaterialCompiler` validates a `MaterialDefinition` and produces a
`CompiledMaterialVariant`. Validation rejects invalid fragments and call graphs,
cycles, type errors, duplicate or incompatible resource declarations, invalid
instance values, unsupported surface outputs, and emitted WGSL top-level name
collisions. A compact `source_id` is a SHA-256 digest for diagnostics; the full
canonical key and full composed source remain the cache identity.

The implemented surface roots are `Color`, `MetallicRoughness`, and their UV0
variants (including a tangent-space normal evaluator). Code Nodes participate in
the same typed call graph and preserve their source-unit identity for Shader
diagnostics. Renderer adds a pass template and entry points, then Shader
assembles and validates the complete logical program. Concrete resources and
their physical access ranges remain Renderer responsibilities.
The Renderer sphere consumer uses these same evaluators for packed ORM
properties and occlusion, with a decoded tangent-space normal and the same
authored definition in Forward and Deferred. Environment resources and lighting
remain Renderer inputs.

`SurfacePolicy` carries static alpha mode, cutoff, culling, and two-sided
lighting. Defaults follow Babylon: opaque, cutoff 0.4, back-face culling enabled,
and two-sided lighting disabled. `SurfacePolicy::double_sided()` selects both
faces and back-face normal reversal. These choices participate in variant
identity; generated alpha helpers remain pure. Renderer applies rejection after
sampling and owns transparent ordering, blending, and Graph accesses.

`TransmissionExtensions` provides static PBR transmission and Beer-Lambert
volume-attenuation inputs for a UV0 normal metallic-roughness surface. It emits
pure evaluation helpers only; Renderer owns refraction-source selection,
transparent composition, and GPU lifetime.

Source layout follows ownership rather than API history: `authoring/` contains
asset identities, material definitions, typed values, and node descriptions;
`compiler/` owns validation, normalization, generated glue, and Shader
contributions; `runtime/` owns compiled variants and dynamic instances.
`lib.rs` is only the stable public API facade. Internal cross-module access uses
narrow crate-private methods, not exposed representation fields.

Material definitions and instances are runtime data, while extension
implementations use concrete Rust types and generics. The crate has no
trait-object plugin registry.

Read the durable design in [documents/design.md](documents/design.md). See
`fluxel-shader` for logical program assembly and `fluxel-renderer` for scene,
preparation, Graph, and execution ownership.

The next planned consumer expands typed node authoring and material semantics
through both Forward and Deferred, including transparency, refraction and
thickness-based absorption. The authored transmission contract and bounded
Forward environment route do not close that full delivery. See the
[planned design](documents/design.md#babylon-aligned-material-direction-planned)
and the [ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).
Blender-facing semantic preparation does not include an editor, parser, file
format or plugin implementation.
