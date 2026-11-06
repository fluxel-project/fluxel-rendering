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

The implemented surface roots are `Color` and selected
`MetallicRoughness { properties }`. Code Nodes participate in the same typed call
graph and preserve their source-unit identity for Shader diagnostics. Renderer
adds a pass template and entry points, then Shader assembles and validates the
complete logical program. Concrete resources and their physical access ranges
remain Renderer responsibilities.

Material definitions and instances are runtime data, while extension
implementations use concrete Rust types and generics. The crate has no
trait-object plugin registry.

Read the durable design in [documents/design.md](documents/design.md). See
`fluxel-shader` for logical program assembly and `fluxel-renderer` for scene,
preparation, Graph, and execution ownership.
