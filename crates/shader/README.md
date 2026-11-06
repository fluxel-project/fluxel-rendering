# Fluxel Shader

`fluxel-shader` assembles portable WGSL source contributions into one logical
program. It validates and reflects that program with Naga and returns a
source-provenanced `ShaderArtifact` containing composed source, source spans,
bindings, vertex inputs, inferred capabilities, requirements, and a logical
cache key.

`ShaderCompositionRequest` accepts ordered source units, selected entry points,
binding declarations, vertex semantics, callables, static options, and portable
requirements. Invalid WGSL, missing or conflicting entries, incompatible
interfaces, and unsatisfied logical requirements are structured errors.

The crate has no dependency on Material, Renderer, a GPU device, or a backend.
It does not choose a native shader route or create executable objects. A
consumer must use the selected RHI device facts to admit the artifact and pass
the same descriptor to native shader or pipeline creation. Logical validation is
therefore never evidence that a particular device can execute the artifact.

Source units and interface descriptions are runtime data. Extension
implementations are assembled through ordinary Rust generics and concrete types;
there is no trait-object shader-plugin registry.

Read the durable design in [documents/design.md](documents/design.md). Material
lowers authoring semantics into this crate's vocabulary; Renderer supplies pass
templates, entry points, concrete bindings, and Graph declarations.
