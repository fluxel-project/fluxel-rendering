# Fluxel Material

Workspace scaffold for material functions, typed composition IR, parameters,
resource schemas, evaluation WGSL, and material assets/instances.

Material contributes source and interfaces in Shader-owned vocabulary. It does
not depend on Renderer or choose a rendering pipeline. Renderer combines those
contributions with pass templates before Shader assembles a complete program.
Geometry, visibility, deformation and phase policy remain with the consuming
Features/pipeline. Their concrete extensions are assembled statically under
[ADR-0026](../../documents/adr/0026-static-extension-composition.md); Material
runtime data does not require a Rust type per material or trait-object plugins.

No material API is implemented by this scaffold. Dependencies will be introduced
with implemented consumers, following the [Material design](../../documents/design-material.md)
and the organization roadmap.
