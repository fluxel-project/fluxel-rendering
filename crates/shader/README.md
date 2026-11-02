# Fluxel Shader

Workspace scaffold for logical shader composition, source validation/reflection,
interfaces, requirements, source artifacts, variants, and cache identity.

Shader consumes generic source/interface contributions. It does not depend on
Material or Renderer. The independent RHI owns device routes, backend ABI and
lowering, native compilation, and executable objects. Logical input interfaces
do not prescribe Geometry stream packing or a fixed vertex struct. Concrete
consumer extensions use the static composition rule in
[ADR-0026](../../documents/adr/0026-static-extension-composition.md); source and
interface descriptions remain runtime data.

No shader API is implemented by this scaffold. Dependencies will be introduced
with implemented consumers, following the [Shader design](../../documents/design-shader.md)
and the organization roadmap.
