# ADR-0021: Keep executable shader requirements authoritative

**Status:** Superseded as a concrete RHI ABI record; retained principle.

## Historical context

This ADR formerly named `ShaderInterface`, `PipelineInterface`, and
`ShaderAbiVersion` as current RHI types. They are not the selected independent
RHI public contract and must not be recreated by migration code.

## Durable principle

Only a compiled shader program can establish which bindings and immediate data
it actually reads. Logical layouts may be compatible supersets, but must not
silently assign backend-native argument locations or make an artifact-unused
range mandatory. Composition/reflection verifies the portable interface before
pipeline use; the opened RHI device owns route selection and private native ABI
lowering.

The target Shader boundary is specified by [Shader design](../design-shader.md)
and [Material design](../design-material.md). Exact current RHI shader types
and validation belong to the independent RHI documentation.
