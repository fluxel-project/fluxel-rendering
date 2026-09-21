# ADR-0021: Make executable shaders own immediate-data ABI

**Status:** Accepted

## Context

`PipelineInterface` may intentionally be a compatible superset so renderers can
share it across pipelines. Native shader argument locations, however, are
compiled into an executable artifact. Deriving immediate-data argument presence
or indices from every range in a pipeline-interface superset lets an unused
range change a Metal `[[buffer(n)]]` ABI, or leave a shader-read argument
unbound.

## Decision

Each `ShaderInterface` declares the non-overlapping immediate byte intervals
its entry point reads. Pipeline construction requires every such interval to be
contained in a stage-visible `PipelineInterface` immediate range. The interface
remains a logical superset; it never assigns native immediate ABI.

For a fixed `ShaderAbiVersion`, a backend derives native immediate argument
presence, index and executable byte extent exclusively from the participating
shader interfaces and its frozen private ABI. Legal writes to a declared but
artifact-unused range have no executable observer and may be omitted by native
lowering.

## Consequences

Toolchains must reflect immediate reads into `ShaderInterface`, alongside
ordinary resource requirements. An artifact declaration is still a toolchain
contract; the RHI does not parse arbitrary native source to reconstruct it.
The shared pipeline validation prevents an incomplete immediate-layout contract
on every backend, while Metal additionally relies on it to keep direct argument
indices stable.
