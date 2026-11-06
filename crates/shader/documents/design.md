# Shader design

`fluxel-shader` composes portable WGSL contributors into one validated logical
program. It is independent of Material and Renderer so that a material pipeline,
a compute consumer, or another rendering strategy can use the same service.

The crate owns source composition, source provenance, logical validation and
reflection, portable requirements, logical artifacts, and the cache for those
artifacts. It does not own material authoring semantics, scene data, render-pass
policy, RenderGraph construction, device selection, backend ABI, native lowering,
or executable shader and pipeline objects.

The durable neighbouring boundaries are documented in the
[Material design](../../material/documents/design.md) and
[Renderer design](../../renderer/documents/design.md). The portable RHI contract
remains the authority for an opened device, route selection, lowering, recording,
submission, completion, and presentation.

## Composition boundary

```text
Generic source/interface contributors      Pipeline consumer
  Material or another producer              pass template, entries,
  supplies ordered source units             logical inputs, static choices
                    \                         /
                     +-- ShaderCompositionRequest
                                  |
                                  v
                    compose, validate, reflect WGSL
                                  |
                                  v
                ShaderArtifact + logical requirements
                                  |
                                  v
                  selected RHI device and shader route
                                  |
                                  v
                  backend lowering and executable objects
```

`ShaderCompositionRequest` is Shader vocabulary. It combines ordered
`SourceUnit`s, selected `EntryPoint`s, binding declarations, optional logical
vertex-input mappings, callable declarations, static options, and portable
requirements. It deliberately contains no material IR, concrete GPU resource,
device identity, backend target, or native object.

Material lowering may contribute generated evaluator functions and interface
metadata, but it cannot create a complete request because only the active
consumer knows the pass template, entry points, geometry interface, and target
requirements. A non-material consumer may supply a complete request directly.

Source units are runtime data with caller-provided identities; they are not a
Rust type per shader or vertex layout. Extension assembly stays statically
dispatched in the owning application or consumer. Shader does not provide a
trait-object registry or a runtime plugin mechanism.

## Products and validation

`ShaderComposer::compose` concatenates the ordered contributors, records their
byte ranges, parses and validates WGSL with Naga, validates the declared
interface against the module, and returns a `ShaderArtifact`. The artifact holds
the composed source, source map, selected logical interface, portable
requirements, and its complete logical request key. It is not DXIL, SPIR-V,
MSL, a backend ABI, or a device-executable object.

The request is rejected before it becomes an artifact when it has empty source
or entries, duplicate contributors or selected entries, a missing/stage-mismatched
entry, more than one selected vertex entry, invalid WGSL, unsupported reflected
binding or value types, callable disagreement, or binding/vertex metadata that
does not match the reflected program. Parse and validation diagnostics retain a
source-unit identity and line/column when Naga provides a span.

Reflection is limited to selected entries. For each entry-reachable binding it
reports group/binding identity, resource class, any modeled shape, stage
visibility, and read/write/query/atomic use. Declared bindings must match that
reachable set in both directions: a declaration cannot invent an unused binding,
and reflection cannot introduce an undeclared one. Declared exact shapes,
visibility, and aggregate access, when supplied, must agree with reflection.

Vertex reflection is similarly logical. It derives locations and scalar/vector
types from selected vertex-entry arguments, including struct members; built-ins
and fragment inputs are not vertex attributes. A supplied semantic mapping must
cover the reflected locations exactly. Current public types are scalar or
two-, three-, and four-component F16, F32, I32, and U32 values. Physical stream
packing, offsets, strides, step rates, and concrete RHI layouts are deliberately
outside this crate.

Callable declarations are a second contract: a named WGSL function must exist
and have exactly the declared argument and result types. This lets authored
contributions expose selected evaluator functions without granting them control
over the full pass program.

## Requirements and device admission

`PipelineRequirements` combines caller-provided portable feature names with a
sufficient set of Naga capabilities inferred from the validated full module.
The capability algorithm removes candidate bits only while revalidation remains
successful, then validates the resulting set again. This preserves alternatives
and capability dominance relationships instead of assuming one bit per WGSL
construct.

Some inferred capabilities map to portable names, currently including primitive
index, early depth test, and ray query. Other capabilities may have no portable
name. A consumer must use the selected RHI device's actual capabilities and
route to admit every requirement it can represent, and explicitly refuse a
requirement it cannot represent or verify. Logical validation never proves that
a particular device or backend can execute the artifact.

Capability inference currently validates the full module. Consequently an
unselected entry can conservatively add a requirement. Any move to per-entry
requirements needs a real consumer, an explicit cache and diagnostic contract,
and proof that it cannot weaken device admission.

## Cache and lifecycle

`ShaderComposer` caches only validated logical artifacts. Its key is the complete
structural request: ordered contributor identities and content, selected entries,
binding/vertex/callable declarations, static options, and requirements. Dynamic
uniform values and concrete resource instances do not enter that key. Equal
requests return the same shared artifact; a source or interface change creates a
different logical identity.

The logical cache is not a cache of RHI shaders or pipelines. Device-local
executable objects belong to their consumer and must include the selected device
identity, device facts affecting admission, route/target ABI, and any exposed
lowerer or compiler identity. They cannot cross device replacement or loss.

Logical composition has a benchmark harness for repeatable cache-miss and
cache-hit workloads. It is a diagnostic and measurement tool, not a performance
contract. Changes to capability inference, composition inputs, Naga, or the
toolchain require renewed measurement before an optimization claim or a new
cache layer is accepted.

## Integration with rendering

Renderer preparation matches the logical vertex interface against concrete
geometry descriptors and chooses a physical vertex layout/pipeline. Compatible
interleaved, separate, and mixed streams may serve one logical interface;
missing or incompatible attributes are rejected before recording.

Renderer also resolves every reflected binding to a concrete resource, physical
range, access mode, incoming state, and device provenance. It declares that
binding-use manifest to RenderGraph before recording. Graph—not Shader—owns
resource dependencies, hazards, scheduling, lifetime analysis, and completion
based retirement. A sampler is a validated binding input but not a memory hazard.

Changing a material resource instance alone can reuse a compatible graph plan
after concrete-manifest validation. A binding layout, access mode, physical range,
or dependency-edge change requires a new graph definition. Persistent resource
residency and per-frame data lifetimes are likewise Renderer responsibilities.

## Current scope and retained direction

The implemented public boundary is WGSL composition, Naga validation/reflection,
source provenance, portable requirements, and logical artifact caching. The
bounded material consumer contributes ordered author-attributed source units and
validates its evaluator callables through this same request path. The selected
material surface consumer uses generated evaluators for base color and
metallic/perceptual-roughness values; that establishes the logical handoff only.

No current Shader contract establishes a general material authoring system,
complete PBR, a general programmable pipeline, arbitrary backend support,
native-binary caching, or device execution merely because the logical artifact
is valid. Future consumers may add only the narrow request, reflection, or
requirement vocabulary exercised by an actual producer and execution path.
