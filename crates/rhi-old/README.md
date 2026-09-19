# fluxel-rhi

Fluxel's portable GPU execution crate. It is the implementation home for
device-affine resources, shaders, bindings, pipelines, command recording,
submission, completion, presentation, retirement, diagnostics, and
backend-private lowering.

## Authority and current status

The only semantic authority for the public RHI is
[the RHI architecture](../../documents/design-rhi.md) and its
[RHI design modules](../../documents/rhi-design/).

The checked-in crate may contain legacy 0.15 code while the 0.16 foundation is
implemented. Legacy source, examples, fixed artifacts, adapters, tests, and
feature names do not define or constrain the 0.16 public API. Do not preserve a
legacy API merely because it currently compiles; migrate it to the authoritative
contract or keep it explicitly dormant until its owning integration gate.

The RHI is not a wrapper for any native graphics API. Native queues, barriers,
fences, semaphores, descriptor heaps, resource states, browser sessions, and
platform handles remain private implementation details.

## Scope

The public model is based on opaque Device/Context identity plus generation.
GPU-affine values are validated before backend work, device loss is terminal,
and a new request creates a new identity/generation domain. Browser and
mini-game adapters participate as RHI backends; they do not define a separate
resource architecture.

RenderGraph declares logical resource use and dependency. RHI records actual
command use and executes accepted plans. Their declared-versus-actual boundary
is the graph bridge described by the authoritative design documents.

Portable capture/replay requires RHI-observable, reconstructable semantics, but
RHI does not own capture artifact storage, dependency closure, snapshot policy,
or ReplayRuntime.

## Installation

Use the workspace release selected by the ecosystem release plan:

~~~toml
[dependencies]
fluxel-rhi = "0.16"
~~~

During workspace development, use the repository workspace dependency rather
than inventing local public API shims. Published dependency revisions and
features must follow the release process.

## Features and backends

Backend selection, feature availability, and supported target matrix are
implementation and release facts. Capability is queried from the concrete
Device; it is not inferred from Cargo feature presence, a backend name, or a
Rust trait.

The foundation target matrix is DX12, Vulkan, Metal, WebGPU, and the GL family.
A backend/profile becomes supported only after its version-plan implementation,
contract tests, real-platform evidence, and ecosystem integration gate close.

## Verification

Run the workspace baseline checks from the repository root:

~~~powershell
cargo +1.87.0 fmt --all -- --check
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo +1.87.0 doc --workspace --all-features --no-deps --locked
~~~

GPU and cross-repository evidence requirements are defined by the
[foundation version plan](../../documents/version-plan.md). A local successful
test does not by itself establish portable backend support.

## Related documents

- [RHI architecture](../../documents/design-rhi.md)
- [RHI design modules](../../documents/rhi-design/)
- [RenderGraph architecture](../../documents/design-rendergraph.md)
- [Capture/replay architecture](../../documents/design-capture-replay.md)
- [Foundation version plan](../../documents/version-plan.md)

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
