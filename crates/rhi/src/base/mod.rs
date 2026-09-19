//! The backend seam: what a native implementation must provide, and what it is
//! forbidden to decide.
//!
//! [`crate::api`] is the frozen portable contract. A backend lowers that contract
//! onto one native API, and this module is the private interface between the two.
//! Nothing here is public, and that is a decision rather than an omission:
//! `design-rhi.md` opens by declaring native lowering *not frozen* — "Not frozen
//! here: native lowering, backend-internal objects, ABI" (`design-rhi.md:7-8`) —
//! so the shape of this seam is this crate's own design, not a transcription of
//! the specification.
//!
//! # Why the seam is private
//!
//! `08-governance-freeze-checklist.md` section 59 lists nine capability traits
//! and six native types that 0.16 does not **export**: `IndirectApi`, `QueryApi`,
//! `BindlessApi`, `RayTracingApi`, `MeshShaderApi`, `AsyncComputeApi`,
//! `MultiviewApi`, `DeviceAddressApi`, `SparseResourceApi`, `NativeQueue`,
//! `NativeFence`, `NativeSemaphore`, `NativeBarrier`, `DescriptorHeap`,
//! `GpuAddress`.
//!
//! Read precisely, that bans a *public* capability API. It does not ban a
//! crate-private lowering interface — the word is "export", and a `pub(crate)`
//! trait is not exported. Reading it the other way would invert the conclusion
//! and make the backends impossible to write. What the rule really protects is
//! the caller: a caller never names a backend type, never picks a trait to bound
//! on, and never learns the platform in order to write correct code (root
//! specification section 3.1). The traits below keep that true by being
//! unreachable from outside the crate, and `BackendKind` — already public — stays
//! the only vocabulary in which "which backend" is ever said.
//!
//! # The four disciplines
//!
//! Every trait below is bound by these. A backend that breaks one is defective
//! even when the driver accepts the call and the picture looks right.
//!
//! 1. **Portable validation runs first, and it runs in [`crate::api`].** Section
//!    3.1 requires O(1) identity validation before any backend call, and section
//!    4 forbids handing a driver a problem that portable code could have found.
//!    By the time a method here is reached, the portable layer has already
//!    decided everything it is able to decide.
//! 2. **A backend does not make legality decisions.** It may report a fact — a
//!    capability, a support answer, an enumeration — and it may fail for a native
//!    reason, but it never has an opinion about whether the caller's request was
//!    well-formed. A native failure that a portable rule already covers is a gap
//!    in the portable layer, not a case for the backend to handle.
//! 3. **A backend never silently substitutes a fallback.** Section 9.4
//!    (`02:507-532`) names three by name — a blit must not become a fullscreen
//!    shader pass, a copy must not become a CPU staging round trip, a resolve
//!    must not become a compute dispatch — and the rule behind them is general.
//!    An operation the backend cannot perform returns
//!    [`crate::api::RhiErrorKind::Unsupported`] and the caller decides. A
//!    substitute also corrupts the RHI's own reported facts: a backend that
//!    quietly allocates a staging buffer breaks every working-set number the
//!    statistics module publishes.
//! 4. **`InvalidUsage` belongs to the portable layer.** A backend reports
//!    `Unsupported`, `DeviceLost`, `OutOfMemory`, or a native failure; it does
//!    not report `InvalidUsage`, because reporting it would mean re-deciding a
//!    question the portable layer has already answered.
//!
//!    The fourth rule is **this crate's own convention, and not the
//!    specification's**. The specification never states it in the negative; the
//!    nearest it comes is mapping `InvalidUsage` onto range, alignment, and usage
//!    errors (`01:373`), every one of which sits inside the seventeen portable
//!    checks it *does* enumerate. It is written down here as a decision so that a
//!    backend author meets it as a rule rather than as a surprise in review, and
//!    it is enforced by tests rather than assumed.
//!
//! # Threading
//!
//! A backend is `Send + Sync + 'static` and reached through a shared pointer,
//! because the portable handles that own one are `Clone` and cloning a handle
//! must not clone a native device. Interior mutability therefore belongs to the
//! backend, which is why the traits take `&self` wherever the operation is not
//! inherently sequential — a device request is the exception, because section 5.9
//! makes it single-shot.
//!
//! # Layout
//!
//! ```text
//! base/platform   provider, device request, device   (specification module 01)
//! base/resource   buffer, texture, view, sampler     (specification module 02)
//! base/shader     the native entry point             (specification module 03)
//! base/binding    the native descriptor packet       (specification module 03)
//! base/pipeline   the native pipeline state          (specification module 03)
//! base/mock       the CPU/mock backend
//! ```
//!
//! The platform backends live outside this module, under
//! `crate::backend::{dx12, vulkan, metal, webgpu, gl}` (`01:126-132`). Machinery
//! that more than one backend needs lands here beside the seam, as it is needed —
//! not in advance of a consumer.

pub(crate) mod platform;

// The resource chapter's seam. It is a separate module from `platform` because
// the two seams are reached from different handles and grow for different
// reasons: a platform backend is asked about the domain, a resource backend
// carries one native object. The split is not cosmetic — it is what keeps
// `base/platform.rs` from becoming the file that must know about every resource
// in the crate, the same argument adjudication A28 makes for keeping
// `create_buffer` in the resource chapter rather than in `api::platform`.
pub(crate) mod resource;

// The shader chapter's seam. Separate from `resource` for the reason in this
// module's layout note, and it is the one seam whose object a *later* chapter
// consumes rather than the chapter that created it: a module is created here and
// read when a pipeline state is built.
pub(crate) mod shader;

// The bind-group seam. Separate from `resource` because a bind group is not an
// allocation: it is the packet that points at allocations, it is created from a
// layout rather than from a descriptor of its own, and its native shape is a
// descriptor write rather than a memory placement. It has no companion trait for
// the *layout*, which is a Direct3D 12 fact rather than an oversight — the module
// documentation records which backends will need one.
pub(crate) mod binding;

// The pipeline seam. It carries the object a dispatch binds and a draw binds, and
// it is split from `shader` because the two fail for different reasons: a module
// is bytes a producer made, while a pipeline is what a driver built out of them and
// is the first object here whose creation the driver can refuse for a reason about
// the *program*.
pub(crate) mod pipeline;

// The recording and submission chapter's seam. It is separate from `platform`
// for the reason that one is separate from `resource`: the vocabulary is reached
// from a different handle and grows for a different reason. A submission request
// carries a whole validated plan rather than one descriptor, and the types here
// are what keep `base/platform.rs` from having to name a `RecordedWork`.
//
// The two operations that consume them are declared on the device trait in
// `platform`, and the module documentation of `command` records why.
pub(crate) mod command;

// Beside the seam rather than under `api`, because it is crate-private machinery
// with no portable vocabulary of its own. Its consumer is the capability
// fingerprint (specification section 7.1); the layout and pipeline fingerprints
// of module 03 will be the next ones, and it lands here now because that first
// consumer exists — not in advance of it.
pub(crate) mod digest;

// The conformance backend is compiled for this crate's own test build only; its
// module note records why it is not behind the `test-support` feature yet.
#[cfg(test)]
pub(crate) mod mock;
