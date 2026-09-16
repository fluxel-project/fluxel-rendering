//! The fixtures and readers every suite in this module shares.
//!
//! Responsibility: build the adapter under test, make the transients and
//! descriptors a frame would hand it, and read back what it asked the driver to
//! do.  Nothing here asserts anything -- a helper that decided what a correct
//! answer was would belong to the suite that knows the question.
//!
//! Not owned here: which calls are correct (each suite says), the shapes of the
//! values handed in (the contract declares them), and the backend itself
//! (`MockGlFamilyApi` is Layer 1's own test double and records the calls this
//! module reads).
//!
//! # Why this is a module and not the head of `super`
//!
//! It was the head of `super`, and it grew to the point where the three suites
//! sharing it could not be read without scrolling past it -- the same ceiling
//! `CLAUDE.md` §8 sets for any other module, reached by a file whose declared
//! job is module declarations.  Splitting it keeps `super` as what its doc says
//! it is: the module list and the tests that belong to no narrower subject.
//!
//! The two readers worth naming are [`refused`] and [`invalid`].  They are the
//! pair that keeps "this adapter does not implement it" apart from "this adapter
//! tried and the request was wrong", and a suite that used one where the other
//! was true would be asserting the wrong thing about the boundary.
//!
//! # Why the compute fixtures are here too
//!
//! The two compute suites were one file, and the fixtures they both needed --
//! [`storage_buffer`], [`read_write`], [`registered`] and the four call predicates
//! -- came with them when it split.  Putting them in either suite would have been
//! a suite reaching into its sibling for a value neither owns; the split's whole
//! point is that each file's contents are about its own subject.  They are shaped
//! like the raster ones above: built through the adapter rather than by hand, so
//! the ids are the ones a real creation produces, and left for the caller to
//! account for.

use fluxel_rendergraph::{
    AttachmentOps, BindingResourceSemantic, BindingSetId, BoundBindings, BoundBuffer,
    BoundComputePipeline, BufferDesc, BufferRange, BufferReadUse, BufferReadWriteUse, BufferUsage,
    BufferUsageKind, ComputePipelineId, ExecutionBackend, Extent3d, LoadOp, RasterColorAttachment,
    RasterDepthStencilAttachment, RenderObjectProvider, ResolvedBindingResource, StoreOp,
    TextureDesc, TextureDimension, TextureFormat, TextureRange, TextureReadUse, TextureUsage,
    TextureUsageKind, WriteCoverage,
};

use super::super::GlCompatibilityDevice;
use super::super::compute::WithCompute;
use super::super::object::{Bindings, ComputePipeline, Recipe};
use super::super::retention::GlRetentionLease;
use crate::resource::{ComputeKernel, RasterKernel};
use crate::webgl2::api::tests::{compute_storage_snapshot, snapshot};
use crate::webgl2::api::{
    BufferId, GlError, GlFamilyProfile, MockCall, MockComputeStorageApi, MockGlFamilyApi, TextureId,
};

/// The adapter under test, over the WebGL2 snapshot.
pub(super) type Adapter = GlCompatibilityDevice<MockGlFamilyApi>;

/// The adapter over a backend that *has* the optional compute domains.
///
/// The witness is named rather than defaulted, and the backend is Layer 1's
/// opt-in wrapper: the wrapper is the only way to a `GlOptionalComputeBackend` in
/// this crate, and it refuses to be built over a snapshot that did not prove
/// compute and storage buffers.  So this type is reachable exactly when the
/// context is, which is the property the design is built on.
pub(super) type ComputeAdapter = GlCompatibilityDevice<MockComputeStorageApi, WithCompute>;

pub(super) fn adapter() -> Adapter {
    GlCompatibilityDevice::new(MockGlFamilyApi::from_discovery(snapshot(
        GlFamilyProfile::WebGl2,
    )))
}

/// The compute-capable adapter, over the one snapshot that proved all three
/// capabilities and whose format table can carry a storage image.
pub(super) fn compute_adapter() -> ComputeAdapter {
    GlCompatibilityDevice::new(compute_backend(true))
}

/// The same adapter over a context that proved compute and storage buffers and
/// did *not* prove storage images.
///
/// The pair is what makes the second refusal testable: an adapter of this type
/// exists and dispatches, and the one verb whose capability the snapshot never
/// proved has to refuse on the snapshot's answer rather than on the bound -- see
/// [`super::super::compute`].
pub(super) fn compute_adapter_without_storage_images() -> ComputeAdapter {
    GlCompatibilityDevice::new(compute_backend(false))
}

fn compute_backend(storage_image: bool) -> MockComputeStorageApi {
    MockComputeStorageApi::new(MockGlFamilyApi::from_discovery(compute_storage_snapshot(
        storage_image,
    )))
    .expect("the snapshot proved compute and storage buffers")
}

/// The adapter over the one snapshot that proved compute, storage and a storage
/// image, which is the only one whose format table can carry a storage request.
pub(super) fn desktop() -> Adapter {
    GlCompatibilityDevice::new(MockGlFamilyApi::from_discovery(compute_storage_snapshot(
        true,
    )))
}

/// A common texture description, with the two things this adapter decides
/// together -- the dimension and the array-layer count -- left to the caller.
pub(super) fn texture(dimension: TextureDimension, array_layers: u32, depth: u32) -> TextureDesc {
    TextureDesc {
        dimension,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth,
        },
        mip_levels: 1,
        array_layers,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

/// The plain case: one two-dimensional layer.
pub(super) fn plain_texture() -> TextureDesc {
    texture(TextureDimension::D2, 1, 1)
}

/// A usage set an attachment-shaped transient is compiled with.
pub(super) fn colour_usage() -> TextureUsage {
    TextureUsage::from_kinds([
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::CopySource,
    ])
}

/// A usage set a sampled transient is compiled with.
pub(super) fn sampled_usage() -> TextureUsage {
    TextureUsage::from_kinds([TextureUsageKind::Sampled, TextureUsageKind::CopySource])
}

/// One colour attachment at `index`, with the operations the caller names.
///
/// Borrowed from the caller's own id, like every fixture here: the contract's
/// descriptor borrows its physical resources
/// (`RasterColorAttachment<'a, T>` holds `&'a T`), and a helper that leaked an id
/// to hand back a `'static` one would give a suite a shape no frame produces.
pub(super) fn attachment<'a>(
    index: u32,
    texture: &'a TextureId,
    load: LoadOp<[f32; 4]>,
    store: StoreOp,
) -> RasterColorAttachment<'a, TextureId> {
    RasterColorAttachment {
        index,
        texture,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load,
            store,
            write_coverage: WriteCoverage::Full,
        },
    }
}

/// The attachment every pass that renders in this module uses: index zero,
/// cleared, stored.  It is the only shape a pass here can run, and stating it
/// once keeps a suite that varies one field from restating the other three.
pub(super) fn cleared<'a>(texture: &'a TextureId) -> RasterColorAttachment<'a, TextureId> {
    attachment(
        0,
        texture,
        LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        StoreOp::Store,
    )
}

/// A depth-stencil attachment over `texture`, cleared and stored.
pub(super) fn depth(texture: &TextureId) -> RasterDepthStencilAttachment<'_, TextureId> {
    RasterDepthStencilAttachment {
        texture,
        range: TextureRange::Whole,
        depth: Some(AttachmentOps {
            load: LoadOp::Clear(1.0),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        }),
        stencil: None,
    }
}

/// One `ResolvedBindingResource::Texture`, borrowed from the caller's own id.
///
/// Borrowed rather than leaked: the contract's resolution borrows its physical
/// resources (`ResolvedBindingResource<'a, T, B>`), and a suite that leaked ids
/// to get a `'static` one would be exercising a shape no frame produces.
pub(super) fn sampled(texture: &TextureId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Texture {
        physical: texture,
        range: TextureRange::Whole,
        semantic: BindingResourceSemantic::TextureRead(TextureReadUse::Sampled),
    }
}

/// One `ResolvedBindingResource::Buffer` authorized as a uniform read.
pub(super) fn uniform(buffer: &BufferId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Buffer {
        physical: buffer,
        range: BufferRange::Whole,
        semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Uniform),
    }
}

/// The resources a frame would resolve for `kernel`, one per WGSL binding.
///
/// The filler for a binding the artifact does not read is a sampled texture,
/// which is what the only such binding in the set -- the sampler that folds into
/// the texture's -- actually resolves as.  Which resource sits where is the
/// recipe's own business and is read off its identity rather than decided here,
/// so a suite that needs a *different* arrangement builds it by hand and says
/// so.
pub(super) fn resolved<'a>(
    kernel: RasterKernel,
    texture: &'a TextureId,
    buffer: &'a BufferId,
) -> Vec<ResolvedBindingResource<'a, TextureId, BufferId>> {
    let identity = kernel.portable_identity();
    (0..identity.binding_count)
        .map(|binding| {
            if identity.uniform_binding == Some(binding) {
                uniform(buffer)
            } else {
                sampled(texture)
            }
        })
        .collect()
}

/// A transient buffer, in the two storage roles the family's one GL bit covers.
///
/// Handed back whole rather than as its identity for
/// [`super::super::object`]'s reason: the range a storage binding point is given
/// is validated against the real allocation, so a test that let the lease go early
/// would be binding a buffer the device no longer holds.
pub(super) fn storage_buffer(
    adapter: &mut ComputeAdapter,
    size: u64,
) -> BoundBuffer<BufferId, GlRetentionLease> {
    adapter
        .create_transient_buffer(
            BufferDesc { size },
            BufferUsage::from_kinds([BufferUsageKind::StorageRead, BufferUsageKind::StorageWrite]),
        )
        .unwrap_or_else(|error| panic!("a transient storage buffer: {error:?}"))
}

/// One storage buffer, authorized as the artifact's declaration states.
///
/// `ReadWrite` rather than a read: all five artifacts declare their storage block
/// read-write, and the semantic rule this exercises is the one that requires a
/// write to *name* the storage role -- so the agreeing case has to name it.
pub(super) fn read_write(buffer: &BufferId) -> ResolvedBindingResource<'_, TextureId, BufferId> {
    ResolvedBindingResource::Buffer {
        physical: buffer,
        range: BufferRange::Whole,
        semantic: BindingResourceSemantic::BufferReadWrite(BufferReadWriteUse::Storage),
    }
}

/// The pipeline and binding set a frame would be handed for a compute `kernel`.
///
/// Registered through the adapter's own registry rather than built directly, so
/// that every test drives the two objects the way a frame does -- and so that the
/// registry's own compute path, which F4(c) is what put there, is on the route
/// every one of these assertions takes.
pub(super) fn registered(
    adapter: &mut ComputeAdapter,
    kernel: ComputeKernel,
    resources: &[ResolvedBindingResource<'_, TextureId, BufferId>],
) -> (
    BoundComputePipeline<ComputePipeline, GlRetentionLease>,
    BoundBindings<Bindings, GlRetentionLease>,
) {
    let mut objects = adapter.object_registry();
    let pipeline_id = ComputePipelineId::new(0);
    let set_id = BindingSetId::new(0);
    objects
        .register_compute_pipeline(pipeline_id, kernel)
        .unwrap_or_else(|error| panic!("{kernel:?} is lowered on this profile: {error:?}"));
    objects.register_bindings(set_id, Recipe::Compute(kernel));
    let pipeline = objects
        .compute_pipeline(pipeline_id)
        .unwrap_or_else(|error| panic!("{kernel:?} was registered: {}", error.context.detail));
    let bindings = objects
        .bindings(set_id, resources, &[])
        .unwrap_or_else(|error| panic!("a set for {kernel:?}: {}", error.context.detail));
    (pipeline, bindings)
}

/// Whether this call is a Layer 1 pass boundary.
///
/// The pair a raster pass brackets its draws with, and the pair a compute pass
/// has none of: a dispatch into a framebuffer that does not exist would be a
/// command this family cannot express, so the absence a suite asserts is the claim
/// and not an omission.
pub(super) fn is_pass_boundary(call: &MockCall) -> bool {
    matches!(call, MockCall::BeginRenderPass(_) | MockCall::EndRenderPass)
}

pub(super) fn is_create_program(call: &MockCall) -> bool {
    matches!(call, MockCall::CreateProgram(_))
}

pub(super) fn is_install_program(call: &MockCall) -> bool {
    matches!(call, MockCall::SetComputeProgram(_))
}

pub(super) fn is_dispatch(call: &MockCall) -> bool {
    matches!(call, MockCall::Dispatch(_))
}

/// Every mock call the adapter made, in order.
pub(super) fn calls(adapter: &mut Adapter) -> Vec<MockCall> {
    adapter.machine.backend().calls().to_vec()
}

/// How many of the mock calls so far satisfy `matched`.
pub(super) fn count(adapter: &mut Adapter, matched: fn(&MockCall) -> bool) -> usize {
    calls(adapter).iter().filter(|call| matched(call)).count()
}

/// The compute adapter's three readers, which exist beside the three above
/// rather than replacing them.
///
/// The two mocks are two types with no shared trace trait, so a reader has to be
/// written per backend type; the alternative is a trait whose whole content is
/// these three methods and whose only implementors are Layer 1's two recorders,
/// which is an abstraction with no consumer beyond this module.  What is *shared*
/// is the comparison: [`MockCall`] is one enum for both, so the predicates below
/// and every test's own matching work on either trace unchanged.
pub(super) fn compute_calls(adapter: &mut ComputeAdapter) -> Vec<MockCall> {
    adapter.machine.backend().calls().to_vec()
}

/// How many of the compute adapter's calls so far satisfy `matched`.
pub(super) fn compute_count(adapter: &mut ComputeAdapter, matched: fn(&MockCall) -> bool) -> usize {
    compute_calls(adapter)
        .iter()
        .filter(|call| matched(call))
        .count()
}

/// Forgets the compute adapter's trace so far, on [`trace_from_here`]'s terms.
pub(super) fn compute_trace_from_here(adapter: &mut ComputeAdapter) {
    adapter.machine.backend().clear_calls();
}

pub(super) fn is_destroy_texture(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyTexture(_))
}

pub(super) fn is_create_texture(call: &MockCall) -> bool {
    matches!(call, MockCall::CreateTexture(_))
}

pub(super) fn is_destroy_buffer(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyBuffer(_))
}

pub(super) fn is_destroy_fence(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyFence(_))
}

/// Forgets the trace so far, so a later count is about what follows.
pub(super) fn trace_from_here(adapter: &mut Adapter) {
    adapter.machine.backend().clear_calls();
}

/// The operation name of the fail-closed refusal `result` carries.
///
/// Every fail-closed verb in this slice refuses with `GlError::Unsupported` and
/// names itself, so a test that only checked for an error would accept a
/// validation failure or a context error as evidence that the verb refuses.  The
/// name is checked because the name is what an operator reads.
pub(super) fn refused<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to refuse this"),
        Err(GlError::Unsupported { operation, .. }) => operation,
        Err(other) => panic!("expected a fail-closed refusal, got {other:?}"),
    }
}

/// The reason the fail-closed refusal `result` carries.
///
/// [`refused`] answers *which verb*, and that is all most suites need.  This
/// answers *which fact*, and it exists because the same verb can refuse twice
/// over for two different reasons: every compute verb refuses once when the
/// adapter's type has no such command domain and once when the context's snapshot
/// did not prove the capability the verb needs, and both are
/// `GlError::Unsupported` at the same operation name.  So the pair is only
/// distinguishable by the sentence, and a suite asserting on the one that applies
/// has to read it.
pub(super) fn refusal_reason<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to refuse this"),
        Err(GlError::Unsupported { reason, .. }) => reason,
        Err(other) => panic!("expected a fail-closed refusal, got {other:?}"),
    }
}

/// The operation name of the validation failure `result` carries.
///
/// The counterpart of [`refused`] for the verbs that *do* reach a provider: what
/// a bad request gets back from there is a validation failure and not a
/// fail-closed refusal, and a test that accepted either would not be able to
/// tell "this adapter does not implement it" from "this adapter tried and the
/// request was wrong".
pub(super) fn invalid<T>(result: Result<T, GlError>) -> &'static str {
    match result {
        Ok(_) => panic!("the adapter was expected to reject this"),
        Err(GlError::Validation { operation, .. }) => operation,
        Err(other) => panic!("expected a validation failure, got {other:?}"),
    }
}
