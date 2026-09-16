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

use fluxel_rendergraph::{
    AttachmentOps, BindingResourceSemantic, BufferRange, BufferReadUse, Extent3d, LoadOp,
    RasterColorAttachment, RasterDepthStencilAttachment, ResolvedBindingResource, StoreOp,
    TextureDesc, TextureDimension, TextureFormat, TextureRange, TextureReadUse, TextureUsage,
    TextureUsageKind, WriteCoverage,
};

use super::super::GlCompatibilityDevice;
use crate::resource::RasterKernel;
use crate::webgl2::api::tests::{compute_storage_snapshot, snapshot};
use crate::webgl2::api::{
    BufferId, GlError, GlFamilyProfile, MockCall, MockGlFamilyApi, TextureId,
};

/// The adapter under test, over the WebGL2 snapshot.
pub(super) type Adapter = GlCompatibilityDevice<MockGlFamilyApi>;

pub(super) fn adapter() -> Adapter {
    GlCompatibilityDevice::new(MockGlFamilyApi::from_discovery(snapshot(
        GlFamilyProfile::WebGl2,
    )))
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

/// Every mock call the adapter made, in order.
pub(super) fn calls(adapter: &mut Adapter) -> Vec<MockCall> {
    adapter.machine.backend().calls().to_vec()
}

/// How many of the mock calls so far satisfy `matched`.
pub(super) fn count(adapter: &mut Adapter, matched: fn(&MockCall) -> bool) -> usize {
    calls(adapter).iter().filter(|call| matched(call)).count()
}

pub(super) fn is_destroy_texture(call: &MockCall) -> bool {
    matches!(call, MockCall::DestroyTexture(_))
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
