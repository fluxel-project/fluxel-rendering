//! Sections 29 through 38: recording and actual resource uses.
//!
//! The chapter's tests are split the same way its submodules are, and for the same
//! reason: the recorder's state machine, the attachment set, the draw verbs, the
//! dispatch verb, and the copy and transfer verbs are separately specified and
//! separately refusable. One file per group keeps a reader's working set to the
//! rules that actually interact.
//!
//! Every test names the rule it drives and asserts the exact
//! [`RhiErrorKind`](crate::api::error::RhiErrorKind), because the kind is the part
//! a caller branches on — `InvalidUsage` means "you described this wrongly",
//! `WrongDevice` means "this belongs to another device", `IncompatibleInterface`
//! means "these two agree on nothing", and `Unsupported` means "this device cannot
//! do it". A test that accepted any error would not notice the difference.
//!
//! Two conventions run through the whole group:
//!
//! * Objects are assembled through the crate-private constructors the device verbs
//!   will call, the same way `tests/resource/*` does. A test that could not name an
//!   object could not test what an accessor reports about it.
//! * A device-dependent question is answered by stating a device answer, not by
//!   asserting that the verb stopped. [`recorder_reporting`] builds a device that
//!   reports exactly the facts a test names and hands the recorder *its* snapshot,
//!   so a refusal here is the verb reading a device's own table. The devices in
//!   this chapter that report nothing are the ones that make "this device cannot"
//!   observable as [`RhiErrorKind::Unsupported`].

mod attachment;
mod copy;
mod raster;
mod recorder;
mod transfer;

use std::sync::Arc;

use crate::api::binding::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor, BindingKind, BindingResource,
    BindingSlot, BindingSlotId, LayoutFingerprint,
};
use crate::api::capability::CapabilityFacts;
use crate::api::command::attachment::ColorAttachmentView;
use crate::api::command::{
    BufferCopy, BufferTextureCopy, ColorAttachment, CommandRecorder, LoadOp, RasterScopeDescriptor,
    StoreOp,
};
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, Label, ObjectId};
use crate::api::pipeline::{
    ColorTargetState, PipelineInterface, PipelineInterfaceCompatibilityId,
    PipelineInterfaceDescriptor, RasterPipeline, RasterPipelineDescriptor,
};
use crate::api::presentation::{AcquiredFrameId, FrameAttachment};
use crate::api::resource::buffer::{
    Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUsage,
};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport, TexelCopyLayoutLimits,
};
use crate::api::resource::subresource::{
    Origin3d, TextureAspect, TextureAspects, TextureSubresourceLayers,
};
use crate::api::resource::texture::{Extent3d, Texture, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{BufferUploadDescriptor, UploadDescriptor, UploadJob};
use crate::api::resource::view::{TextureView, TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderLocation, ShaderModule, ShaderRequirements, ShaderStage,
    ShaderStages,
};
use crate::api::tests::fixture;
use crate::base::mock::{recorder_for_test, recorder_without_facts_for_test};

fn identity(instance: u64, generation: u64) -> DeviceIdentity {
    DeviceIdentity::new(
        DeviceInstanceId::new(instance),
        DeviceGeneration::new(generation),
    )
}

fn device() -> DeviceIdentity {
    identity(1, 1)
}

fn other_device() -> DeviceIdentity {
    identity(2, 1)
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

/// Asserts a result is the exact error kind the specification requires.
fn assert_kind(result: RhiResult<()>, expected: RhiErrorKind) {
    match result {
        Ok(()) => panic!("expected {expected}, but the operation was accepted"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

/// An open recorder on the test device.
///
/// Built through the real creation verb over a mock device that reports no
/// capability at all, so every device-gated verb in this chapter answers
/// `Unsupported` here. A test that needs a device answer to be *present* states it
/// through [`recorder_for_test`], which is also what keeps the fact table a device
/// answer rather than a value a test invented.
fn recorder() -> CommandRecorder {
    recorder_without_facts_for_test(device())
}

/// A recorder whose device reports exactly the facts the test states.
///
/// The device is built first and the recorder takes *its* snapshot, so what a
/// verb decides against here is a device's own answer rather than a table handed
/// to the recorder. That distinction is the point of the round that made these
/// verbs decidable at all: a test could otherwise prove only that the recorder
/// reads the field it was given.
fn recorder_reporting(facts: CapabilityFacts) -> CommandRecorder {
    recorder_for_test(device(), facts, lanes())
}

/// One lane accepting the three domains the command chapter records.
///
/// Wider than `MockDevice`'s default, which deliberately omits `COMPUTE` because
/// a device reporting no compute feature must not also report a lane that accepts
/// compute work. A test that states the feature states the lane with it.
fn lanes() -> crate::api::submission::SubmissionCapabilities {
    use crate::api::submission::{
        LaneWorkDomains, SubmissionCapabilities, SubmissionLaneClass, SubmissionLaneId,
        SubmissionLaneInfo,
    };
    let mut domains = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
    domains = domains.union(LaneWorkDomains::COMPUTE);
    SubmissionCapabilities::new(vec![SubmissionLaneInfo::new(
        SubmissionLaneId::new(0),
        SubmissionLaneClass::General,
        domains,
    )])
}

/// Facts in which the buffer-to-buffer route exists with a 4-byte alignment.
///
/// Section 12.4's alignment half needs a route that *has* a layout to check
/// against; a device reporting none is answering a different question, so this is
/// the smallest fact table that makes the alignment rule reachable.
fn facts_with_buffer_copy_route(offset: u64, size: u64) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(
            Some(BufferCopyLayoutLimits::new(offset, size)),
            None,
        )),
    );
    facts
}

/// Facts in which the buffer/texture route exists with a texel-copy alignment.
fn facts_with_texel_copy_route(buffer_offset: u64, bytes_per_row: u32) -> CapabilityFacts {
    let mut facts = CapabilityFacts::empty();
    let shape = crate::api::resource::route::RouteQuery::BufferToTexture {
        dimension: crate::api::resource::texture::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        aspect: crate::api::resource::subresource::TextureAspect::Color,
    };
    facts.record_route(
        shape,
        RouteSupport::Supported(RouteCapabilities::new(
            None,
            Some(TexelCopyLayoutLimits::new(buffer_offset, bytes_per_row)),
        )),
    );
    facts
}

fn buffer_with(usage: BufferUsage, size: u64) -> Buffer {
    buffer_of(10, usage, size)
}

/// A buffer with an identity the caller picks.
///
/// Two buffers that are the same object are one buffer, and a copy between them
/// is the overlap refusal rather than a copy — so a test that wants a *legal*
/// copy has to name two different identities, which is what the explicit id is
/// for.
fn buffer_of(id: u64, usage: BufferUsage, size: u64) -> Buffer {
    fixture::buffer(object(id), device(), BufferDescriptor::new(size, usage))
}

/// A byte-range copy that passes every portable check.
///
/// Section 34.1's list is what it has to satisfy to reach the device route
/// question, and the two distinct buffer identities are part of that: a copy from
/// a buffer to itself is the overlap refusal, not a copy.
fn buffer_copy() -> BufferCopy {
    BufferCopy {
        src: buffer_of(10, BufferUsage::COPY_SRC, 64),
        src_offset: 0,
        dst: buffer_of(11, BufferUsage::COPY_DST, 64),
        dst_offset: 0,
        size: 64,
    }
}

/// A prepared 16-byte upload into a 64-byte copy-destination buffer.
fn buffer_upload() -> UploadJob {
    UploadJob::new(
        object(80),
        device(),
        UploadDescriptor::Buffer(BufferUploadDescriptor {
            label: Label::default(),
            dst: buffer_with(BufferUsage::COPY_DST, 64),
            dst_offset: 0,
            bytes: vec![7u8; 16].into(),
        }),
    )
}

/// A 4x4 texture that can be both rendered into and read back.
fn renderable_texture(format: TextureFormat) -> Texture {
    Texture::new(
        object(20),
        device(),
        TextureDescriptor::new_2d(
            4,
            4,
            format,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ),
    )
}

/// A one-mip, one-layer color selection.
///
/// Shared by every copy-shaped test rather than owned by one of them: the
/// subresource and the origin appear in all six copy verbs, and the two facts
/// they carry — which mip, which layers — are the ones a route key and a region
/// check both read.
fn color_layers(layer_count: u32) -> TextureSubresourceLayers {
    TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count,
    }
}

/// The origin every copy-shaped test starts a region at.
fn origin() -> Origin3d {
    Origin3d { x: 0, y: 0, z: 0 }
}

/// A 4x4 texture that accepts a copy into it.
///
/// A different object id from [`renderable_texture`]'s, because the two are
/// different textures: one is read from and this one is written to, and a test
/// that named both with one id would be describing a copy to and from one object.
fn copy_dst_texture(format: TextureFormat) -> Texture {
    Texture::new(
        object(24),
        device(),
        TextureDescriptor::new_2d(
            4,
            4,
            format,
            TextureUsage::COPY_DST.union(TextureUsage::COLOR_ATTACHMENT),
        ),
    )
}

/// A 4x4 RGBA8 buffer-to-texture copy inside a 1024-byte buffer.
///
/// One row of the region is 4 texels of 4 bytes, so `bytes_per_row` is stated
/// generously and the *alignment* of that number is what the device checks. The
/// footprint is `bytes_per_row * rows_per_image`, which is why the buffer is
/// sized for the generous pitch rather than for the tight one.
fn buffer_texture_copy(bytes_per_row: u32, buffer_size: u64) -> BufferTextureCopy {
    BufferTextureCopy {
        buffer: buffer_with(BufferUsage::COPY_SRC, buffer_size),
        buffer_offset: 0,
        bytes_per_row,
        rows_per_image: 4,
        texture: copy_dst_texture(TextureFormat::Rgba8Unorm),
        texture_subresource: color_layers(1),
        texture_origin: origin(),
        extent: Extent3d::d2(4, 4),
    }
}

/// A multisampled 4x4 renderable texture.
///
/// Separate from [`renderable_texture`] rather than a parameter on it, because
/// `sample_count` is the one field that makes a texture a resolve *source* — and
/// a scope that resolves needs both a multisampled source and a legal target, so
/// the two are always built together and never at a default.
fn multisampled_renderable(format: TextureFormat, sample_count: u32) -> Texture {
    let mut descriptor = TextureDescriptor::new_2d(
        4,
        4,
        format,
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
    );
    descriptor.sample_count = sample_count;
    Texture::new(object(23), device(), descriptor)
}

/// A one-mip, one-layer color view of a 4x4 texture.
fn color_view_of(texture: &Texture) -> TextureView {
    TextureView::new(
        object(21),
        device(),
        texture.clone(),
        TextureViewDescriptor::new(TextureViewDimension::D2, TextureAspects::COLOR, 0, 1, 0, 1),
    )
}

/// A one-attachment scope: location 0, cleared to opaque black, stored.
fn color_scope(label: &str) -> RasterScopeDescriptor {
    RasterScopeDescriptor::new().with_label(label).with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            view: ColorAttachmentView::Texture(color_view_of(&renderable_texture(
                TextureFormat::Rgba8Unorm,
            ))),
            load: LoadOp::Clear(crate::api::command::ColorClearValue::Float([
                0.0, 0.0, 0.0, 1.0,
            ])),
            store: StoreOp::Store,
            resolve: None,
        },
    )
}

fn vertex_module(id: u64) -> ShaderModule {
    let artifact = ShaderArtifact::new(
        ShaderStage::Vertex,
        "main",
        ShaderCode::Wgsl(Arc::from("@vertex fn main() {}")),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new().with_writes_position(true),
        ShaderRequirements::new(),
        ArtifactHash([3; 32]),
        ArtifactProducerId("fluxel-shaderc".to_string()),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    );
    ShaderModule::new(
        object(id),
        device(),
        artifact.clone(),
        crate::base::mock::module_backend_for_test(&artifact),
    )
}

/// A one-slot layout: slot 0 is a uniform buffer visible to the vertex stage.
fn uniform_layout(compatibility: u64) -> BindGroupLayout {
    BindGroupLayout::new(
        object(30 + compatibility),
        device(),
        BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 16 },
        )])
        .canonicalized(),
        BindGroupLayoutCompatibilityId::new(compatibility),
        LayoutFingerprint([1; 32]),
    )
}

/// An interface whose only group is the given layout.
fn interface_of(layout: BindGroupLayout) -> PipelineInterface {
    PipelineInterface::new(
        object(40),
        device(),
        PipelineInterfaceDescriptor::new(vec![layout]),
        PipelineInterfaceCompatibilityId::new(1),
        LayoutFingerprint([2; 32]),
    )
}

/// A pipeline that renders into exactly one RGBA8 color target.
fn raster_pipeline(layout: BindGroupLayout) -> RasterPipeline {
    RasterPipeline::new(
        object(50),
        device(),
        RasterPipelineDescriptor::new(vertex_module(60), interface_of(layout)).with_color_target(
            ShaderLocation::new(0),
            ColorTargetState::new(TextureFormat::Rgba8Unorm),
        ),
    )
}

/// The same pipeline, but rendering into a format [`color_scope`] does not attach.
fn mismatched_pipeline(layout: BindGroupLayout) -> RasterPipeline {
    RasterPipeline::new(
        object(51),
        device(),
        RasterPipelineDescriptor::new(vertex_module(61), interface_of(layout)).with_color_target(
            ShaderLocation::new(0),
            ColorTargetState::new(TextureFormat::Bgra8Unorm),
        ),
    )
}

/// A group that fills slot 0 with a 16-byte uniform range.
fn uniform_group(layout: BindGroupLayout) -> BindGroup {
    BindGroup::new(
        object(70),
        device(),
        BindGroupDescriptor::new(layout).with_entry(BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(
                buffer_with(BufferUsage::UNIFORM, 64),
                BufferRange::new(0, 16),
            )),
        )),
    )
}

/// A 64-byte vertex source, bound at slot 0 by the tests that need one.
fn vertex_binding() -> BufferBinding {
    BufferBinding::new(
        buffer_with(BufferUsage::VERTEX, 64),
        BufferRange::new(0, 64),
    )
}

/// A frame attachment on the test device.
fn frame_attachment(format: TextureFormat) -> FrameAttachment {
    FrameAttachment::new(
        AcquiredFrameId::new(device(), 1),
        device(),
        format,
        Extent3d::d2(4, 4),
    )
}
