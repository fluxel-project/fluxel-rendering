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
//! * A verb that stops at a device-dependent question is driven until it stops, and
//!   the stop is asserted by the panic message that names the missing device
//!   answer. That is the only observable it has, and it is the honest one: the
//!   portable half of the verb has already run and returned by then.

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
use crate::api::command::attachment::ColorAttachmentView;
use crate::api::command::{
    BufferCopy, ColorAttachment, CommandRecorder, LoadOp, RasterScopeDescriptor, StoreOp,
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
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::{Extent3d, Texture, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{BufferUploadDescriptor, UploadDescriptor, UploadJob};
use crate::api::resource::view::{TextureView, TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderLocation, ShaderModule, ShaderRequirements, ShaderStage,
    ShaderStages,
};

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
fn recorder() -> CommandRecorder {
    CommandRecorder::new(object(1), device(), Label(Some("test".to_string())))
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
    Buffer::new(object(id), device(), BufferDescriptor::new(size, usage))
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
    ShaderModule::new(
        object(id),
        device(),
        ShaderArtifact::new(
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
        ),
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
