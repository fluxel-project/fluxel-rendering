//! Portable validation and actual-use derivation for recorded commands.
//!
//! This module owns the checks rhi-design sections 32.3, 33, and 34 describe,
//! plus the resource-use summary from section 37.
//!
//! # Why it is separate from the recorder
//!
//! The recorder owns the state machine; this module owns the arithmetic. Every
//! function here takes the recorder's portable state by reference, never the
//! recorder, so no check can mutate what it is judging.
//!
//! # What "actual use" means
//!
//! Use is generated where a resource is actually consumed — at `draw`,
//! `dispatch`, or a copy — not at `set_bind_group`. A bind group holding ten
//! resources whose shaders reference three produces three uses. For an array
//! binding the whole declared element count is covered, because P0 refuses to
//! narrow it from backend reflection: a backend that silently reads fewer
//! elements than the binding declares would otherwise hide a hazard.

use std::ops::Range;

use super::super::binding::{
    BindGroup, BindGroupEntry, BindingKind, BindingResource, BufferBindingAccess, StorageAccess,
};
use super::super::capability::EnabledCapabilities;
use super::super::format::{RouteQuery, RouteSupport};
use super::super::graph_bridge::{
    AccessMask, BufferUse, FrameAttachmentUse, PipelineScope, ResourceUse, TextureUse,
    TextureUseIntent,
};
use super::super::pipeline::{RenderTargetSignature, ResourceMergeOutcome, VertexStepMode};
use super::super::platform::{DeviceIdentity, RhiError, RhiErrorKind, RhiResult};
use super::super::resource::{
    BufferRange, BufferUsage, ReadbackRequest, Texture, TextureAspect, TextureAspects,
    TextureSubresourceLayers, TextureSubresourceRange, TextureUsage, UploadJob,
};
use super::attachment::{
    ColorAttachmentView, DepthAttachmentMode, RasterScopeDescriptor, StencilAttachmentMode,
};
use super::values::{
    BufferCopy, BufferTextureCopy, TextureBlit, TextureCopy, TextureResolve,
};
use super::{values, ScopeState};

pub(crate) use super::values::{validate_draw_ranges, validate_indexed_draw_ranges};

/// The aspect set that addresses exactly one aspect.
fn aspect_set(aspect: TextureAspect) -> TextureAspects {
    match aspect {
        TextureAspect::Color => TextureAspects::COLOR,
        TextureAspect::Depth => TextureAspects::DEPTH,
        TextureAspect::Stencil => TextureAspects::STENCIL,
    }
}

/// The subresource range one copy layer selection addresses.
fn layers_range(layers: &TextureSubresourceLayers) -> TextureSubresourceRange {
    TextureSubresourceRange {
        aspects: aspect_set(layers.aspect),
        base_mip: layers.mip_level,
        mip_count: 1,
        base_layer: layers.base_layer,
        layer_count: layers.layer_count,
    }
}

/// The access bits a storage buffer's declared access implies.
fn buffer_storage_access(access: BufferBindingAccess) -> AccessMask {
    match access {
        BufferBindingAccess::ReadOnly => AccessMask::SHADER_READ,
        BufferBindingAccess::ReadWrite => AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE),
    }
}

/// The access bits a storage texture's declared access implies.
fn texture_storage_access(access: StorageAccess) -> AccessMask {
    match access {
        StorageAccess::ReadOnly => AccessMask::SHADER_READ,
        StorageAccess::WriteOnly => AccessMask::SHADER_WRITE,
        StorageAccess::ReadWrite => AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE),
    }
}

/// The access bits and texture intent a merged binding outcome contributes.
///
/// `None` means the binding carries no memory this library tracks, which is the
/// case for a sampler: it occupies a layout slot but creates no hazard.
fn outcome_access(outcome: &ResourceMergeOutcome) -> Option<(AccessMask, TextureUseIntent)> {
    match &outcome.kind {
        BindingKind::UniformBuffer { .. } => {
            Some((AccessMask::UNIFORM_READ, TextureUseIntent::ShaderRead))
        }
        BindingKind::StorageBuffer { access, .. } => {
            Some((buffer_storage_access(*access), TextureUseIntent::ShaderRead))
        }
        BindingKind::SampledTexture { .. } => {
            Some((AccessMask::SHADER_READ, TextureUseIntent::ShaderRead))
        }
        BindingKind::StorageTexture { access, .. } => Some((
            texture_storage_access(*access),
            TextureUseIntent::ShaderReadWrite,
        )),
        // A sampler and any kind a later version adds that carries no memory.
        _ => None,
    }
}

/// The attachment signature of an open raster scope.
pub(crate) fn target_signature_of(desc: &RasterScopeDescriptor) -> RenderTargetSignature {
    let color_formats = desc
        .colors
        .iter()
        .map(|slot| slot.as_ref().map(|slot| slot.view.format()))
        .collect::<Vec<_>>();
    let depth_stencil_format = desc
        .depth_stencil
        .as_ref()
        .map(|attachment| attachment.view.format());
    // Every main attachment already agreed on one sample count when the scope
    // opened, so the first live attachment answers for all of them.
    let sample_count = desc
        .colors
        .iter()
        .flatten()
        .map(|slot| slot.view.sample_count())
        .next()
        .or_else(|| {
            desc.depth_stencil
                .as_ref()
                .map(|attachment| attachment.view.sample_count())
        })
        .unwrap_or(1);
    RenderTargetSignature::new(color_formats, depth_stencil_format, sample_count)
}

/// Checks that every group a pipeline's shaders reference is bound and that its
/// layout is the one the pipeline's interface declares.
fn check_shader_groups(
    scope: &ScopeState,
    used: &[ResourceMergeOutcome],
    interface: &super::super::pipeline::PipelineInterface,
) -> RhiResult<()> {
    for outcome in used {
        let index = outcome.group.get() as usize;
        let Some(Some(bound)) = scope.bind_groups.get(index) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the shader references group {} but no bind group is bound there",
                    outcome.group.get()
                ),
            ));
        };
        let Some(expected) = interface.group(outcome.group) else {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "the pipeline interface declares no group at index {}",
                    outcome.group.get()
                ),
            ));
        };
        if bound.layout().compatibility_id() != expected.compatibility_id() {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "the bind group at index {} is not compatible with the pipeline's layout",
                    outcome.group.get()
                ),
            ));
        }
    }
    Ok(())
}

/// Checks that a group's dynamic offsets address real data.
///
/// An offset past the end of its bound range would make one recording mean
/// "clamp" on one backend and "fault" on another, so it is refused here rather
/// than left to a driver.
pub(crate) fn check_dynamic_offsets(group: &BindGroup, offsets: &[u32]) -> RhiResult<()> {
    let layout = group.layout();
    let descriptor = group.descriptor();
    let mut index = 0usize;
    for slot in &layout.descriptor().entries {
        if !slot.dynamic_offset {
            continue;
        }
        let Some(offset) = offsets.get(index) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the group is missing a dynamic offset",
            ));
        };
        index += 1;
        let Some(min_size) = slot.kind.min_size() else {
            continue;
        };
        let Some(entry) = descriptor
            .entries
            .iter()
            .find(|entry| entry.slot == slot.slot)
        else {
            continue;
        };
        let size = match &entry.resource {
            BindingResource::Buffer(binding) => binding.range.size,
            BindingResource::BufferArray(bindings) => {
                bindings.first().map(|binding| binding.range.size).unwrap_or(0)
            }
            _ => continue,
        };
        if u64::from(*offset).saturating_add(min_size) > size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a dynamic offset addresses past the end of its bound range",
            ));
        }
    }
    Ok(())
}

/// Appends the uses one bind group entry contributes.
fn push_entry_uses(
    entry: &BindGroupEntry,
    stages: PipelineScope,
    access: AccessMask,
    intent: TextureUseIntent,
    uses: &mut Vec<ResourceUse>,
) {
    match &entry.resource {
        BindingResource::Buffer(binding) => uses.push(ResourceUse::Buffer(BufferUse {
            buffer: binding.buffer.clone(),
            range: binding.range,
            stages,
            access,
        })),
        BindingResource::BufferArray(bindings) => {
            for binding in bindings {
                uses.push(ResourceUse::Buffer(BufferUse {
                    buffer: binding.buffer.clone(),
                    range: binding.range,
                    stages,
                    access,
                }));
            }
        }
        BindingResource::Texture(view) => uses.push(ResourceUse::Texture(TextureUse {
            texture: view.texture().clone(),
            subresources: view.descriptor().subresource(),
            stages,
            access,
            intent,
        })),
        BindingResource::TextureArray(views) => {
            for view in views {
                uses.push(ResourceUse::Texture(TextureUse {
                    texture: view.texture().clone(),
                    subresources: view.descriptor().subresource(),
                    stages,
                    access,
                    intent,
                }));
            }
        }
        // A sampler carries no memory hazard, so it contributes no use. It still
        // occupies a layout slot, which is why `check_shader_groups` sees it.
        _ => {}
    }
}

/// Appends the uses every shader-referenced binding of a scope contributes.
fn push_shader_uses(
    scope: &ScopeState,
    used: &[ResourceMergeOutcome],
    stages: PipelineScope,
    uses: &mut Vec<ResourceUse>,
) {
    for outcome in used {
        let index = outcome.group.get() as usize;
        let Some(Some(group)) = scope.bind_groups.get(index) else {
            continue;
        };
        let Some(entry) = group
            .descriptor()
            .entries
            .iter()
            .find(|entry| entry.slot == outcome.slot)
        else {
            continue;
        };
        if let Some((access, intent)) = outcome_access(outcome) {
            push_entry_uses(entry, stages, access, intent, uses);
        }
    }
}

/// The texture or frame use one color attachment contributes.
fn attachment_use(
    view: &ColorAttachmentView,
    access: AccessMask,
    intent: TextureUseIntent,
) -> ResourceUse {
    match view {
        ColorAttachmentView::Texture(view) => ResourceUse::Texture(TextureUse {
            texture: view.texture().clone(),
            subresources: view.descriptor().subresource(),
            stages: PipelineScope::FRAGMENT,
            access,
            intent,
        }),
        ColorAttachmentView::Frame(frame) => ResourceUse::Frame(FrameAttachmentUse {
            frame: frame.frame_id(),
            stages: PipelineScope::FRAGMENT,
            access,
        }),
    }
}

/// Appends the uses a scope's attachment set contributes to every draw in it.
fn push_attachment_uses(desc: &RasterScopeDescriptor, uses: &mut Vec<ResourceUse>) {
    for slot in desc.colors.iter().flatten() {
        // A color attachment is written by the draw and read whenever blending
        // or a load happens, so both bits are present regardless of load op.
        let access = AccessMask::COLOR_READ.union(AccessMask::COLOR_WRITE);
        uses.push(attachment_use(
            &slot.view,
            access,
            TextureUseIntent::ColorAttachment,
        ));
        if let Some(resolve) = &slot.resolve {
            uses.push(attachment_use(
                resolve,
                AccessMask::COLOR_WRITE,
                TextureUseIntent::ResolveDst,
            ));
        }
    }

    let Some(depth_stencil) = &desc.depth_stencil else {
        return;
    };
    let mut access = AccessMask::from_bits(0);
    match depth_stencil.depth {
        Some(DepthAttachmentMode::ReadOnly) => access = access.union(AccessMask::DEPTH_READ),
        Some(DepthAttachmentMode::ReadWrite { .. }) => {
            access = access
                .union(AccessMask::DEPTH_READ)
                .union(AccessMask::DEPTH_WRITE)
        }
        None => {}
    }
    match depth_stencil.stencil {
        Some(StencilAttachmentMode::ReadOnly) => access = access.union(AccessMask::STENCIL_READ),
        Some(StencilAttachmentMode::ReadWrite { .. }) => {
            access = access
                .union(AccessMask::STENCIL_READ)
                .union(AccessMask::STENCIL_WRITE)
        }
        None => {}
    }
    if access.is_empty() {
        return;
    }
    let intent =
        if access.contains(AccessMask::DEPTH_WRITE) || access.contains(AccessMask::STENCIL_WRITE) {
            TextureUseIntent::DepthStencilWrite
        } else {
            TextureUseIntent::DepthStencilRead
        };
    uses.push(ResourceUse::Texture(TextureUse {
        texture: depth_stencil.view.texture().clone(),
        subresources: depth_stencil.view.descriptor().subresource(),
        stages: PipelineScope::FRAGMENT,
        access,
        intent,
    }));
}

/// Validates a draw and derives its actual resource use.
///
/// `primary_extent` is the fetch bound the draw actually reads: the vertex range
/// for a non-indexed draw, the index range for an indexed one. An
/// instance-stepped slot is bounded by `instance_range` instead.
pub(crate) fn draw_uses(
    scope: &ScopeState,
    primary_extent: &Range<u32>,
    instance_range: &Range<u32>,
    indices: Option<&Range<u32>>,
) -> RhiResult<Vec<ResourceUse>> {
    let Some(pipeline) = &scope.raster_pipeline else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "no raster pipeline is bound",
        ));
    };
    let Some(desc) = &scope.descriptor else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the scope has no attachment set",
        ));
    };

    check_shader_groups(scope, pipeline.used_bindings(), pipeline.interface())?;

    let mut uses = Vec::new();

    // Vertex fetch. Every slot the pipeline declares must be bound, carry
    // VERTEX usage, and be large enough for the fetch this draw performs.
    for (slot, layout) in pipeline.descriptor().vertex_input.buffers.iter().enumerate() {
        let Some(Some(binding)) = scope.vertex_buffers.get(slot) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("vertex slot {slot} is declared by the pipeline but not bound"),
            ));
        };
        if !binding.buffer.descriptor().usage.contains(BufferUsage::VERTEX) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("vertex slot {slot} is bound to a buffer without VERTEX usage"),
            ));
        }
        let extent = match layout.step_mode {
            VertexStepMode::Vertex => primary_extent.end,
            VertexStepMode::Instance => instance_range.end,
        };
        let needed = u64::from(extent) * layout.stride;
        if needed > binding.range.size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "vertex slot {slot} needs {needed} bytes but its binding exposes {}",
                    binding.range.size
                ),
            ));
        }
        uses.push(ResourceUse::Buffer(BufferUse {
            buffer: binding.buffer.clone(),
            range: BufferRange::new(binding.range.offset, needed),
            stages: PipelineScope::VERTEX,
            access: AccessMask::VERTEX_READ,
        }));
    }

    if let Some(indices) = indices {
        let Some((binding, format)) = &scope.index_buffer else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an indexed draw needs a bound index buffer",
            ));
        };
        if !binding.buffer.descriptor().usage.contains(BufferUsage::INDEX) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the index buffer was not created with INDEX usage",
            ));
        }
        let needed = u64::from(indices.end) * u64::from(format.byte_size());
        if needed > binding.range.size {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the index range needs {needed} bytes but its binding exposes {}",
                    binding.range.size
                ),
            ));
        }
        uses.push(ResourceUse::Buffer(BufferUse {
            buffer: binding.buffer.clone(),
            range: BufferRange::new(binding.range.offset, needed),
            stages: PipelineScope::VERTEX,
            access: AccessMask::INDEX_READ,
        }));
    }

    // A strip topology makes the index width part of the pipeline, so a mismatch
    // would silently reinterpret the same bytes as a different count.
    if pipeline.descriptor().primitive.topology.is_strip() {
        let declared = pipeline.descriptor().primitive.strip_index_format;
        let bound = scope.index_buffer.as_ref().map(|(_, format)| *format);
        if declared != bound {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the pipeline was built for {declared:?} strip indices but {bound:?} is bound"
                ),
            ));
        }
    }

    // A merged outcome does not record which stage referenced the slot, so the
    // use names every stage this pipeline can run in.
    let mut stages = PipelineScope::VERTEX;
    if pipeline.descriptor().fragment.is_some() {
        stages = stages.union(PipelineScope::FRAGMENT);
    }
    push_shader_uses(scope, pipeline.used_bindings(), stages, &mut uses);
    push_attachment_uses(desc, &mut uses);
    Ok(uses)
}

/// Validates a dispatch and derives its actual resource use.
pub(crate) fn dispatch_uses(
    scope: &ScopeState,
    x: u32,
    y: u32,
    z: u32,
    max_per_dimension: Option<u64>,
) -> RhiResult<Vec<ResourceUse>> {
    let Some(pipeline) = &scope.compute_pipeline else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "no compute pipeline is bound",
        ));
    };
    if let Some(limit) = max_per_dimension {
        for (value, axis) in [(x, "x"), (y, "y"), (z, "z")] {
            if u64::from(value) > limit {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!("dispatch {axis} of {value} exceeds the device's limit of {limit}"),
                ));
            }
        }
    }
    check_shader_groups(scope, pipeline.used_bindings(), pipeline.interface())?;

    let mut uses = Vec::new();
    push_shader_uses(
        scope,
        pipeline.used_bindings(),
        PipelineScope::COMPUTE,
        &mut uses,
    );
    Ok(uses)
}

/// Validates an upload job and returns its uses.
pub(crate) fn upload_uses(upload: &UploadJob) -> RhiResult<Vec<ResourceUse>> {
    let mut uses = Vec::new();
    match upload.descriptor() {
        super::super::resource::UploadDescriptor::Buffer(descriptor) => {
            if !descriptor
                .dst
                .descriptor()
                .usage
                .contains(BufferUsage::COPY_DST)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the upload destination was not created with COPY_DST",
                ));
            }
            let size = descriptor.bytes.len() as u64;
            let range = BufferRange::new(descriptor.dst_offset, size);
            if !range.covers(&descriptor.dst) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the upload destination range is outside the buffer",
                ));
            }
            uses.push(ResourceUse::Buffer(BufferUse {
                buffer: descriptor.dst.clone(),
                range,
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_WRITE,
            }));
        }
        super::super::resource::UploadDescriptor::Texture(descriptor) => {
            if !descriptor
                .dst
                .descriptor()
                .usage
                .contains(TextureUsage::COPY_DST)
            {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the upload destination was not created with COPY_DST",
                ));
            }
            uses.push(ResourceUse::Texture(TextureUse {
                texture: descriptor.dst.clone(),
                subresources: layers_range(&descriptor.subresource),
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_WRITE,
                intent: TextureUseIntent::CopyDst,
            }));
        }
    }
    Ok(uses)
}

/// The devices every object a readback request names belongs to.
///
/// The recorder checks these before it encodes, because a readback is the one
/// command whose result leaves the GPU behind: a cross-device source would
/// otherwise surface only as a ticket that never completes.
pub(crate) fn readback_devices(request: &ReadbackRequest) -> Vec<DeviceIdentity> {
    match request {
        ReadbackRequest::Buffer { src, .. } => vec![src.device_identity()],
        ReadbackRequest::Texture { src, .. } => vec![src.device_identity()],
    }
}

/// Validates a readback request and returns its uses.
pub(crate) fn readback_uses(request: &ReadbackRequest) -> RhiResult<Vec<ResourceUse>> {
    let mut uses = Vec::new();
    match request {
        ReadbackRequest::Buffer { src, range, .. } => {
            if !src.descriptor().usage.contains(BufferUsage::COPY_SRC) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the readback source was not created with COPY_SRC",
                ));
            }
            if !range.covers(src) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the readback range is outside the buffer",
                ));
            }
            uses.push(ResourceUse::Buffer(BufferUse {
                buffer: src.clone(),
                range: *range,
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_READ,
            }));
        }
        ReadbackRequest::Texture {
            src, subresource, ..
        } => {
            if !src.descriptor().usage.contains(TextureUsage::COPY_SRC) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the readback source was not created with COPY_SRC",
                ));
            }
            uses.push(ResourceUse::Texture(TextureUse {
                texture: src.clone(),
                subresources: layers_range(subresource),
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_READ,
                intent: TextureUseIntent::CopySrc,
            }));
        }
    }
    Ok(uses)
}

/// Validates a buffer-to-buffer copy and returns its uses.
pub(crate) fn copy_buffer_uses(copy: &BufferCopy) -> RhiResult<Vec<ResourceUse>> {
    if copy.size == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a copy must move at least one byte",
        ));
    }
    let src_range = BufferRange::new(copy.src_offset, copy.size);
    let dst_range = BufferRange::new(copy.dst_offset, copy.size);
    if !src_range.covers(&copy.src) || !dst_range.covers(&copy.dst) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a copy range is outside its buffer",
        ));
    }
    if !copy.src.descriptor().usage.contains(BufferUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the copy source was not created with COPY_SRC",
        ));
    }
    if !copy.dst.descriptor().usage.contains(BufferUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the copy destination was not created with COPY_DST",
        ));
    }
    // One command reading and writing the same bytes is undefined on every
    // backend, so it is refused here rather than left to a driver.
    if copy.src.id() == copy.dst.id()
        && values::byte_ranges_overlap(copy.src_offset, copy.size, copy.dst_offset, copy.size)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a buffer copy may not overlap itself",
        ));
    }
    Ok(vec![
        ResourceUse::Buffer(BufferUse {
            buffer: copy.src.clone(),
            range: src_range,
            stages: PipelineScope::COPY,
            access: AccessMask::COPY_READ,
        }),
        ResourceUse::Buffer(BufferUse {
            buffer: copy.dst.clone(),
            range: dst_range,
            stages: PipelineScope::COPY,
            access: AccessMask::COPY_WRITE,
        }),
    ])
}

/// The bytes a texture copy's buffer side must expose.
fn texture_copy_buffer_bytes(copy: &BufferTextureCopy) -> RhiResult<u64> {
    let row_bytes = super::super::resource::logical_row_bytes(
        copy.texture.descriptor().format,
        copy.extent.width,
    )
    .ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "the texture format has no portable row layout",
        )
    })?;
    let rows = u64::from(copy.extent.height.max(1));
    let depth = u64::from(copy.extent.depth.max(1));
    let overflow = || RhiError::new(RhiErrorKind::InvalidUsage, "the copy size overflows");

    if copy.rows_per_image == 0 {
        // With no inter-slice stride declared, only a single slice is
        // expressible; a depth above one would silently pack slices adjacently.
        if depth > 1 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a multi-slice texture copy needs a non-zero rows_per_image",
            ));
        }
        return row_bytes.checked_mul(rows).ok_or_else(overflow);
    }

    if u64::from(copy.bytes_per_row) < row_bytes {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "bytes_per_row is smaller than one texture row",
        ));
    }
    let stride_total = u64::from(copy.rows_per_image)
        .checked_mul(u64::from(copy.bytes_per_row))
        .and_then(|value| value.checked_mul(depth - 1))
        .ok_or_else(overflow)?;
    let last_slice = row_bytes.checked_mul(rows).ok_or_else(overflow)?;
    stride_total.checked_add(last_slice).ok_or_else(overflow)
}

/// Validates a buffer/texture copy and returns its uses.
///
/// `buffer_is_destination` picks the direction. The field set is the same either
/// way, which is why one descriptor type serves both directions.
///
/// `capabilities` is needed because section 34.2 makes two of a texel copy's
/// rules route questions rather than descriptor questions: whether a direct
/// route exists at all, and which copy-buffer layout that route demands. A
/// backend that imposes no row-pitch alignment declares no
/// `TexelCopyLayoutLimits`, and that is not the same as declaring zero — the
/// missing entry means the rule does not apply, so no portable constant is
/// substituted for it.
pub(crate) fn buffer_texture_uses(
    copy: &BufferTextureCopy,
    buffer_is_destination: bool,
    capabilities: &EnabledCapabilities,
) -> RhiResult<Vec<ResourceUse>> {
    if !copy.extent.is_non_zero() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a copy extent must be non-empty",
        ));
    }
    let (buffer_usage, texture_usage, buffer_access, texture_access, intent) =
        if buffer_is_destination {
            (
                BufferUsage::COPY_DST,
                TextureUsage::COPY_SRC,
                AccessMask::COPY_WRITE,
                AccessMask::COPY_READ,
                TextureUseIntent::CopySrc,
            )
        } else {
            (
                BufferUsage::COPY_SRC,
                TextureUsage::COPY_DST,
                AccessMask::COPY_READ,
                AccessMask::COPY_WRITE,
                TextureUseIntent::CopyDst,
            )
        };
    if !copy.buffer.descriptor().usage.contains(buffer_usage) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the buffer side of the copy has the wrong usage",
        ));
    }
    if !copy.texture.descriptor().usage.contains(texture_usage) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the texture side of the copy has the wrong usage",
        ));
    }

    // Section 34.2: the route key carries dimension, format, and aspect, so a
    // route answer for one texture says nothing about another. The direction
    // chooses the key, because a buffer-to-texture route and a texture-to-buffer
    // route are separate declarations.
    let texture = copy.texture.descriptor();
    let route_query = if buffer_is_destination {
        RouteQuery::BufferToTexture {
            dimension: texture.dimension,
            format: texture.format,
            aspect: copy.texture_subresource.aspect,
        }
    } else {
        RouteQuery::TextureToBuffer {
            dimension: texture.dimension,
            format: texture.format,
            aspect: copy.texture_subresource.aspect,
        }
    };
    match capabilities.route(&route_query) {
        RouteSupport::Unsupported => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "no direct route copies between this buffer and this texture",
            ));
        }
        RouteSupport::Supported(route) => {
            if let Some(layout) = route.texel_copy_layout() {
                // A declared zero would mean the route constrains nothing. It is
                // skipped rather than divided by, because a modulus by zero is
                // not a legality answer.
                let offset_alignment = layout.buffer_offset_alignment();
                if offset_alignment != 0 && !copy.buffer_offset.is_multiple_of(offset_alignment) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "buffer_offset does not satisfy the route's offset alignment",
                    ));
                }
                let row_alignment = layout.bytes_per_row_alignment();
                if row_alignment != 0 && !copy.bytes_per_row.is_multiple_of(row_alignment) {
                    return Err(RhiError::new(
                        RhiErrorKind::InvalidUsage,
                        "bytes_per_row does not satisfy the route's row alignment",
                    ));
                }
            }
        }
    }

    let needed = texture_copy_buffer_bytes(copy)?;
    let buffer_range = BufferRange::new(copy.buffer_offset, needed);
    if !buffer_range.covers(&copy.buffer) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the buffer side of the copy is too small for the requested extent",
        ));
    }
    Ok(vec![
        ResourceUse::Buffer(BufferUse {
            buffer: copy.buffer.clone(),
            range: buffer_range,
            stages: PipelineScope::COPY,
            access: buffer_access,
        }),
        ResourceUse::Texture(TextureUse {
            texture: copy.texture.clone(),
            subresources: layers_range(&copy.texture_subresource),
            stages: PipelineScope::COPY,
            access: texture_access,
            intent,
        }),
    ])
}

/// The two texture uses a copy-shaped command contributes.
fn texture_pair_uses(
    src: &Texture,
    src_subresource: &TextureSubresourceLayers,
    dst: &Texture,
    dst_subresource: &TextureSubresourceLayers,
    src_intent: TextureUseIntent,
    dst_intent: TextureUseIntent,
) -> Vec<ResourceUse> {
    vec![
        ResourceUse::Texture(TextureUse {
            texture: src.clone(),
            subresources: layers_range(src_subresource),
            stages: PipelineScope::COPY,
            access: AccessMask::COPY_READ,
            intent: src_intent,
        }),
        ResourceUse::Texture(TextureUse {
            texture: dst.clone(),
            subresources: layers_range(dst_subresource),
            stages: PipelineScope::COPY,
            access: AccessMask::COPY_WRITE,
            intent: dst_intent,
        }),
    ]
}

/// Validates a texture-to-texture copy and returns its uses.
pub(crate) fn texture_copy_uses(copy: &TextureCopy) -> RhiResult<Vec<ResourceUse>> {
    if !copy.extent.is_non_zero() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a copy extent must be non-empty",
        ));
    }
    if !copy.src.descriptor().usage.contains(TextureUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the copy source was not created with COPY_SRC",
        ));
    }
    if !copy.dst.descriptor().usage.contains(TextureUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the copy destination was not created with COPY_DST",
        ));
    }
    if copy.src.descriptor().format != copy.dst.descriptor().format {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture copy cannot change format",
        ));
    }
    if copy.src.id() == copy.dst.id()
        && values::texel_regions_overlap(
            copy.src_origin,
            copy.extent,
            copy.dst_origin,
            copy.extent,
        )
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture copy may not overlap itself",
        ));
    }
    Ok(texture_pair_uses(
        &copy.src,
        &copy.src_subresource,
        &copy.dst,
        &copy.dst_subresource,
        TextureUseIntent::CopySrc,
        TextureUseIntent::CopyDst,
    ))
}

/// Validates a standalone multisample resolve and returns its uses.
///
/// A resolve has its own command because it is not expressible as a copy: the
/// source is multisampled and the destination is not, so the two sides can never
/// share a sample count and no generic copy may accept them.
pub(crate) fn resolve_uses(resolve: &TextureResolve) -> RhiResult<Vec<ResourceUse>> {
    if resolve.src.id() == resolve.dst.id() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve source and destination must be different textures",
        ));
    }
    let source = resolve.src.descriptor();
    let destination = resolve.dst.descriptor();
    if source.sample_count <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve source must be multisampled",
        ));
    }
    if destination.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve destination must be single-sampled",
        ));
    }
    if source.format != destination.format {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve cannot change format",
        ));
    }
    if !source.usage.contains(TextureUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the resolve source was not created with COPY_SRC",
        ));
    }
    if !destination.usage.contains(TextureUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the resolve destination was not created with COPY_DST",
        ));
    }
    if !resolve.extent.is_non_zero() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve extent must be non-empty",
        ));
    }
    if resolve.src_subresource.layer_count != resolve.dst_subresource.layer_count {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a resolve cannot change the layer count",
        ));
    }
    Ok(texture_pair_uses(
        &resolve.src,
        &resolve.src_subresource,
        &resolve.dst,
        &resolve.dst_subresource,
        TextureUseIntent::ResolveSrc,
        TextureUseIntent::ResolveDst,
    ))
}

/// Validates a blit and returns its uses.
///
/// A blit is a copy whose source and destination boxes may differ in size, so
/// the two extents are checked independently and then against each other only
/// for overlap.
pub(crate) fn blit_uses(blit: &TextureBlit) -> RhiResult<Vec<ResourceUse>> {
    if !blit.src_extent.is_non_zero() || !blit.dst_extent.is_non_zero() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a blit extent must be non-empty",
        ));
    }
    if !blit.src.descriptor().usage.contains(TextureUsage::COPY_SRC) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the blit source was not created with COPY_SRC",
        ));
    }
    if !blit.dst.descriptor().usage.contains(TextureUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the blit destination was not created with COPY_DST",
        ));
    }
    if blit.src.id() == blit.dst.id()
        && values::texel_regions_overlap(
            blit.src_origin,
            blit.src_extent,
            blit.dst_origin,
            blit.dst_extent,
        )
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a blit may not overlap itself",
        ));
    }
    Ok(texture_pair_uses(
        &blit.src,
        &blit.src_subresource,
        &blit.dst,
        &blit.dst_subresource,
        TextureUseIntent::CopySrc,
        TextureUseIntent::CopyDst,
    ))
}

/// Merges per-command use sequences into one summary.
///
/// Two uses of the same resource, stages, access, and intent become one whose
/// range covers both. The result answers what the work touched, never in what
/// order: it is a summary, not a schedule.
pub(crate) fn merge_uses(commands: &[Vec<ResourceUse>]) -> Vec<ResourceUse> {
    let mut merged: Vec<ResourceUse> = Vec::new();
    for command in commands {
        for use_ in command {
            match merged
                .iter_mut()
                .find(|candidate| same_target(candidate, use_))
            {
                Some(existing) => union_into(existing, use_),
                None => merged.push(use_.clone()),
            }
        }
    }
    merged
}

/// Whether two uses name the same resource, stages, access, and intent.
fn same_target(left: &ResourceUse, right: &ResourceUse) -> bool {
    match (left, right) {
        (ResourceUse::Buffer(left), ResourceUse::Buffer(right)) => {
            left.buffer.id() == right.buffer.id()
                && left.stages == right.stages
                && left.access == right.access
        }
        (ResourceUse::Texture(left), ResourceUse::Texture(right)) => {
            left.texture.id() == right.texture.id()
                && left.stages == right.stages
                && left.access == right.access
                && left.intent == right.intent
        }
        (ResourceUse::Frame(left), ResourceUse::Frame(right)) => {
            left.frame == right.frame && left.stages == right.stages && left.access == right.access
        }
        _ => false,
    }
}

/// Widens one use to cover another.
fn union_into(existing: &mut ResourceUse, other: &ResourceUse) {
    match (existing, other) {
        (ResourceUse::Buffer(existing), ResourceUse::Buffer(other)) => {
            existing.range = widen_range(existing.range, other.range);
            existing.access = existing.access.union(other.access);
        }
        (ResourceUse::Texture(existing), ResourceUse::Texture(other)) => {
            existing.subresources = widen_subresources(existing.subresources, other.subresources);
            existing.access = existing.access.union(other.access);
        }
        (ResourceUse::Frame(existing), ResourceUse::Frame(other)) => {
            existing.access = existing.access.union(other.access);
        }
        _ => {}
    }
}

/// The smallest range covering both.
fn widen_range(left: BufferRange, right: BufferRange) -> BufferRange {
    let offset = left.offset.min(right.offset);
    match (left.end(), right.end()) {
        (Some(left_end), Some(right_end)) => {
            BufferRange::new(offset, right_end.max(left_end) - offset)
        }
        // An unrepresentable end means the range already runs to the end of the
        // buffer, so the union does too.
        _ => BufferRange::new(offset, u64::MAX - offset),
    }
}

/// The smallest subresource range covering both.
fn widen_subresources(
    left: TextureSubresourceRange,
    right: TextureSubresourceRange,
) -> TextureSubresourceRange {
    let base_mip = left.base_mip.min(right.base_mip);
    let mip_end = (left.base_mip + left.mip_count).max(right.base_mip + right.mip_count);
    let base_layer = left.base_layer.min(right.base_layer);
    let layer_end = (left.base_layer + left.layer_count).max(right.base_layer + right.layer_count);
    TextureSubresourceRange {
        aspects: left.aspects.union(right.aspects),
        base_mip,
        mip_count: mip_end - base_mip,
        base_layer,
        layer_count: layer_end - base_layer,
    }
}
