//! Portable state, usage, dimension, and format lowering.
//!
//! Device discovery and HAL bootstrap belong to `device_open`; this module is
//! intentionally side-effect free so resource and command lowering share one
//! portable-to-native mapping without opening native objects.

use super::*;

pub(crate) fn buffer_state(state: ResourceAccessState) -> Result<wgt::BufferUses, String> {
    Ok(match state {
        ResourceAccessState::Undefined => wgt::BufferUses::empty(),
        ResourceAccessState::ShaderStorageRead => wgt::BufferUses::STORAGE_READ_ONLY,
        ResourceAccessState::ShaderStorageWrite | ResourceAccessState::ShaderStorageReadWrite => {
            wgt::BufferUses::STORAGE_READ_WRITE
        }
        ResourceAccessState::UniformRead => wgt::BufferUses::UNIFORM,
        ResourceAccessState::VertexRead => wgt::BufferUses::VERTEX,
        ResourceAccessState::IndexRead => wgt::BufferUses::INDEX,
        ResourceAccessState::IndirectRead => wgt::BufferUses::INDIRECT,
        ResourceAccessState::CopySource => wgt::BufferUses::COPY_SRC,
        ResourceAccessState::CopyDestination => wgt::BufferUses::COPY_DST,
        _ => return Err("texture-only state used for buffer transition".into()),
    })
}

pub(crate) fn texture_state(state: ResourceAccessState) -> Result<wgt::TextureUses, String> {
    Ok(match state {
        ResourceAccessState::Undefined => wgt::TextureUses::UNINITIALIZED,
        ResourceAccessState::ColorAttachmentRead
        | ResourceAccessState::ColorAttachmentWrite
        | ResourceAccessState::ColorAttachmentReadWrite => wgt::TextureUses::COLOR_TARGET,
        ResourceAccessState::DepthStencilRead => wgt::TextureUses::DEPTH_STENCIL_READ,
        ResourceAccessState::DepthStencilWrite | ResourceAccessState::DepthStencilReadWrite => {
            wgt::TextureUses::DEPTH_STENCIL_WRITE
        }
        ResourceAccessState::ShaderSampledRead => wgt::TextureUses::RESOURCE,
        ResourceAccessState::ShaderStorageRead => wgt::TextureUses::STORAGE_READ_ONLY,
        ResourceAccessState::ShaderStorageWrite => wgt::TextureUses::STORAGE_WRITE_ONLY,
        ResourceAccessState::ShaderStorageReadWrite => wgt::TextureUses::STORAGE_READ_WRITE,
        ResourceAccessState::CopySource => wgt::TextureUses::COPY_SRC,
        ResourceAccessState::CopyDestination => wgt::TextureUses::COPY_DST,
        ResourceAccessState::Present => wgt::TextureUses::PRESENT,
        _ => return Err("buffer-only state used for texture transition".into()),
    })
}

pub(crate) fn resource_error(
    backend: Backend,
    error: impl core::fmt::Display,
) -> ResourceCreateError {
    ResourceCreateError::NativeFailure {
        backend,
        reason: error.to_string(),
    }
}

pub(crate) fn validate_texture_capabilities(
    caps: wgpu_hal::TextureFormatCapabilities,
    descriptor: TextureDescriptor,
) -> Result<(), ResourceCreateError> {
    let usage = descriptor.usage;
    let supported = (!usage.contains(TextureUsageKind::Sampled)
        || caps.contains(wgpu_hal::TextureFormatCapabilities::SAMPLED))
        && (!usage.contains(TextureUsageKind::StorageRead)
            || caps.intersects(
                wgpu_hal::TextureFormatCapabilities::STORAGE_READ_ONLY
                    | wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE,
            ))
        && (!usage.contains(TextureUsageKind::StorageWrite)
            || caps.intersects(
                wgpu_hal::TextureFormatCapabilities::STORAGE_WRITE_ONLY
                    | wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE,
            ))
        && (!(usage.contains(TextureUsageKind::StorageRead)
            && usage.contains(TextureUsageKind::StorageWrite))
            || caps.contains(wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE))
        && (!usage.contains(TextureUsageKind::ColorAttachment)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COLOR_ATTACHMENT))
        && (!usage.contains(TextureUsageKind::DepthStencilAttachment)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::DEPTH_STENCIL_ATTACHMENT))
        && (!usage.contains(TextureUsageKind::CopySource)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COPY_SRC))
        && (!usage.contains(TextureUsageKind::CopyDestination)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COPY_DST));
    let sample_supported = match descriptor.texture.sample_count {
        1 => true,
        2 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X2),
        4 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X4),
        8 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X8),
        16 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X16),
        _ => false,
    };
    if supported && sample_supported {
        Ok(())
    } else {
        Err(ResourceCreateError::InvalidDescriptor {
            resource: crate::ResourceKind::Texture,
            reason: crate::InvalidResourceReason::IncompatibleUsage,
        })
    }
}

pub(crate) fn lower_buffer_usage(usage: BufferUsage) -> wgt::BufferUses {
    let mut native = wgt::BufferUses::empty();
    for (kind, flag) in [
        (BufferUsageKind::Uniform, wgt::BufferUses::UNIFORM),
        (
            BufferUsageKind::StorageRead,
            wgt::BufferUses::STORAGE_READ_ONLY,
        ),
        (
            BufferUsageKind::StorageWrite,
            wgt::BufferUses::STORAGE_READ_WRITE,
        ),
        (BufferUsageKind::Vertex, wgt::BufferUses::VERTEX),
        (BufferUsageKind::Index, wgt::BufferUses::INDEX),
        (BufferUsageKind::Indirect, wgt::BufferUses::INDIRECT),
        (BufferUsageKind::CopySource, wgt::BufferUses::COPY_SRC),
        (BufferUsageKind::CopyDestination, wgt::BufferUses::COPY_DST),
    ] {
        if usage.contains(kind) {
            native |= flag;
        }
    }
    native
}

pub(crate) fn buffer_usage_from_native(native: wgt::BufferUses) -> BufferUsage {
    let mut kinds = Vec::new();
    if native.contains(wgt::BufferUses::UNIFORM) {
        kinds.push(BufferUsageKind::Uniform);
    }
    if native.intersects(wgt::BufferUses::STORAGE_READ_ONLY | wgt::BufferUses::STORAGE_READ_WRITE) {
        kinds.push(BufferUsageKind::StorageRead);
    }
    if native.contains(wgt::BufferUses::STORAGE_READ_WRITE) {
        kinds.push(BufferUsageKind::StorageWrite);
    }
    if native.contains(wgt::BufferUses::VERTEX) {
        kinds.push(BufferUsageKind::Vertex);
    }
    if native.contains(wgt::BufferUses::INDEX) {
        kinds.push(BufferUsageKind::Index);
    }
    if native.contains(wgt::BufferUses::INDIRECT) {
        kinds.push(BufferUsageKind::Indirect);
    }
    if native.contains(wgt::BufferUses::COPY_SRC) {
        kinds.push(BufferUsageKind::CopySource);
    }
    if native.contains(wgt::BufferUses::COPY_DST) {
        kinds.push(BufferUsageKind::CopyDestination);
    }
    BufferUsage::from_kinds(kinds)
}

pub(crate) fn lower_texture_usage(usage: TextureUsage) -> wgt::TextureUses {
    let mut native = wgt::TextureUses::empty();
    if usage.contains(TextureUsageKind::Sampled) {
        native |= wgt::TextureUses::RESOURCE;
    }
    match (
        usage.contains(TextureUsageKind::StorageRead),
        usage.contains(TextureUsageKind::StorageWrite),
    ) {
        (true, true) => native |= wgt::TextureUses::STORAGE_READ_WRITE,
        (true, false) => native |= wgt::TextureUses::STORAGE_READ_ONLY,
        (false, true) => native |= wgt::TextureUses::STORAGE_WRITE_ONLY,
        _ => {}
    }
    if usage.contains(TextureUsageKind::ColorAttachment) {
        native |= wgt::TextureUses::COLOR_TARGET;
    }
    if usage.contains(TextureUsageKind::DepthStencilAttachment) {
        native |= wgt::TextureUses::DEPTH_STENCIL_WRITE;
    }
    if usage.contains(TextureUsageKind::CopySource) {
        native |= wgt::TextureUses::COPY_SRC;
    }
    if usage.contains(TextureUsageKind::CopyDestination) {
        native |= wgt::TextureUses::COPY_DST;
    }
    native
}

pub(crate) fn texture_usage_from_native(
    native: wgt::TextureUses,
    format: TextureFormat,
) -> TextureUsage {
    let mut kinds = Vec::new();
    if native.contains(wgt::TextureUses::RESOURCE) {
        kinds.push(TextureUsageKind::Sampled);
    }
    if native.intersects(wgt::TextureUses::STORAGE_READ_ONLY | wgt::TextureUses::STORAGE_READ_WRITE)
    {
        kinds.push(TextureUsageKind::StorageRead);
    }
    if native
        .intersects(wgt::TextureUses::STORAGE_WRITE_ONLY | wgt::TextureUses::STORAGE_READ_WRITE)
    {
        kinds.push(TextureUsageKind::StorageWrite);
    }
    if native.contains(wgt::TextureUses::COLOR_TARGET) && format != TextureFormat::Depth32Float {
        kinds.push(TextureUsageKind::ColorAttachment);
    }
    if native.contains(wgt::TextureUses::DEPTH_STENCIL_WRITE)
        && format == TextureFormat::Depth32Float
    {
        kinds.push(TextureUsageKind::DepthStencilAttachment);
    }
    if native.contains(wgt::TextureUses::COPY_SRC) {
        kinds.push(TextureUsageKind::CopySource);
    }
    if native.contains(wgt::TextureUses::COPY_DST) {
        kinds.push(TextureUsageKind::CopyDestination);
    }
    TextureUsage::from_kinds(kinds)
}

pub(crate) fn lower_dimension(dimension: TextureDimension) -> wgt::TextureDimension {
    match dimension {
        TextureDimension::D1 => wgt::TextureDimension::D1,
        TextureDimension::D2 => wgt::TextureDimension::D2,
        TextureDimension::D3 => wgt::TextureDimension::D3,
        _ => unreachable!("validated dimension"),
    }
}

pub(crate) fn lower_format(format: TextureFormat) -> wgt::TextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => wgt::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgt::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgt::TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float => wgt::TextureFormat::Rgba16Float,
        TextureFormat::Depth32Float => wgt::TextureFormat::Depth32Float,
        _ => unreachable!("known portable format"),
    }
}
