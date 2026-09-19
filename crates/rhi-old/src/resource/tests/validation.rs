//! Resource validation contract tests.

use super::*;

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(super) fn compute_pipeline_creation_stages_map_to_public_error_variants() {
    let cases = [
        (
            crate::imp::ComputePipelineCreateError::ShaderValidation("parse".into()),
            ComputeCreateError::ShaderValidation("parse".into()),
        ),
        (
            crate::imp::ComputePipelineCreateError::ShaderCompilation("compile".into()),
            ComputeCreateError::ShaderCompilation("compile".into()),
        ),
        (
            crate::imp::ComputePipelineCreateError::NativeObjectCreation("layout".into()),
            ComputeCreateError::NativeObjectCreation("layout".into()),
        ),
    ];
    for (native, public) in cases {
        assert_eq!(map_compute_pipeline_create_error(native), public);
    }
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(super) fn raster_pipeline_creation_stages_map_to_public_error_variants() {
    let cases = [
        (
            crate::imp::RasterPipelineCreateError::ShaderValidation("parse".into()),
            RasterCreateError::ShaderValidation("parse".into()),
        ),
        (
            crate::imp::RasterPipelineCreateError::ShaderCompilation("compile".into()),
            RasterCreateError::ShaderCompilation("compile".into()),
        ),
        (
            crate::imp::RasterPipelineCreateError::NativeObjectCreation("layout".into()),
            RasterCreateError::NativeObjectCreation("layout".into()),
        ),
    ];
    for (native, public) in cases {
        assert_eq!(map_raster_pipeline_create_error(native), public);
    }
}

pub(super) fn rejects_zero_buffer_and_empty_usage() {
    let empty = BufferDescriptor {
        buffer: BufferDesc { size: 0 },
        usage: BufferUsage::empty(),
        memory: MemoryPolicy::DeviceOnly,
    };
    assert_eq!(
        validate_buffer(empty),
        Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::ZeroSize
        ))
    );
    assert_eq!(
        validate_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 4 },
            ..empty
        }),
        Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::EmptyUsage
        ))
    );
}

pub(super) fn rejects_surface_and_incompatible_texture_contracts() {
    let base = TextureDescriptor {
        texture: TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 4,
                height: 4,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
        usage: TextureUsage::empty().with(TextureUsageKind::Present),
        memory: MemoryPolicy::DeviceOnly,
    };
    assert_eq!(
        validate_texture(base),
        Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::PresentRequiresSurface
        ))
    );
    let depth_color = TextureDescriptor {
        texture: TextureDesc {
            format: TextureFormat::Depth32Float,
            ..base.texture
        },
        usage: TextureUsage::empty().with(TextureUsageKind::ColorAttachment),
        ..base
    };
    assert_eq!(
        validate_texture(depth_color),
        Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::IncompatibleUsage
        ))
    );
}

pub(super) fn fixed_workgroup_requires_each_native_dimension_and_total_invocations() {
    assert_eq!(
        validate_compute_workgroup_limits([64, 1, 1], [63, 1, 1], 64),
        Err(ComputeCreateError::UnsupportedComputeLimits)
    );
    assert_eq!(
        validate_compute_workgroup_limits([64, 1, 1], [64, 1, 1], 63),
        Err(ComputeCreateError::UnsupportedComputeLimits)
    );
    assert_eq!(
        validate_compute_workgroup_limits([64, 0, 1], [64, 1, 1], 64),
        Err(ComputeCreateError::UnsupportedComputeLimits)
    );
    assert_eq!(
        validate_compute_workgroup_limits([64, 1, 1], [64, 1, 1], 64),
        Ok(())
    );
}

pub(super) fn storage_bindings_require_native_alignment_and_limited_in_bounds_ranges() {
    let validate = |offset, size, buffer_size, alignment, maximum| {
        validate_compute_binding_range(offset, size, buffer_size, alignment, maximum)
    };
    assert_eq!(
        validate(4, 256, 512, 256, 256),
        Err(ComputeCreateError::InvalidBindingRange)
    );
    assert_eq!(validate(0, 256, 256, 256, 256), Ok(()));
    assert_eq!(validate(256, 256, 512, 256, 256), Ok(()));
    assert_eq!(
        validate(0, 260, 512, 256, 256),
        Err(ComputeCreateError::InvalidBindingRange)
    );
    assert_eq!(
        validate(0, 256, 512, 256, 252),
        Err(ComputeCreateError::InvalidBindingRange)
    );
    assert_eq!(
        validate(u64::MAX - 3, 4, u64::MAX, 4, u64::MAX),
        Err(ComputeCreateError::InvalidBindingRange)
    );
}
