//! Private native construction and command binding for fixed raster recipes.
//!
//! Safe resource wrappers establish device affinity, descriptor shape, and
//! lifetime leases before reaching this boundary. This module owns only the
//! corresponding HAL bind groups, views, and samplers; it neither defines the
//! public binding contract nor lets native objects escape it.

use super::*;

pub(crate) fn create_raster_uniform_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    buffer: &OwnedBuffer,
) -> Result<NativeRasterUniformBindings, String> {
    if !Arc::ptr_eq(owner, &pipeline.0.owner)
        || !Arc::ptr_eq(owner, &buffer.owner)
        || buffer.size != 80
        || !buffer.allowed_usage.contains(BufferUsageKind::Uniform)
    {
        return Err("invalid camera/material uniform binding".into());
    }
    let entries = [wgpu_hal::BindGroupEntry {
        binding: 0,
        resource_index: 0,
        count: 1,
    }];
    let size = NonZeroU64::new(80).expect("fixed uniform size");
    match (
        &owner.native,
        pipeline.0.native.as_ref(),
        buffer.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeRasterPipelineInner::Dx12 {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Dx12(buffer)),
        ) => {
            let group = unsafe {
                // SAFETY: owner/pipeline/buffer identity, exact 80-byte range,
                // Uniform usage and fixed non-dynamic layout were revalidated.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel fixed camera/material bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[],
                    textures: &[],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            }
            .map_err(|e| format!("DX12 raster uniform bind-group creation failed: {e}"))?;
            Ok(NativeRasterUniformBindings {
                native: Some(NativeRasterUniformBindingsInner::Dx12(group)),
                pipeline: pipeline.clone(),
                expected_normal_vertex_streams: None,
                expected_vertex_color_streams: None,
            })
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeRasterPipelineInner::Vulkan {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Vulkan(buffer)),
        ) => {
            let group = unsafe {
                // SAFETY: same fixed range/layout/device/lifetime proof as DX12.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel fixed camera/material bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[],
                    textures: &[],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            }
            .map_err(|e| format!("Vulkan raster uniform bind-group creation failed: {e}"))?;
            Ok(NativeRasterUniformBindings {
                native: Some(NativeRasterUniformBindingsInner::Vulkan(group)),
                pipeline: pipeline.clone(),
                expected_normal_vertex_streams: None,
                expected_vertex_color_streams: None,
            })
        }
        _ => Err("raster uniform binding does not match the camera/material pipeline".into()),
    }
}

pub(crate) fn set_raster_uniform_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeRasterUniformBindings,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner) {
        return Err("raster uniform bindings belong to another native device".into());
    }
    if encoder.active_render_view.is_none()
        || encoder.active_raster_pipeline != Some(Arc::as_ptr(&bindings.pipeline.0) as usize)
    {
        return Err("raster uniform binding requires its active raster pipeline".into());
    }
    match (
        encoder.native.as_mut().expect("live encoder"),
        bindings.pipeline.0.native.as_ref(),
        bindings.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeRasterPipelineInner::Dx12 {
                pipeline_layout, ..
            }),
            Some(NativeRasterUniformBindingsInner::Dx12(group)),
        ) => unsafe {
            // SAFETY: active pass and exact active pipeline were checked above;
            // group uses that pipeline's static group-zero layout.
            encoder.set_bind_group(pipeline_layout, 0, group, &[])
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeRasterPipelineInner::Vulkan {
                pipeline_layout, ..
            }),
            Some(NativeRasterUniformBindingsInner::Vulkan(group)),
        ) => unsafe {
            // SAFETY: same checked active-pass/static-layout proof as DX12.
            encoder.set_bind_group(pipeline_layout, 0, group, &[])
        },
        _ => return Err("raster uniform bindings belong to another native backend".into()),
    }
    Ok(())
}

/// Creates the uniform group for the normal-Lambert artifact and records the
/// two independently validated vertex stream facts for the later unsafe bind.
pub(crate) fn create_raster_normal_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    position_identity: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    normal_identity: fluxel_rendergraph::PhysicalResourceIdentity,
    normal_size: u64,
) -> Result<NativeRasterUniformBindings, String> {
    if pipeline.0.kernel != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
        || position_size == 0
        || normal_size == 0
        || position_size != normal_size
        || !position_size.is_multiple_of(12)
    {
        return Err("invalid normal-Lambert binding recipe".into());
    }
    let mut bindings = create_raster_uniform_bindings(owner, pipeline, uniform)?;
    bindings.expected_normal_vertex_streams = Some((
        position_identity,
        position_size,
        normal_identity,
        normal_size,
    ));
    Ok(bindings)
}

/// Creates the uniform group for the closed vertex-color artifact and records
/// both stream roles for the final unsafe command validation.
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native ABI carries both independent stream facts"
)]
pub(crate) fn create_raster_vertex_color_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    position_identity: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    color_identity: fluxel_rendergraph::PhysicalResourceIdentity,
    color_size: u64,
) -> Result<NativeRasterUniformBindings, String> {
    if pipeline.0.kernel != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
        || position_size == 0
        || color_size == 0
        || !position_size.is_multiple_of(12)
        || position_size / 3 != color_size
    {
        return Err("invalid vertex-color binding recipe".into());
    }
    let mut bindings = create_raster_uniform_bindings(owner, pipeline, uniform)?;
    bindings.expected_vertex_color_streams =
        Some((position_identity, position_size, color_identity, color_size));
    Ok(bindings)
}

/// Creates the fixed two-entry textured raster bind group after both safe and
/// native boundary validation established device affinity and complete ranges.
pub(crate) fn create_raster_texture_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    texture: &OwnedTexture,
) -> Result<NativeRasterTextureBindings, String> {
    if !Arc::ptr_eq(owner, &pipeline.0.owner)
        || !Arc::ptr_eq(owner, &uniform.owner)
        || !Arc::ptr_eq(owner, &texture.owner)
        || uniform.size != 80
        || !uniform.allowed_usage.contains(BufferUsageKind::Uniform)
        || texture.descriptor.dimension != TextureDimension::D2
        || texture.descriptor.format != TextureFormat::Rgba8Unorm
        || texture.descriptor.extent.depth != 1
        || texture.descriptor.mip_levels != 1
        || texture.descriptor.array_layers != 1
        || texture.descriptor.sample_count != 1
        || !texture.allowed_usage.contains(TextureUsageKind::Sampled)
    {
        return Err("invalid textured raster binding".into());
    }
    let view_desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel textured raster Rgba8 view"),
        format: wgt::TextureFormat::Rgba8Unorm,
        dimension: wgt::TextureViewDimension::D2,
        usage: wgt::TextureUses::RESOURCE,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let entries = [
        wgpu_hal::BindGroupEntry {
            binding: 0,
            resource_index: 0,
            count: 1,
        },
        wgpu_hal::BindGroupEntry {
            binding: 1,
            resource_index: 0,
            count: 1,
        },
    ];
    let size = NonZeroU64::new(80).expect("fixed uniform size");
    match (
        &owner.native,
        pipeline.0.native.as_ref(),
        uniform.native.as_ref(),
        texture.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeRasterPipelineInner::Dx12 {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Dx12(buffer)),
            Some(NativeTexture::Dx12(image)),
        ) => {
            // SAFETY: all handles originate from `owner`; the closed D2 RGBA8,
            // one-mip/one-layer descriptor and RESOURCE usage were checked above.
            // `view_desc` and `image` outlive this call, and the returned view is
            // retained with the bind group until native destruction.
            let view = unsafe { device.create_texture_view(image, &view_desc) }
                .map_err(|e| format!("DX12 textured raster view creation failed: {e}"))?;
            // SAFETY: `layout` belongs to this pipeline/device; binding 0 is the
            // validated full 80-byte Uniform buffer and binding 1 is the live
            // RESOURCE view above. Both leases are retained by the public binding
            // object, and recording/submission is serialized by the queue lock.
            let group = unsafe {
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel textured raster bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeRasterTextureBindings {
                    native: Some(NativeRasterTextureBindingsInner::Dx12 {
                        group,
                        view,
                        sampler: None,
                    }),
                    pipeline: pipeline.clone(),
                    expected_uv_vertex_streams: None,
                }),
                Err(e) => {
                    // SAFETY: creation failed before ownership escaped; `view`
                    // belongs to this device and has no dependent bind group.
                    unsafe { device.destroy_texture_view(view) };
                    Err(format!("DX12 textured raster binding creation failed: {e}"))
                }
            }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeRasterPipelineInner::Vulkan {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Vulkan(buffer)),
            Some(NativeTexture::Vulkan(image)),
        ) => {
            // SAFETY: all handles originate from `owner`; the closed D2 RGBA8,
            // one-mip/one-layer descriptor and RESOURCE usage were checked above.
            // `view_desc` and `image` outlive this call, and the returned view is
            // retained with the bind group until native destruction.
            let view = unsafe { device.create_texture_view(image, &view_desc) }
                .map_err(|e| format!("Vulkan textured raster view creation failed: {e}"))?;
            // SAFETY: `layout` belongs to this pipeline/device; binding 0 is the
            // validated full 80-byte Uniform buffer and binding 1 is the live
            // RESOURCE view above. Both leases are retained by the public binding
            // object, and recording/submission is serialized by the queue lock.
            let group = unsafe {
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel textured raster bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeRasterTextureBindings {
                    native: Some(NativeRasterTextureBindingsInner::Vulkan {
                        group,
                        view,
                        sampler: None,
                    }),
                    pipeline: pipeline.clone(),
                    expected_uv_vertex_streams: None,
                }),
                Err(e) => {
                    // SAFETY: creation failed before ownership escaped; `view`
                    // belongs to this device and has no dependent bind group.
                    unsafe { device.destroy_texture_view(view) };
                    Err(format!(
                        "Vulkan textured raster binding creation failed: {e}"
                    ))
                }
            }
        }
        _ => Err("textured raster bindings do not match textured raster pipeline".into()),
    }
}

/// Creates the explicit-UV variant of the fixed texture binding and records
/// the two opaque stream generations for a final native-boundary role check.
#[allow(
    clippy::too_many_arguments,
    reason = "the private native boundary receives each fixed binding and both independently checked stream identity/range pairs"
)]
pub(crate) fn create_raster_uv_texture_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    texture: &OwnedTexture,
    positions: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    texture_coordinates: fluxel_rendergraph::PhysicalResourceIdentity,
    texture_coordinate_size: u64,
) -> Result<NativeRasterTextureBindings, String> {
    let mut bindings = create_raster_texture_bindings(owner, pipeline, uniform, texture)?;
    bindings.expected_uv_vertex_streams = Some((
        positions,
        position_size,
        texture_coordinates,
        texture_coordinate_size,
    ));
    Ok(bindings)
}

/// Creates the closed explicit-UV filtering binding. The sampler is owned by
/// this opaque object: its bind group is destroyed first, followed by the
/// view and sampler, so no native descriptor can outlive a dependency.
#[allow(
    clippy::too_many_arguments,
    reason = "the private native boundary receives each fixed binding and both independently checked stream identity/range pairs"
)]
pub(crate) fn create_raster_uv_linear_clamp_texture_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    texture: &OwnedTexture,
    positions: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    texture_coordinates: fluxel_rendergraph::PhysicalResourceIdentity,
    texture_coordinate_size: u64,
) -> Result<NativeRasterTextureBindings, String> {
    create_raster_uv_linear_clamp_texture_bindings_for_format(
        owner,
        pipeline,
        uniform,
        texture,
        positions,
        position_size,
        texture_coordinates,
        texture_coordinate_size,
        TextureFormat::Rgba8Unorm,
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
        owner.capabilities.rgba8_unorm_filterable,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the private native boundary receives each fixed binding and both independently checked stream identity/range pairs"
)]
pub(crate) fn create_raster_uv_linear_clamp_srgb_texture_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    texture: &OwnedTexture,
    positions: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    texture_coordinates: fluxel_rendergraph::PhysicalResourceIdentity,
    texture_coordinate_size: u64,
) -> Result<NativeRasterTextureBindings, String> {
    create_raster_uv_linear_clamp_texture_bindings_for_format(
        owner,
        pipeline,
        uniform,
        texture,
        positions,
        position_size,
        texture_coordinates,
        texture_coordinate_size,
        TextureFormat::Rgba8UnormSrgb,
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
        owner.capabilities.rgba8_unorm_srgb_filterable,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "private closed native ABI facts are deliberately explicit"
)]
fn create_raster_uv_linear_clamp_texture_bindings_for_format(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeRasterPipeline,
    uniform: &OwnedBuffer,
    texture: &OwnedTexture,
    positions: fluxel_rendergraph::PhysicalResourceIdentity,
    position_size: u64,
    texture_coordinates: fluxel_rendergraph::PhysicalResourceIdentity,
    texture_coordinate_size: u64,
    expected_format: TextureFormat,
    expected_kernel: crate::RasterKernel,
    format_filterable: bool,
) -> Result<NativeRasterTextureBindings, String> {
    if !format_filterable {
        return Err(
            "linear sampling is not supported for this texture format on the selected adapter"
                .into(),
        );
    }
    if pipeline.0.kernel != expected_kernel
        || !Arc::ptr_eq(owner, &pipeline.0.owner)
        || !Arc::ptr_eq(owner, &uniform.owner)
        || !Arc::ptr_eq(owner, &texture.owner)
        || uniform.size != 80
        || !uniform.allowed_usage.contains(BufferUsageKind::Uniform)
        || texture.descriptor.dimension != TextureDimension::D2
        || texture.descriptor.format != expected_format
        || texture.descriptor.extent.depth != 1
        || texture.descriptor.mip_levels != 1
        || texture.descriptor.array_layers != 1
        || texture.descriptor.sample_count != 1
        || !texture.allowed_usage.contains(TextureUsageKind::Sampled)
    {
        return Err("invalid linear-clamp textured raster binding".into());
    }
    let view_desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel linear-clamp textured raster view"),
        format: lower_format(expected_format),
        dimension: wgt::TextureViewDimension::D2,
        usage: wgt::TextureUses::RESOURCE,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let sampler_desc = linear_clamp_sampler_descriptor();
    let entries = [
        wgpu_hal::BindGroupEntry {
            binding: 0,
            resource_index: 0,
            count: 1,
        },
        wgpu_hal::BindGroupEntry {
            binding: 1,
            resource_index: 0,
            count: 1,
        },
        wgpu_hal::BindGroupEntry {
            binding: 2,
            resource_index: 0,
            count: 1,
        },
    ];
    let size = NonZeroU64::new(80).expect("fixed uniform size");
    match (
        &owner.native,
        pipeline.0.native.as_ref(),
        uniform.native.as_ref(),
        texture.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeRasterPipelineInner::Dx12 {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Dx12(buffer)),
            Some(NativeTexture::Dx12(image)),
        ) => {
            // SAFETY: validation fixed this to a whole single-mip/layer D2
            // `expected_format` view (Rgba8UnormSrgb on the sRGB path) with
            // RESOURCE use. This device owns image/view; the opaque group and
            // its safe wrapper retain matching pipeline/resource leases until
            // terminal completion before either native object is destroyed.
            let view = unsafe { device.create_texture_view(image, &view_desc) }
                .map_err(|e| format!("DX12 linear-clamp raster view creation failed: {e}"))?;
            // SAFETY: the descriptor is fixed to filtering linear min/mag,
            // nearest mip, clamp-to-edge on U/V/W, no compare and anisotropy 1.
            let sampler = match unsafe { device.create_sampler(&sampler_desc) } {
                Ok(sampler) => sampler,
                Err(e) => {
                    unsafe { device.destroy_texture_view(view) };
                    return Err(format!("DX12 linear-clamp sampler creation failed: {e}"));
                }
            };
            // SAFETY: all same-device resources and the exact three-entry
            // layout were checked above and remain retained by the result.
            let group = unsafe {
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel linear-clamp textured raster bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[&sampler],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeRasterTextureBindings {
                    native: Some(NativeRasterTextureBindingsInner::Dx12 {
                        group,
                        view,
                        sampler: Some(sampler),
                    }),
                    pipeline: pipeline.clone(),
                    expected_uv_vertex_streams: Some((
                        positions,
                        position_size,
                        texture_coordinates,
                        texture_coordinate_size,
                    )),
                }),
                Err(e) => {
                    // SAFETY: creation failed before the group escaped; dispose
                    // dependent objects in reverse creation order.
                    unsafe {
                        device.destroy_sampler(sampler);
                        device.destroy_texture_view(view);
                    }
                    Err(format!(
                        "DX12 linear-clamp raster binding creation failed: {e}"
                    ))
                }
            }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeRasterPipelineInner::Vulkan {
                bind_group_layout: Some(layout),
                ..
            }),
            Some(NativeBuffer::Vulkan(buffer)),
            Some(NativeTexture::Vulkan(image)),
        ) => {
            // SAFETY: the same whole D2 `expected_format` descriptor
            // (Rgba8UnormSrgb on the sRGB path), RESOURCE-use, device-affinity,
            // and completion-retained lease proof as DX12 apply to this image.
            let view = unsafe { device.create_texture_view(image, &view_desc) }
                .map_err(|e| format!("Vulkan linear-clamp raster view creation failed: {e}"))?;
            // SAFETY: fixed descriptor semantics are identical on Vulkan.
            let sampler = match unsafe { device.create_sampler(&sampler_desc) } {
                Ok(sampler) => sampler,
                Err(e) => {
                    unsafe { device.destroy_texture_view(view) };
                    return Err(format!("Vulkan linear-clamp sampler creation failed: {e}"));
                }
            };
            // SAFETY: checked whole uniform/view/sampler inputs match the
            // closed layout and are retained with the opaque result.
            let group = unsafe {
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel linear-clamp textured raster bindings"),
                    layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, 0, size)],
                    samplers: &[&sampler],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeRasterTextureBindings {
                    native: Some(NativeRasterTextureBindingsInner::Vulkan {
                        group,
                        view,
                        sampler: Some(sampler),
                    }),
                    pipeline: pipeline.clone(),
                    expected_uv_vertex_streams: Some((
                        positions,
                        position_size,
                        texture_coordinates,
                        texture_coordinate_size,
                    )),
                }),
                Err(e) => {
                    // SAFETY: no group escaped, so the view/sampler are uniquely
                    // consumable by this same device.
                    unsafe {
                        device.destroy_sampler(sampler);
                        device.destroy_texture_view(view);
                    }
                    Err(format!(
                        "Vulkan linear-clamp raster binding creation failed: {e}"
                    ))
                }
            }
        }
        _ => Err("linear-clamp bindings do not match the linear-clamp pipeline".into()),
    }
}

pub(crate) fn linear_clamp_sampler_descriptor() -> wgpu_hal::SamplerDescriptor<'static> {
    wgpu_hal::SamplerDescriptor {
        label: Some("fluxel fixed linear-clamp sampler"),
        address_modes: [wgt::AddressMode::ClampToEdge; 3],
        mag_filter: wgt::FilterMode::Linear,
        min_filter: wgt::FilterMode::Linear,
        mipmap_filter: wgt::MipmapFilterMode::Nearest,
        // The WGSL has an explicit `textureSampleLevel(..., 0.0)`. Keep the
        // native sampler unconstrained; shader semantics, not clamp repair,
        // choose mip zero.
        lod_clamp: 0.0..32.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    }
}

pub(crate) fn set_raster_texture_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeRasterTextureBindings,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner)
        || encoder.active_render_view.is_none()
        || encoder.active_raster_pipeline != Some(Arc::as_ptr(&bindings.pipeline.0) as usize)
    {
        return Err("textured raster binding requires its active raster pipeline".into());
    }
    match (
        encoder.native.as_mut().expect("live encoder"),
        bindings.pipeline.0.native.as_ref(),
        bindings.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeRasterPipelineInner::Dx12 {
                pipeline_layout, ..
            }),
            Some(NativeRasterTextureBindingsInner::Dx12 { group, .. }),
        ) => {
            // SAFETY: device affinity and the currently active matching pipeline
            // were checked above; group layout slot zero is the closed two-entry
            // recipe and all resources are retained through command completion.
            unsafe { encoder.set_bind_group(pipeline_layout, 0, group, &[]) }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeRasterPipelineInner::Vulkan {
                pipeline_layout, ..
            }),
            Some(NativeRasterTextureBindingsInner::Vulkan { group, .. }),
        ) => {
            // SAFETY: device affinity and the currently active matching pipeline
            // were checked above; group layout slot zero is the closed two-entry
            // recipe and all resources are retained through command completion.
            unsafe { encoder.set_bind_group(pipeline_layout, 0, group, &[]) }
        }
        _ => return Err("textured raster binding backend mismatch".into()),
    }
    Ok(())
}

/// The explicit-UV artifact uses the same closed uniform-plus-texture native
/// group, while its distinct safe wrapper carries the vertex-role identities.
pub(crate) fn set_raster_uv_texture_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeRasterTextureBindings,
) -> Result<(), String> {
    set_raster_texture_bindings(encoder, bindings)
}

/// The linear-clamp UV artifact has the same checked command-side binding
/// lifetime as the integer UV artifact; its differing sampler ABI is fully
/// captured by the opaque group created above.
pub(crate) fn set_raster_uv_linear_clamp_texture_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeRasterTextureBindings,
) -> Result<(), String> {
    if bindings.pipeline.0.kernel
        != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
    {
        return Err("linear-clamp bindings require their linear-clamp raster pipeline".into());
    }
    set_raster_texture_bindings(encoder, bindings)
}

pub(crate) fn set_raster_uv_linear_clamp_srgb_texture_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeRasterTextureBindings,
) -> Result<(), String> {
    if bindings.pipeline.0.kernel
        != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
    {
        return Err("sRGB linear-clamp bindings require their sRGB raster pipeline".into());
    }
    set_raster_texture_bindings(encoder, bindings)
}
