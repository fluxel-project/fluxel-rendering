//! Conservative, device-probed facts for Vulkan.
//!
//! A Vulkan format name is not by itself an RHI texture capability. The image
//! creator uses optimal tiling, an exact image type/usage/flag tuple, and a
//! concrete sample count; all of those reach the native image-format query
//! below. Consequently this module records only finite v13 keys which the
//! physical device accepted. Missing entries mean unsupported, never "the
//! driver will probably accept it".
//!
//! Alternate view-format pairs deliberately remain absent. The current image
//! path can request `MUTABLE_FORMAT`, but has not yet enabled/validated Vulkan's
//! image-format-list contract nor supplied conformance cases for each view
//! creation pair. `CapabilityFacts` therefore rejects non-empty `view_formats`,
//! rather than promising a texture whose later view can fail.

use ash::vk;

use crate::api::capability::CapabilityFacts;
use crate::api::error::RhiResult;
use crate::api::format::{TextureSupport, TextureSupportLimits, TextureSupportQuery};
use crate::api::platform::LimitKey;
use crate::api::resource::buffer::{BufferSupport, BufferSupportLimits, BufferUsage};
use crate::api::resource::route::{
    BufferCopyLayoutLimits, RouteCapabilities, RouteQuery, RouteSupport,
};
use crate::api::resource::texture::{
    Extent3d, TextureDimension, TextureUsage, TextureViewCompatibility,
};
use crate::backend::vulkan::ffi;

use crate::backend::vulkan::format::{FORMATS, vk_format};

/// Probes the resource-creation subset of Vulkan 1.0 exposed by this backend.
pub(super) fn probe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    general_ceiling: u64,
    uniform_ceiling: u64,
    storage_ceiling: u64,
) -> RhiResult<CapabilityFacts> {
    let mut facts = CapabilityFacts::empty();
    record_buffer_support(
        &mut facts,
        general_ceiling,
        uniform_ceiling,
        storage_ceiling,
    );
    facts.record_limit(LimitKey::MaxBufferSize, general_ceiling);
    facts.record_limit(LimitKey::MaxStorageBufferBindingSize, storage_ceiling);
    facts.record_route(
        RouteQuery::BufferToBuffer,
        RouteSupport::Supported(RouteCapabilities::new(
            Some(BufferCopyLayoutLimits::new(4, 4)),
            None,
        )),
    );

    for &format in FORMATS {
        let native = vk_format(format).expect("FORMATS contains only mapped Vulkan formats");
        let properties =
            unsafe { instance.get_physical_device_format_properties(physical, native) };
        for dimension in [
            TextureDimension::D1,
            TextureDimension::D2,
            TextureDimension::D3,
        ] {
            for usage in TextureUsage::all().filter(|usage| !usage.is_empty()) {
                for view_compatibility in [
                    TextureViewCompatibility::NONE,
                    TextureViewCompatibility::CUBE,
                ] {
                    let Some(limits) = texture_properties(
                        instance,
                        physical,
                        properties.optimal_tiling_features,
                        native,
                        dimension,
                        usage,
                        view_compatibility,
                    )?
                    else {
                        continue;
                    };
                    // `vkGetPhysicalDeviceImageFormatProperties` answers the
                    // native image tuple once and returns its whole sample-mask.
                    // The RHI key then selects a member from that mask; querying
                    // the driver again for each member would be identical work.
                    for sample_count in [1, 2, 4, 8, 16, 32, 64] {
                        // Keep requirements queries inside the public P0
                        // descriptor domain too: only 2D images may be
                        // multisampled, and cube-compatible images must be 1x.
                        if (dimension != TextureDimension::D2
                            || view_compatibility == TextureViewCompatibility::CUBE)
                            && sample_count != 1
                        {
                            continue;
                        }
                        if !limits
                            .sample_counts
                            .contains(vk::SampleCountFlags::from_raw(sample_count))
                        {
                            continue;
                        }
                        let query =
                            TextureSupportQuery::new(dimension, format, usage, sample_count)
                                .with_view_compatibility(view_compatibility);
                        facts.record_texture_support(
                            &query,
                            TextureSupport::Supported(TextureSupportLimits::new(
                                Extent3d {
                                    width: limits.max_extent.width,
                                    height: limits.max_extent.height,
                                    depth: limits.max_extent.depth,
                                },
                                limits.max_mip_levels,
                                limits.max_array_layers,
                            )),
                        );
                    }
                }
            }
        }
    }
    Ok(facts)
}

fn record_buffer_support(
    facts: &mut CapabilityFacts,
    general_ceiling: u64,
    uniform_ceiling: u64,
    storage_ceiling: u64,
) {
    for usage in BufferUsage::all() {
        let support = if usage.is_empty() {
            BufferSupport::Unsupported
        } else {
            let mut ceiling = general_ceiling;
            if usage.contains(BufferUsage::UNIFORM) {
                ceiling = ceiling.min(uniform_ceiling);
            }
            if usage.contains(BufferUsage::STORAGE) {
                ceiling = ceiling.min(storage_ceiling);
            }
            BufferSupport::Supported(BufferSupportLimits::new(ceiling))
        };
        facts.record_buffer_support(usage, support);
    }
}

fn texture_properties(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    features: vk::FormatFeatureFlags,
    format: vk::Format,
    dimension: TextureDimension,
    usage: TextureUsage,
    view_compatibility: TextureViewCompatibility,
) -> RhiResult<Option<vk::ImageFormatProperties>> {
    if usage.is_empty() || !format_features_cover(features, usage) {
        return Ok(None);
    }
    if view_compatibility == TextureViewCompatibility::CUBE && dimension != TextureDimension::D2 {
        return Ok(None);
    }
    let flags = if view_compatibility == TextureViewCompatibility::CUBE {
        vk::ImageCreateFlags::CUBE_COMPATIBLE
    } else {
        vk::ImageCreateFlags::empty()
    };
    let result = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            format,
            image_type(dimension),
            vk::ImageTiling::OPTIMAL,
            image_usage(usage),
            flags,
        )
    };
    let properties = match result {
        Ok(properties) => properties,
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => return Ok(None),
        Err(result) => {
            return Err(ffi::to_rhi(
                result,
                "VulkanProvider::probe_texture_image_format",
            ));
        }
    };
    // Cube intent has a descriptor-level minimum of six layers.  A native
    // format query that cannot allocate that many layers is not a useful cube
    // capability even if it accepted the creation flag.
    if view_compatibility == TextureViewCompatibility::CUBE && properties.max_array_layers < 6 {
        return Ok(None);
    }
    Ok(Some(properties))
}

fn format_features_cover(features: vk::FormatFeatureFlags, usage: TextureUsage) -> bool {
    (!usage.contains(TextureUsage::SAMPLED)
        || features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE))
        && (!usage.contains(TextureUsage::STORAGE)
            || features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE))
        && (!usage.contains(TextureUsage::COLOR_ATTACHMENT)
            || features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT))
        && (!usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
            || features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT))
}

fn image_type(dimension: TextureDimension) -> vk::ImageType {
    match dimension {
        TextureDimension::D1 => vk::ImageType::TYPE_1D,
        TextureDimension::D2 => vk::ImageType::TYPE_2D,
        TextureDimension::D3 => vk::ImageType::TYPE_3D,
    }
}

fn image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut native = vk::ImageUsageFlags::empty();
    if usage.contains(TextureUsage::COPY_SRC) {
        native |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        native |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(TextureUsage::SAMPLED) {
        native |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::STORAGE) {
        native |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        native |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        native |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    native
}
