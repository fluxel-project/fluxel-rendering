//! Vulkan pipeline-cache ownership and persistence.

use std::any::Any;
use std::sync::Arc;

use ash::vk;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::backend::PipelineCacheBackend;
use crate::api::pipeline::{
    PipelineCache, PipelineCacheDescriptor, PipelineCacheFallback, PipelineCacheValidationKey,
};
use crate::api::platform::DeviceLossInfo;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::platform::device::VulkanShared;

/// Native cache state is device-owned and can safely be shared by many
/// pipeline creations through the portable `PipelineCache` handle.
pub(crate) struct VulkanPipelineCache {
    shared: Arc<VulkanShared>,
    cache: vk::PipelineCache,
}

impl VulkanPipelineCache {
    pub(crate) fn cache(&self) -> vk::PipelineCache {
        self.cache
    }
}

impl PipelineCacheBackend for VulkanPipelineCache {
    fn serialized_data(&self) -> RhiResult<Vec<u8>> {
        unsafe { self.shared.device.get_pipeline_cache_data(self.cache) }.map_err(|result| {
            if result == vk::Result::ERROR_DEVICE_LOST {
                self.shared.mark_lost(DeviceLossInfo::new(
                    "Vulkan reported VK_ERROR_DEVICE_LOST from vkGetPipelineCacheData".to_string(),
                ));
                return RhiError::new(
                    RhiErrorKind::DeviceLost,
                    "the Vulkan device was lost while serializing a pipeline cache",
                )
                .at("VulkanPipelineCache::serialized_data");
            }
            RhiError::new(
                RhiErrorKind::BackendFailure,
                format!("vkGetPipelineCacheData failed: {result:?}"),
            )
            .at("VulkanPipelineCache::serialized_data")
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanPipelineCache {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_pipeline_cache(self.cache, None) };
    }
}

pub(crate) fn native_cache(
    cache: Option<&PipelineCache>,
) -> Result<vk::PipelineCache, VulkanFailure> {
    let Some(cache) = cache else {
        return Ok(vk::PipelineCache::null());
    };
    cache
        .native()
        .as_any()
        .downcast_ref::<VulkanPipelineCache>()
        .map(VulkanPipelineCache::cache)
        .ok_or(VulkanFailure::Unsupported {
            what: "a pipeline cache this Vulkan device did not create",
            why: "its native cache belongs to another backend",
        })
}

pub(crate) fn create_pipeline_cache(
    shared: Arc<VulkanShared>,
    descriptor: &PipelineCacheDescriptor,
) -> RhiResult<(Box<dyn PipelineCacheBackend>, PipelineCacheValidationKey)> {
    let key = shared.pipeline_cache_validation_key();
    let initial = match (
        descriptor.initial_data.as_deref(),
        descriptor.validation_key,
    ) {
        (Some(data), Some(saved_key)) if saved_key == key => data,
        (Some(_), Some(_)) if descriptor.fallback == PipelineCacheFallback::IgnoreInvalidData => {
            &[]
        }
        (Some(_), Some(_)) => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "pipeline cache bytes were created for a different Vulkan driver/device contract",
            )
            .at("VulkanDevice::create_pipeline_cache"));
        }
        (None, None) => &[],
        // Portable validation rejects a partial pair before this backend seam.
        _ => unreachable!("pipeline cache descriptor was validated by Device"),
    };
    let info = vk::PipelineCacheCreateInfo::default().initial_data(initial);
    let cache = match unsafe { shared.device.create_pipeline_cache(&info, None) } {
        Ok(cache) => cache,
        // Vulkan does not expose one portable invalid-cache result across all
        // driver versions. The explicit ignore policy retries empty after a
        // non-terminal cache-creation failure; device loss remains terminal.
        Err(result)
            if descriptor.fallback == PipelineCacheFallback::IgnoreInvalidData
                && result != vk::Result::ERROR_DEVICE_LOST =>
        {
            unsafe {
                shared
                    .device
                    .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
            }
            .map_err(|result| native(&shared, result, "vkCreatePipelineCache after ignored data"))?
        }
        Err(result) => return Err(native(&shared, result, "vkCreatePipelineCache")),
    };
    Ok((Box::new(VulkanPipelineCache { shared, cache }), key))
}

fn native(shared: &VulkanShared, result: vk::Result, operation: &'static str) -> RhiError {
    if result == vk::Result::ERROR_DEVICE_LOST {
        shared.mark_lost(DeviceLossInfo::new(format!(
            "Vulkan reported VK_ERROR_DEVICE_LOST from {operation}",
        )));
        return RhiError::new(
            RhiErrorKind::DeviceLost,
            format!("the Vulkan device was lost while executing {operation}"),
        )
        .at("VulkanDevice::create_pipeline_cache");
    }
    RhiError::new(
        RhiErrorKind::BackendFailure,
        format!("{operation} failed: {result:?}"),
    )
    .at("VulkanDevice::create_pipeline_cache")
}
