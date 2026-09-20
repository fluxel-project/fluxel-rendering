//! Vulkan pipeline state objects.
//!
//! Pipeline layouts deliberately belong to pipelines, rather than to the
//! portable `PipelineInterface`: the latter is a logical compatibility contract
//! and has no native lifetime.  A Vulkan pipeline owns the descriptor-set
//! layouts and `VkPipelineLayout` required to consume that contract.

mod cache;
mod compute;
mod raster;

pub(in crate::backend::vulkan) use cache::create_pipeline_cache;
pub(in crate::backend::vulkan) use compute::{VulkanComputePipeline, create_compute_pipeline};
pub(in crate::backend::vulkan) use raster::{VulkanRasterPipeline, create_raster_pipeline};
