//! Vulkan query-pool ownership.
//!
//! Query pools are core Vulkan objects.  The command spine performs all query
//! commands; this object only owns the pool and keeps the device alive until
//! every recorded reference has retired.

use std::any::Any;
use std::sync::Arc;

use ash::vk;

use crate::api::query::{PipelineStatistics, QuerySetDescriptor, QueryType};
use crate::api::resource::backend::QuerySetBackend;
use crate::backend::vulkan::failure::VulkanFailure;
use crate::backend::vulkan::platform::device::VulkanShared;

pub(in crate::backend::vulkan) struct VulkanQuerySet {
    shared: Arc<VulkanShared>,
    pool: vk::QueryPool,
    _ty: QueryType,
}

impl QuerySetBackend for VulkanQuerySet {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl VulkanQuerySet {
    pub(in crate::backend::vulkan) fn pool(&self) -> vk::QueryPool {
        self.pool
    }
}

impl Drop for VulkanQuerySet {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_query_pool(self.pool, None) };
    }
}

pub(in crate::backend::vulkan) fn create_query_set(
    shared: Arc<VulkanShared>,
    descriptor: &QuerySetDescriptor,
) -> Result<VulkanQuerySet, VulkanFailure> {
    let (query_type, statistics) = match descriptor.ty {
        QueryType::Occlusion => (
            vk::QueryType::OCCLUSION,
            vk::QueryPipelineStatisticFlags::empty(),
        ),
        QueryType::Timestamp => (
            vk::QueryType::TIMESTAMP,
            vk::QueryPipelineStatisticFlags::empty(),
        ),
        QueryType::PipelineStatistics(selection) => {
            (vk::QueryType::PIPELINE_STATISTICS, statistics(selection))
        }
    };
    let info = vk::QueryPoolCreateInfo::default()
        .query_type(query_type)
        .query_count(descriptor.count)
        .pipeline_statistics(statistics);
    let pool = unsafe { shared.device.create_query_pool(&info, None) }.map_err(|result| {
        VulkanFailure::Native(crate::backend::vulkan::ffi::NativeError::new(
            result,
            "vkCreateQueryPool",
        ))
    })?;
    Ok(VulkanQuerySet {
        shared,
        pool,
        _ty: descriptor.ty,
    })
}

fn statistics(selection: PipelineStatistics) -> vk::QueryPipelineStatisticFlags {
    let mut flags = vk::QueryPipelineStatisticFlags::empty();
    if selection.contains(PipelineStatistics::VERTEX_SHADER_INVOCATIONS) {
        flags |= vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS;
    }
    if selection.contains(PipelineStatistics::CLIPPER_INVOCATIONS) {
        flags |= vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS;
    }
    if selection.contains(PipelineStatistics::CLIPPER_PRIMITIVES_OUT) {
        flags |= vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES;
    }
    if selection.contains(PipelineStatistics::FRAGMENT_SHADER_INVOCATIONS) {
        flags |= vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS;
    }
    if selection.contains(PipelineStatistics::COMPUTE_SHADER_INVOCATIONS) {
        flags |= vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS;
    }
    flags
}
