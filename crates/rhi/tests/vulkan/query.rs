//! Vulkan fixture for the portable empty-occlusion-query conformance case.
//!
//! Native provider setup and the one compatible queue lane are Vulkan-private;
//! the actual RHI workload is owned by `tests/common`.

use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::submission::{LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder};
use crate::backend::test_harness::{block_on, require_complete};

use crate::backend::vulkan::platform::VulkanProvider;

fn device() -> Option<Device> {
    let identity = DeviceInstanceId::new(0x0C14);
    let native = VulkanProvider::new(identity).ok()?;
    let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
    let adapters = block_on(provider.enumerate_adapters()).ok()??;
    if adapters.is_empty() {
        return None;
    }
    block_on(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .ok()
}

fn raster_copy_lane(device: &Device) -> SubmissionLaneId {
    let domains = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(domains))
        .map(|lane| lane.id())
        .expect("published Vulkan raster/query/resolve routes need one compatible lane")
}

#[test]
fn empty_occlusion_query_fixture_runs_the_portable_case() {
    let Some(device) = device() else {
        // Missing loader/adapter is an unavailable environment, not a portable
        // capability verdict. An available device must execute the common case.
        return;
    };
    let mut common = crate::backend::conformance::record_empty_occlusion_query(
        &device,
        "Vulkan empty occlusion query",
    );
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(raster_copy_lane(&device), vec![common.take_work()])
        .expect("Vulkan lane accepts raster, query, resolve, and readback work");
    let receipt = block_on(device.submit(plan.build().expect("plan"))).expect("submit");
    block_on(require_complete(
        &device,
        receipt.completion_for(point).expect("completion"),
        "Vulkan empty occlusion query",
    ));
    block_on(crate::backend::conformance::assert_empty_occlusion_result(
        &common.ticket,
        "Vulkan empty occlusion query",
    ));
}
