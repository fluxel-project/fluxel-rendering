//! Vulkan fixture for the portable depth/stencil clear-store workload.

use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::submission::{LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder};
use crate::backend::conformance::cases::depth_stencil::{
    DepthStencilCaseSkip, DepthStencilClearStoreCase, assert_depth_stencil_clear_store,
    record_depth_stencil_clear_store,
};
use crate::backend::test_harness::{block_on, require_complete};
use crate::backend::vulkan::platform::VulkanProvider;

fn device() -> Option<Device> {
    let identity = DeviceInstanceId::new(0xD357);
    let native = VulkanProvider::new(identity).ok()?;
    let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
    let adapters = block_on(provider.enumerate_adapters()).ok()??;
    (!adapters.is_empty()).then_some(())?;
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
        .expect("published Vulkan raster/copy routes need one compatible lane")
}

#[test]
fn depth_stencil_fixture_runs_the_portable_clear_store_case() {
    let Some(device) = device() else {
        return;
    };
    let case = DepthStencilClearStoreCase {
        format: crate::api::format::TextureFormat::Depth24PlusStencil8,
        depth_clear: Some(0.25),
        stencil_clear: Some(0x5A),
        color: [32, 128, 223, 255],
    };
    let mut recording = match record_depth_stencil_clear_store(
        &device,
        case,
        "Vulkan combined depth/stencil clear-store",
    ) {
        Ok(recording) => recording,
        Err(
            DepthStencilCaseSkip::ColorReadbackRoute | DepthStencilCaseSkip::DepthStencilRoute(_),
        ) => return,
    };
    let mut plan = SubmissionPlanBuilder::new(&device);
    let ticket = recording.ticket.clone();
    let point = plan
        .add_batch(raster_copy_lane(&device), vec![recording.take_work()])
        .expect("Vulkan raster/copy lane accepts the portable clear-store workload");
    let receipt = block_on(device.submit(plan.build().expect("plan"))).expect("submit");
    block_on(require_complete(
        &device,
        receipt.completion_for(point).expect("completion"),
        "Vulkan combined depth/stencil attachment",
    ));
    block_on(assert_depth_stencil_clear_store(
        &ticket,
        case.color,
        "Vulkan combined depth/stencil attachment",
    ));
}
