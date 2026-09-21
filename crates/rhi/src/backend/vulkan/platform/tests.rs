//! Native Vulkan platform smoke tests.
//!
//! The smoke path covers platform ownership plus the first capability-closed
//! resource slice: Dedicated allocation, core sampler creation, and transfer.
//! Compute has its own native conformance module; these tests are not evidence
//! for raster or presentation support.

use crate::api::error::RhiErrorKind;
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, PlatformProvider};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};
use crate::backend::test_harness::{block_on, require_complete};
use core::future::Future;
use core::task::{Context, Poll, Waker};

use super::provider::VulkanProvider;

fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the native Vulkan request unexpectedly deferred"),
    }
}

fn provider() -> Option<PlatformProvider> {
    let identity = DeviceInstanceId::new(0x0a13);
    match VulkanProvider::new(identity) {
        Ok(native) => Some(PlatformProvider::new(
            BackendKind::Vulkan,
            identity,
            Box::new(native),
        )),
        Err(error) => {
            // Dynamic loading is part of the platform contract: on a host with
            // no Vulkan loader this is a structured setup failure, not a panic
            // and not a test failure for an API unavailable on that host.
            assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
            None
        }
    }
}

#[test]
fn loader_enumeration_request_identity_and_idle() {
    let Some(provider) = provider() else {
        return;
    };
    let adapters = ready(provider.enumerate_adapters()).expect("Vulkan enumeration failed");
    let Some(adapters) = adapters else {
        panic!("native Vulkan provider must expose physical-device enumeration");
    };
    if adapters.is_empty() {
        // A working loader with no accessible physical device is a legitimate
        // CI/container state. There is nothing to request or wait on.
        return;
    }
    assert!(
        adapters
            .iter()
            .all(|adapter| adapter.backend() == BackendKind::Vulkan)
    );

    let descriptor =
        DeviceRequestDescriptor::new(AdapterSelection::Default, DeviceRequirements::new());
    let first = ready(provider.request_device(descriptor.clone()))
        .expect("default Vulkan device request failed");
    crate::backend::conformance::cases::core_device::logical_creation(
        &first,
        "Vulkan core logical creation",
    )
    .require_pass("Vulkan core logical creation");
    ready(first.wait_idle()).expect("fresh Vulkan device did not idle");
    let second =
        ready(provider.request_device(descriptor)).expect("second Vulkan device request failed");
    assert_ne!(first.identity(), second.identity());
    ready(second.wait_idle()).expect("second Vulkan device did not idle");
}

fn copy_lane(device: &crate::api::platform::Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::COPY))
        .map(|lane| lane.id())
        .expect("the base guarantee requires one COPY-capable lane")
}

#[test]
fn an_empty_plan_is_a_completed_noop_without_native_submission() {
    let Some(provider) = provider() else {
        return;
    };
    if ready(provider.enumerate_adapters())
        .expect("Vulkan enumeration failed")
        .as_ref()
        .is_none_or(Vec::is_empty)
    {
        return;
    }
    let device = ready(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .expect("default Vulkan device request failed");
    let plan =
        crate::backend::conformance::cases::core_device::empty_plan(&device, "Vulkan empty plan");
    let receipt = ready(device.submit(plan)).expect("an empty Vulkan plan is an accepted no-op");
    assert!(matches!(
        device.completion_state(receipt.completion()).unwrap(),
        CompletionState::Complete
    ));
}

#[test]
fn upload_copy_and_readback_move_bytes_on_a_real_vulkan_queue() {
    let Some(provider) = provider() else {
        return;
    };
    let adapters = ready(provider.enumerate_adapters()).expect("Vulkan enumeration failed");
    if adapters.as_ref().is_none_or(Vec::is_empty) {
        return;
    }
    let device = ready(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .expect("default Vulkan device request failed");
    let mut recording =
        crate::backend::conformance::cases::transfer::record_buffer_upload_copy_readback(
            &device,
            "Vulkan buffer upload/copy/readback",
        );
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(copy_lane(&device), vec![recording.take_work()])
        .unwrap();
    let receipt = ready(device.submit(plan.build().unwrap())).unwrap();
    let completion = receipt.completion_for(point).unwrap();

    block_on(require_complete(
        &device,
        completion,
        "Vulkan buffer upload/copy/readback",
    ));
    block_on(
        crate::backend::conformance::cases::transfer::assert_buffer_transfer(
            &recording,
            "Vulkan buffer upload/copy/readback",
        ),
    );
    println!(
        "Vulkan byte-movement evidence: adapter={:?} bytes={}",
        device.adapter_info().name(),
        crate::backend::conformance::cases::transfer::BUFFER_TRANSFER_SIZE,
    );
}

#[test]
fn texture_upload_copy_and_readback_move_texels_on_a_real_vulkan_queue() {
    let Some(provider) = provider() else {
        return;
    };
    if ready(provider.enumerate_adapters())
        .expect("Vulkan enumeration failed")
        .as_ref()
        .is_none_or(Vec::is_empty)
    {
        return;
    }
    let device = ready(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .unwrap();
    let mut recording =
        crate::backend::conformance::cases::transfer::record_texture_upload_copy_readback(
            &device,
            "Vulkan texture upload/copy/readback",
        );
    let mut upload_plan = SubmissionPlanBuilder::new(&device);
    let upload_point = upload_plan
        .add_batch(copy_lane(&device), vec![recording.take_upload_work()])
        .unwrap();
    let upload_receipt = ready(device.submit(upload_plan.build().unwrap())).unwrap();
    let upload_completion = upload_receipt.completion_for(upload_point).unwrap();
    block_on(require_complete(
        &device,
        upload_completion,
        "Vulkan texture upload",
    ));

    // The common recording's second submit must start from the first accepted
    // submit's image layout, not UNDEFINED.  It therefore verifies the
    // backend's persistent image-state tracking across plan boundaries.
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(copy_lane(&device), vec![recording.take_work()])
        .unwrap();
    let receipt = ready(device.submit(plan.build().unwrap())).unwrap();
    let completion = receipt.completion_for(point).unwrap();
    block_on(require_complete(
        &device,
        completion,
        "Vulkan texture copy/readback",
    ));
    block_on(
        crate::backend::conformance::cases::transfer::assert_texture_transfer(
            &recording,
            "Vulkan texture upload/copy/readback",
        ),
    );
}
