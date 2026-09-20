//! Native Vulkan platform smoke tests.
//!
//! The smoke path covers platform ownership plus the first capability-closed
//! resource slice: Dedicated resource allocation and core sampler creation. It
//! is not evidence for binding, pipeline, raster, compute, or presentation support.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::command::{BufferCopy, RecorderDescriptor};
use crate::api::error::RhiErrorKind;
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceInstanceId, Label};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, PlatformProvider};
use crate::api::resource::buffer::{
    BufferDescriptor, BufferRange, BufferSupportQuery, BufferUsage,
};
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::texture::{TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{BufferUploadDescriptor, ReadbackRequest, ReadbackViewData};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};

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
    let usage = BufferUsage::COPY_SRC
        .union(BufferUsage::COPY_DST)
        .union(BufferUsage::VERTEX);
    assert!(
        first
            .capabilities()
            .buffer_support(&BufferSupportQuery::new(usage))
            .is_supported(),
        "the Vulkan core buffer usages implemented by the Dedicated path must be advertised"
    );
    let buffer = first
        .create_buffer(&BufferDescriptor::new(4096, usage))
        .expect("advertised Vulkan buffer creation must have native backing");
    assert_eq!(buffer.descriptor().size, 4096);
    first
        .create_sampler(&SamplerDescriptor::new())
        .expect("the core non-anisotropic Vulkan sampler path must be reachable");
    let texture = first
        .create_texture(&TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
        ))
        .expect("an advertised Vulkan RGBA8 transfer texture must have native backing");
    let view = TextureViewDescriptor::whole(&texture, TextureViewDimension::D2).unwrap();
    first
        .create_texture_view(&texture, &view)
        .expect("a same-format full Vulkan image view must be creatable");
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
fn upload_copy_and_readback_move_bytes_on_a_real_vulkan_queue() {
    const SIZE: u64 = 4096;
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
    let usage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);
    let source = device
        .create_buffer(&BufferDescriptor::new(SIZE, usage))
        .expect("Vulkan source buffer creation failed");
    let destination = device
        .create_buffer(&BufferDescriptor::new(SIZE, usage))
        .expect("Vulkan destination buffer creation failed");
    let pattern = (0..SIZE)
        .map(|index| (index.wrapping_mul(73).wrapping_add(19) & 0xff) as u8)
        .collect::<Vec<_>>();
    let upload = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Label(Some("Vulkan end-to-end upload".into())),
            dst: source.clone(),
            dst_offset: 0,
            bytes: Arc::from(pattern.as_slice()),
        })
        .expect("the advertised Vulkan buffer route must accept an aligned upload");
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Vulkan recorder creation failed");
    recorder.encode_upload(&upload).unwrap();
    recorder
        .copy_buffer(&BufferCopy {
            src: source,
            src_offset: 0,
            dst: destination.clone(),
            dst_offset: 0,
            size: SIZE,
        })
        .unwrap();
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("Vulkan end-to-end readback".into())),
            src: destination,
            range: BufferRange::new(0, SIZE),
        })
        .unwrap();
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(copy_lane(&device), vec![work]).unwrap();
    let receipt = ready(device.submit(plan.build().unwrap())).unwrap();
    let completion = receipt.completion_for(point).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match device.completion_state(completion).unwrap() {
            CompletionState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            CompletionState::Complete => break,
            other => panic!("Vulkan copy did not complete successfully: {other:?}"),
        }
    }
    let view = ticket
        .try_read()
        .expect("completed Vulkan readback must be terminally readable")
        .expect("completed Vulkan readback must publish bytes");
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("buffer readback returned texture data")
    };
    assert_eq!(bytes, pattern.as_slice());
    println!(
        "Vulkan byte-movement evidence: adapter={:?} bytes={} first8={:?}",
        device.adapter_info().name(),
        bytes.len(),
        &bytes[..8]
    );
}
