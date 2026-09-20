//! Native Vulkan platform smoke tests.
//!
//! The smoke path covers platform ownership plus the first capability-closed
//! resource slice: Dedicated allocation, core sampler creation, and transfer.
//! Compute has its own native conformance module; these tests are not evidence
//! for raster or presentation support.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::command::{BufferCopy, RecorderDescriptor, TextureCopy};
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
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{
    BufferUploadDescriptor, ReadbackRequest, ReadbackViewData, TextureUploadDescriptor,
};
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
    let plan = SubmissionPlanBuilder::new(&device)
        .build()
        .expect("the portable contract permits an empty plan");
    let receipt = ready(device.submit(plan)).expect("an empty Vulkan plan is an accepted no-op");
    assert!(matches!(
        device.completion_state(receipt.completion()).unwrap(),
        CompletionState::Complete
    ));
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

#[test]
fn texture_upload_copy_and_readback_move_texels_on_a_real_vulkan_queue() {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 4;
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
    let usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let source = device
        .create_texture(&TextureDescriptor::new_2d(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            usage,
        ))
        .unwrap();
    let destination = device
        .create_texture(&TextureDescriptor::new_2d(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            usage,
        ))
        .unwrap();
    let bytes = (0..WIDTH * HEIGHT * 4)
        .map(|i| ((i * 29 + 7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let layers = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let origin = Origin3d { x: 0, y: 0, z: 0 };
    let extent = crate::api::resource::texture::Extent3d::d2(WIDTH, HEIGHT);
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some("Vulkan texture upload".into())),
            dst: source.clone(),
            subresource: layers,
            origin,
            extent,
            source_layout: HostTexelLayout {
                bytes_per_row: WIDTH * 4,
                rows_per_image: HEIGHT,
            },
            bytes: Arc::from(bytes.as_slice()),
        })
        .unwrap();
    let mut upload_recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    upload_recorder.encode_upload(&upload).unwrap();
    let upload_work = upload_recorder.finish().unwrap();
    let mut upload_plan = SubmissionPlanBuilder::new(&device);
    let upload_point = upload_plan
        .add_batch(copy_lane(&device), vec![upload_work])
        .unwrap();
    let upload_receipt = ready(device.submit(upload_plan.build().unwrap())).unwrap();
    let upload_completion = upload_receipt.completion_for(upload_point).unwrap();
    let upload_deadline = Instant::now() + Duration::from_secs(10);
    while matches!(
        device.completion_state(upload_completion).unwrap(),
        CompletionState::Pending
    ) && Instant::now() < upload_deadline
    {
        std::thread::yield_now();
    }
    assert!(matches!(
        device.completion_state(upload_completion).unwrap(),
        CompletionState::Complete
    ));

    // A second submit must start from the first submit's accepted image layout,
    // not UNDEFINED: using UNDEFINED here would legally discard the uploaded
    // texels and make this cross-submit comparison fail nondeterministically.
    let mut recorder = device.create_recorder(&RecorderDescriptor::new()).unwrap();
    recorder
        .copy_texture(&TextureCopy {
            src: source,
            src_subresource: layers,
            src_origin: origin,
            dst: destination.clone(),
            dst_subresource: layers,
            dst_origin: origin,
            extent,
        })
        .unwrap();
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some("Vulkan texture readback".into())),
            src: destination,
            subresource: layers,
            origin,
            extent,
        })
        .unwrap();
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan.add_batch(copy_lane(&device), vec![work]).unwrap();
    let receipt = ready(device.submit(plan.build().unwrap())).unwrap();
    let completion = receipt.completion_for(point).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while matches!(
        device.completion_state(completion).unwrap(),
        CompletionState::Pending
    ) && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    assert!(matches!(
        device.completion_state(completion).unwrap(),
        CompletionState::Complete
    ));
    let view = ticket
        .try_read()
        .unwrap()
        .expect("texture readback was not published");
    let ReadbackViewData::Texture { bytes: got, layout } = view.data() else {
        panic!("texture readback returned buffer data")
    };
    assert_eq!(layout.bytes_per_row, WIDTH * 4);
    assert_eq!(got, bytes.as_slice());
}
