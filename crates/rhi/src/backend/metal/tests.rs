//! macOS Metal conformance tests.
//!
//! These compile on the Windows cross target but execute only when the crate is
//! tested on an Apple host. They deliberately use the portable API above the
//! provider: success therefore covers capability admission, opaque ownership,
//! native lowering, completion and readback publication together.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingCount,
    BindingKind, BindingResource, BindingSlot, BindingSlotId, BufferBindingAccess,
};
use crate::api::command::{ComputeScopeDescriptor, RecorderDescriptor};
use crate::api::identity::{DeviceInstanceId, Label};
use crate::api::pipeline::{ComputePipelineDescriptor, PipelineInterfaceDescriptor};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::resource::buffer::{BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::transfer::{ReadbackRequest, ReadbackViewData};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ComputeWorkgroupSize, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderRequirements, ShaderResourceRequirement, ShaderStage,
    ShaderStages,
};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};

use super::MetalProvider;

const WORDS: u64 = 8;
const SIZE: u64 = WORDS * 4;

fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the native Metal request unexpectedly deferred"),
    }
}

fn provider() -> PlatformProvider {
    let identity = DeviceInstanceId::new(0x4d45_544c);
    PlatformProvider::new(
        BackendKind::Metal,
        identity,
        Box::new(MetalProvider::new(identity)),
    )
}

fn device(provider: &PlatformProvider) -> Option<Device> {
    let adapters = ready(provider.enumerate_adapters()).expect("Metal enumeration failed")?;
    if adapters.is_empty() {
        return None;
    }
    Some(
        ready(provider.request_device(DeviceRequestDescriptor::new(
            AdapterSelection::Default,
            DeviceRequirements::new(),
        )))
        .expect("default Metal device request failed"),
    )
}

fn lane(device: &Device, domain: LaneWorkDomains) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(domain))
        .map(|lane| lane.id())
        .expect("Metal must expose its implemented ordered lane domains")
}

fn settle(device: &Device, point: crate::api::submission::CompletionPoint) -> CompletionState {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let state = device
            .completion_state(point)
            .expect("Metal completion query failed");
        match state {
            CompletionState::Pending if Instant::now() < deadline => std::thread::yield_now(),
            other => return other,
        }
    }
}

fn compute_artifact() -> ShaderArtifact {
    let source: Arc<str> = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fill(device uint *output [[buffer(0)]],
                 uint id [[thread_position_in_grid]]) {
    if (id < 8) {
        output[id] = id + 17;
    }
}
"#
    .into();
    ShaderArtifact::new(
        ShaderStage::Compute,
        "fill",
        ShaderCode::Msl(source),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new()
            .with_compute_workgroup_size(ComputeWorkgroupSize::new(8, 1, 1))
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(0),
                kind: BindingKind::StorageBuffer {
                    access: BufferBindingAccess::ReadWrite,
                    min_size: SIZE,
                },
                count: BindingCount::One,
            }),
        ShaderRequirements::new(),
        ArtifactHash([0x4d; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

#[test]
fn enumeration_resources_and_device_identity_are_real() {
    let provider = provider();
    let Some(first) = device(&provider) else {
        return;
    };
    assert_eq!(first.adapter_info().backend(), BackendKind::Metal);
    first
        .create_buffer(&BufferDescriptor::new(
            4096,
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .expect("advertised Metal buffer allocation must have native backing");
    first
        .create_sampler(&SamplerDescriptor::new())
        .expect("the baseline Metal sampler must be creatable");
    ready(first.wait_idle()).expect("fresh Metal device did not idle");

    let second = device(&provider).expect("the enumerated Metal adapter disappeared");
    assert_ne!(first.identity(), second.identity());
}

#[test]
fn compute_binding_dispatch_completion_and_readback_close_end_to_end() {
    let provider = provider();
    let Some(device) = device(&provider) else {
        return;
    };
    let shader = ready(device.create_shader(&compute_artifact()))
        .expect("inline MSL compute shader must compile");
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::COMPUTE,
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: SIZE,
            },
        )]))
        .expect("Metal storage-buffer layout must match published facts");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .expect("Metal compute interface must validate");
    let pipeline =
        ready(device.create_compute_pipeline(&ComputePipelineDescriptor::new(shader, interface)))
            .expect("Metal compute pipeline creation failed");
    let output = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::STORAGE.union(BufferUsage::COPY_SRC),
        ))
        .expect("Metal compute output buffer creation failed");
    let group = device
        .create_bind_group(
            &BindGroupDescriptor::new(layout).with_entry(BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(crate::api::resource::BufferBinding::new(
                    output.clone(),
                    BufferRange::new(0, SIZE),
                )),
            )),
        )
        .expect("Metal direct bind-group packet creation failed");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Metal recorder creation failed");
    {
        let mut compute = recorder
            .begin_compute(&ComputeScopeDescriptor::new().with_label("Metal compute evidence"))
            .expect("Metal reports compute support");
        compute.set_pipeline(&pipeline).unwrap();
        compute
            .set_bind_group(BindGroupIndex::new(0), &group, &[])
            .unwrap();
        compute.dispatch(1, 1, 1).unwrap();
        compute.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("Metal compute readback".into())),
            src: output,
            range: BufferRange::new(0, SIZE),
        })
        .expect("Metal readback route must be admitted");
    let work = recorder.finish().unwrap();
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(lane(&device, LaneWorkDomains::COMPUTE), vec![work])
        .unwrap();
    let receipt = ready(device.submit(plan.build().unwrap())).expect("Metal submit failed");
    let completion = receipt.completion_for(point).unwrap();
    assert!(matches!(
        settle(&device, completion),
        CompletionState::Complete
    ));

    let view = ticket
        .try_read()
        .expect("completed Metal readback became terminal failure")
        .expect("completed Metal readback did not publish bytes");
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("buffer readback returned texture layout")
    };
    let words = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(words, (17..25).collect::<Vec<_>>());
}
