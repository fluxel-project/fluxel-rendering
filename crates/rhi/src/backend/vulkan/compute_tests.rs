//! Vulkan compute conformance evidence.
//!
//! This is deliberately a native-driver test, not a mock test.  It closes the
//! presently advertised buffer-only compute route end to end: immutable
//! descriptor packet creation, pipeline-layout compatibility, SPIR-V module and
//! pipeline creation, dispatch, completion, and readback publication.  The
//! fixture is inline SPIR-V produced once with `glslangValidator`; the test never
//! invokes a shader compiler, so its result stays evidence about the Vulkan
//! lowering rather than about a host toolchain.

use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingCount,
    BindingKind, BindingResource, BindingSlot, BindingSlotId, BufferBindingAccess,
};
use crate::api::command::{ComputeScopeDescriptor, RecorderDescriptor};
use crate::api::error::RhiErrorKind;
use crate::api::identity::{DeviceInstanceId, Label};
use crate::api::pipeline::{ComputePipelineDescriptor, PipelineInterfaceDescriptor};
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, Device, PlatformProvider};
use crate::api::resource::buffer::{BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackViewData};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ComputeWorkgroupSize, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderRequirements, ShaderResourceRequirement, ShaderStage,
    ShaderStages,
};
use crate::api::submission::{
    CompletionState, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder,
};

use super::platform::VulkanProvider;

const WORDS: u64 = 8;
const SIZE: u64 = WORDS * 4;

fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the native Vulkan request unexpectedly deferred"),
    }
}

fn provider() -> Option<PlatformProvider> {
    let identity = DeviceInstanceId::new(0x0C13);
    match VulkanProvider::new(identity) {
        Ok(native) => Some(PlatformProvider::new(
            BackendKind::Vulkan,
            identity,
            Box::new(native),
        )),
        Err(error) => {
            assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
            None
        }
    }
}

fn compute_lane(device: &Device) -> SubmissionLaneId {
    device
        .capabilities()
        .submission()
        .lanes()
        .iter()
        .find(|lane| lane.domains().contains(LaneWorkDomains::COMPUTE))
        .map(|lane| lane.id())
        .expect("the advertised Vulkan compute path needs a COMPUTE-capable lane")
}

fn compute_artifact() -> ShaderArtifact {
    // GLSL source used once to produce these Vulkan 1.0 words:
    //
    // #version 450
    // layout(local_size_x = 8) in;
    // layout(set = 0, binding = 0, std430) buffer Output { uint values[]; } output_data;
    // void main() { output_data.values[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x + 17; }
    //
    // The resource declaration is deliberately duplicated in the portable
    // ShaderInterface.  SPIR-V reflection is not a v13 API responsibility; the
    // supplied interface is the contract the binding and pipeline validators use.
    let words: Arc<[u32]> = vec![
        0x0723_0203,
        0x0001_0000,
        0x0008_000B,
        0x0000_001D,
        0x0000_0000,
        0x0002_0011,
        0x0000_0001,
        0x0006_000B,
        0x0000_0001,
        0x4C53_4C47,
        0x6473_742E,
        0x3035_342E,
        0x0000_0000,
        0x0003_000E,
        0x0000_0000,
        0x0000_0001,
        0x0006_000F,
        0x0000_0005,
        0x0000_0004,
        0x6E69_616D,
        0x0000_0000,
        0x0000_000F,
        0x0006_0010,
        0x0000_0004,
        0x0000_0011,
        0x0000_0008,
        0x0000_0001,
        0x0000_0001,
        0x0003_0003,
        0x0000_0002,
        0x0000_01C2,
        0x0004_0005,
        0x0000_0004,
        0x6E69_616D,
        0x0000_0000,
        0x0004_0005,
        0x0000_0008,
        0x7074_754F,
        0x0000_7475,
        0x0005_0006,
        0x0000_0008,
        0x0000_0000,
        0x756C_6176,
        0x0000_7365,
        0x0005_0005,
        0x0000_000A,
        0x7074_756F,
        0x645F_7475,
        0x0061_7461,
        0x0008_0005,
        0x0000_000F,
        0x475F_6C67,
        0x6162_6F6C,
        0x766E_496C,
        0x7461_636F,
        0x496E_6F69,
        0x0000_0044,
        0x0004_0047,
        0x0000_0007,
        0x0000_0006,
        0x0000_0004,
        0x0003_0047,
        0x0000_0008,
        0x0000_0003,
        0x0005_0048,
        0x0000_0008,
        0x0000_0000,
        0x0000_0023,
        0x0000_0000,
        0x0004_0047,
        0x0000_000A,
        0x0000_0021,
        0x0000_0000,
        0x0004_0047,
        0x0000_000A,
        0x0000_0022,
        0x0000_0000,
        0x0004_0047,
        0x0000_000F,
        0x0000_000B,
        0x0000_001C,
        0x0004_0047,
        0x0000_001C,
        0x0000_000B,
        0x0000_0019,
        0x0002_0013,
        0x0000_0002,
        0x0003_0021,
        0x0000_0003,
        0x0000_0002,
        0x0004_0015,
        0x0000_0006,
        0x0000_0020,
        0x0000_0000,
        0x0003_001D,
        0x0000_0007,
        0x0000_0006,
        0x0003_001E,
        0x0000_0008,
        0x0000_0007,
        0x0004_0020,
        0x0000_0009,
        0x0000_0002,
        0x0000_0008,
        0x0004_003B,
        0x0000_0009,
        0x0000_000A,
        0x0000_0002,
        0x0004_0015,
        0x0000_000B,
        0x0000_0020,
        0x0000_0001,
        0x0004_002B,
        0x0000_000B,
        0x0000_000C,
        0x0000_0000,
        0x0004_0017,
        0x0000_000D,
        0x0000_0006,
        0x0000_0003,
        0x0004_0020,
        0x0000_000E,
        0x0000_0001,
        0x0000_000D,
        0x0004_003B,
        0x0000_000E,
        0x0000_000F,
        0x0000_0001,
        0x0004_002B,
        0x0000_0006,
        0x0000_0010,
        0x0000_0000,
        0x0004_0020,
        0x0000_0011,
        0x0000_0001,
        0x0000_0006,
        0x0004_002B,
        0x0000_0006,
        0x0000_0016,
        0x0000_0011,
        0x0004_0020,
        0x0000_0018,
        0x0000_0002,
        0x0000_0006,
        0x0004_002B,
        0x0000_0006,
        0x0000_001A,
        0x0000_0008,
        0x0004_002B,
        0x0000_0006,
        0x0000_001B,
        0x0000_0001,
        0x0006_002C,
        0x0000_000D,
        0x0000_001C,
        0x0000_001A,
        0x0000_001B,
        0x0000_001B,
        0x0005_0036,
        0x0000_0002,
        0x0000_0004,
        0x0000_0000,
        0x0000_0003,
        0x0002_00F8,
        0x0000_0005,
        0x0005_0041,
        0x0000_0011,
        0x0000_0012,
        0x0000_000F,
        0x0000_0010,
        0x0004_003D,
        0x0000_0006,
        0x0000_0013,
        0x0000_0012,
        0x0005_0041,
        0x0000_0011,
        0x0000_0014,
        0x0000_000F,
        0x0000_0010,
        0x0004_003D,
        0x0000_0006,
        0x0000_0015,
        0x0000_0014,
        0x0005_0080,
        0x0000_0006,
        0x0000_0017,
        0x0000_0015,
        0x0000_0016,
        0x0006_0041,
        0x0000_0018,
        0x0000_0019,
        0x0000_000A,
        0x0000_000C,
        0x0000_0013,
        0x0003_003E,
        0x0000_0019,
        0x0000_0017,
        0x0001_00FD,
        0x0001_0038,
    ]
    .into();
    ShaderArtifact::new(
        ShaderStage::Compute,
        "main",
        ShaderCode::SpirV(words),
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
        ArtifactHash([0xC0; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

#[test]
fn buffer_compute_dispatch_keeps_dropped_caller_handles_alive_until_readback() {
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

    let shader = ready(device.create_shader(&compute_artifact()))
        .expect("the inline SPIR-V module must create on this Vulkan device");
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::COMPUTE,
            BindingKind::StorageBuffer {
                access: BufferBindingAccess::ReadWrite,
                min_size: SIZE,
            },
        )]))
        .expect("the storage-buffer bind-group layout must validate");
    let interface = device
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))
        .expect("the bind-group layout must make a compute pipeline interface");
    let pipeline = ready(
        device.create_compute_pipeline(&ComputePipelineDescriptor::new(
            shader.clone(),
            interface.clone(),
        )),
    )
    .expect("the inline SPIR-V must create a Vulkan compute pipeline");
    let output = device
        .create_buffer(&BufferDescriptor::new(
            SIZE,
            BufferUsage::STORAGE.union(BufferUsage::COPY_SRC),
        ))
        .expect("the storage output must be creatable");
    let group = device
        .create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entry(
            BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(crate::api::resource::BufferBinding::new(
                    output.clone(),
                    BufferRange::new(0, SIZE),
                )),
            ),
        ))
        .expect("the Vulkan storage-buffer descriptor set must be created");

    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .expect("Vulkan recorder creation failed");
    {
        let mut compute = recorder
            .begin_compute(&ComputeScopeDescriptor::new())
            .expect("Vulkan facts advertise compute");
        compute.set_pipeline(&pipeline).unwrap();
        compute
            .set_bind_group(BindGroupIndex::new(0), &group, &[])
            .unwrap();
        compute.dispatch(1, 1, 1).unwrap();
        compute.end().unwrap();
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("Vulkan compute output".into())),
            src: output.clone(),
            range: BufferRange::new(0, SIZE),
        })
        .expect("the compute output carries COPY_SRC");
    let work = recorder.finish().unwrap();

    // `RecordedWork` and accepted-batch retention, not the caller's variables,
    // must keep every native shader/pipeline/descriptor/resource alive.  Dropping
    // the complete caller-side graph before submit makes an accidental reliance on
    // one of those handles fail deterministically under validation layers.
    drop(group);
    drop(pipeline);
    drop(interface);
    drop(layout);
    drop(shader);
    drop(output);

    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(compute_lane(&device), vec![work])
        .expect("the graphics queue's compute lane accepts compute plus readback");
    let receipt = ready(device.submit(plan.build().unwrap()))
        .expect("the complete dispatch must lower before native work is accepted");
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
        .expect("completed Vulkan compute readback must be terminally readable")
        .expect("completion must publish the compute bytes");
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("compute buffer readback returned texture data");
    };
    let values = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, (17..17 + WORDS as u32).collect::<Vec<_>>());
}
