use core::future::Future;
use core::task::{Context, Poll, Waker};
use std::sync::Arc;

use crate::api::error::RhiErrorKind;
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::provider::AdapterSelection;
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::platform::requirements::DeviceRequirements;
use crate::api::platform::{BackendKind, PlatformProvider};
use crate::api::shader::{
    ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact, ShaderCode,
    ShaderInterface, ShaderRequirements, ShaderStage,
};
use crate::backend::vulkan::platform::VulkanProvider;

fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the native Vulkan request unexpectedly deferred"),
    }
}

#[test]
fn a_spirv_artifact_creates_a_real_vulkan_shader_module() {
    let identity = DeviceInstanceId::new(0x5a13);
    let native = match VulkanProvider::new(identity) {
        Ok(native) => native,
        Err(error) => {
            assert_eq!(error.kind(), RhiErrorKind::BackendFailure);
            return;
        }
    };
    let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
    let adapters = ready(provider.enumerate_adapters()).expect("Vulkan enumeration failed");
    if adapters.as_ref().is_none_or(Vec::is_empty) {
        return;
    }
    let device = ready(provider.request_device(DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new(),
    )))
    .expect("Vulkan device request failed");

    // Minimal Vulkan 1.0 vertex module: Shader capability, Logical/GLSL450
    // memory model, one `main` entry point, and an empty
    // function body. Keeping the words inline makes this a driver-facing test,
    // not a test of a build-time shader compiler.
    let words: Arc<[u32]> = vec![
        0x0723_0203,
        0x0001_0000,
        0,
        6,
        0,
        0x0002_0011,
        1,
        0x0003_000e,
        0,
        1,
        0x0005_000f,
        0,
        4,
        0x6e69_616d,
        0,
        0x0002_0013,
        1,
        0x0003_0021,
        2,
        1,
        0x0005_0036,
        1,
        4,
        0,
        2,
        0x0002_00f8,
        5,
        0x0001_00fd,
        0x0001_0038,
    ]
    .into();
    let artifact = ShaderArtifact::new(
        ShaderStage::Vertex,
        "main",
        ShaderCode::SpirV(words),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new().with_writes_position(true),
        ShaderRequirements::new(),
        ArtifactHash([0x5a; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 15,
        },
    );
    let module = ready(device.create_shader(&artifact)).expect("VkShaderModule creation failed");
    assert_eq!(module.stage(), ShaderStage::Vertex);
    assert_eq!(module.artifact().entry_point, "main");
}
