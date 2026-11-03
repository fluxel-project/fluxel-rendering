#[path = "../src/executable_cache.rs"]
mod executable_cache;

use executable_cache::{ExecutableCache, ExecutableCacheError, ExecutableKey, RouteFingerprint};
use fluxel_rhi::api::{DeviceId, ShaderFormat, ShaderRoute, ShaderTarget};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineFacts {
    vertex_layout: &'static str,
    bindings: &'static str,
    color_target: &'static str,
}

fn route(abi_key: &str) -> RouteFingerprint {
    RouteFingerprint::from_route(&ShaderRoute::Direct(ShaderTarget {
        format: ShaderFormat::Wgsl,
        environment: "test-environment".into(),
        abi_key: abi_key.into(),
    }))
    .unwrap()
}

fn key(
    device: DeviceId,
    capabilities: &'static str,
    abi_key: &str,
    pipeline: PipelineFacts,
) -> ExecutableKey<&'static str, &'static str, PipelineFacts> {
    ExecutableKey::new(
        "logical-source-v1",
        device,
        capabilities,
        route(abi_key),
        pipeline,
    )
}

#[test]
fn executable_key_isolates_device_route_capabilities_and_pipeline_facts() {
    let first_device = DeviceId::allocate().unwrap();
    let replacement_device = DeviceId::allocate().unwrap();
    let pipeline = PipelineFacts {
        vertex_layout: "position:f32x2,color:f32x4",
        bindings: "group0:uniform+texture",
        color_target: "rgba8unorm",
    };
    let first = key(first_device, "compute=false", "abi-a", pipeline.clone());
    let mut cache = ExecutableCache::new();
    cache.insert(first.clone(), "first executable").unwrap();

    assert_eq!(cache.get(&first).unwrap(), Some(&"first executable"));
    assert_eq!(
        cache
            .get(&key(
                first_device,
                "compute=false",
                "abi-b",
                pipeline.clone()
            ))
            .unwrap(),
        None,
    );
    assert_eq!(
        cache
            .get(&key(
                first_device,
                "compute=true",
                "abi-a",
                pipeline.clone()
            ))
            .unwrap(),
        None,
    );
    assert_eq!(
        cache
            .get(&key(
                first_device,
                "compute=false",
                "abi-a",
                PipelineFacts {
                    vertex_layout: "position:f32x3,color:f32x4",
                    ..pipeline.clone()
                },
            ))
            .unwrap(),
        None,
    );
    assert_eq!(
        cache
            .get(&key(replacement_device, "compute=false", "abi-a", pipeline))
            .unwrap(),
        None,
    );
}

#[test]
fn device_loss_evicts_and_permanently_rejects_that_device_domain() {
    let lost_device = DeviceId::allocate().unwrap();
    let replacement_device = DeviceId::allocate().unwrap();
    let facts = PipelineFacts {
        vertex_layout: "position:f32x2",
        bindings: "none",
        color_target: "rgba8unorm",
    };
    let lost_key = key(lost_device, "raster", "abi-a", facts.clone());
    let replacement_key = key(replacement_device, "raster", "abi-a", facts);
    let mut cache = ExecutableCache::new();
    cache.insert(lost_key.clone(), 1_u32).unwrap();
    cache.insert(replacement_key.clone(), 2_u32).unwrap();

    cache.invalidate_device(lost_device);

    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.get(&lost_key),
        Err(ExecutableCacheError::DeviceLost(lost_device)),
    );
    assert_eq!(
        cache.insert(lost_key, 3_u32),
        Err(ExecutableCacheError::DeviceLost(lost_device)),
    );
    assert_eq!(cache.get(&replacement_key).unwrap(), Some(&2_u32));
}

#[test]
fn replacement_retirement_evicts_without_quarantining_a_live_device() {
    let retired_device = DeviceId::allocate().unwrap();
    let facts = PipelineFacts {
        vertex_layout: "position:f32x2",
        bindings: "none",
        color_target: "rgba8unorm",
    };
    let retired_key = key(retired_device, "raster", "abi-a", facts);
    let mut cache = ExecutableCache::new();
    cache.insert(retired_key.clone(), 1_u32).unwrap();

    cache.evict_device(retired_device);

    assert_eq!(cache.get(&retired_key).unwrap(), None);
    assert_eq!(cache.insert(retired_key.clone(), 2_u32).unwrap(), None);
    assert_eq!(cache.get(&retired_key).unwrap(), Some(&2_u32));
}

#[test]
fn unsupported_route_never_creates_a_cache_fingerprint() {
    assert_eq!(
        RouteFingerprint::from_route(&ShaderRoute::Unsupported),
        Err(ExecutableCacheError::UnsupportedRoute),
    );
}
