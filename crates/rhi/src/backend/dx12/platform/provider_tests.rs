use super::*;

use crate::api::platform::requirements::{DeviceRequirements, LimitKey, OptionalFeature};

fn provider() -> Dx12Provider {
    Dx12Provider::new(DeviceInstanceId::new(0xD312_0001))
        .expect("the Windows DXGI provider must open for DX12 hardware tests")
}

#[test]
fn enumeration_publishes_complete_capability_snapshots() {
    let provider = provider();
    let adapters = provider
        .enumerate_adapters()
        .expect("DX12 adapter enumeration must probe each offered adapter")
        .expect("DX12 has portable adapter enumeration");

    assert!(
        !adapters.is_empty(),
        "a Windows DX12 test machine must expose at least one D3D12-capable adapter"
    );
    for adapter in adapters {
        let facts = adapter.available_capabilities();
        assert!(facts.supports_feature(OptionalFeature::Compute));
        assert!(
            facts.limit(LimitKey::MaxBufferSize).is_some(),
            "an offered adapter must carry probed limits, not a hollow snapshot"
        );
    }
}

#[test]
fn request_validates_supported_and_impossible_requirements() {
    let provider = provider();
    let accepted = DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new().require_feature(OptionalFeature::Compute),
    );
    assert!(
        provider.request_device(&accepted).is_ok(),
        "the selected D3D12 adapter advertises and must accept compute"
    );

    let refused = DeviceRequestDescriptor::new(
        AdapterSelection::Default,
        DeviceRequirements::new().require_limit_at_least(LimitKey::MaxBufferSize, u64::MAX),
    );
    let error = match provider.request_device(&refused) {
        Ok(_) => panic!("a limit above the probed capability must be refused"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(error.operation(), Some("Dx12Provider::request_device"));
}
