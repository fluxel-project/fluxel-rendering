use crate::EnvironmentIrradianceData;

#[test]
fn irradiance_payload_requires_exact_tight_rgba8_extent() {
    assert!(
        EnvironmentIrradianceData {
            extent: [2, 1],
            rgba8: vec![0; 8],
        }
        .is_valid()
    );
    assert!(
        !EnvironmentIrradianceData {
            extent: [2, 1],
            rgba8: vec![0; 7],
        }
        .is_valid()
    );
}

#[test]
fn irradiance_payload_rejects_zero_and_overflowed_extent() {
    assert!(
        !EnvironmentIrradianceData {
            extent: [0, 1],
            rgba8: Vec::new(),
        }
        .is_valid()
    );
    assert!(
        !EnvironmentIrradianceData {
            extent: [u32::MAX, u32::MAX],
            rgba8: Vec::new(),
        }
        .is_valid()
    );
}
