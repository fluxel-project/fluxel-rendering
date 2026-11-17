use super::*;

#[test]
fn saved_ao_only_control_rejects_ignored_occlusion_at_runtime_tolerance() {
    let input = crate::scene::input(true, crate::scene::EYE).unwrap();
    let probes = super::super::reference(&input).unwrap();
    let raster = super::super::raster::Raster::from_input(&input).unwrap();
    let mut ignored = input.orm_bytes.clone();
    ignored.chunks_exact_mut(4).for_each(|texel| texel[0] = 255);
    assert!(
        probes
            .iter()
            .filter(|probe| probe.sphere_index == 0)
            .any(|probe| {
                let fragment = raster.at(probe.pixel).unwrap();
                let wrong = encode(
                    shade(
                        &input,
                        &fragment,
                        false,
                        &ignored,
                        &input.radiance_mip_bytes,
                    )
                    .unwrap(),
                );
                !close(wrong, probe.expected_forward)
            }),
        "ignoring the actual AO-only input must exceed the predeclared runtime tolerance"
    );
}
