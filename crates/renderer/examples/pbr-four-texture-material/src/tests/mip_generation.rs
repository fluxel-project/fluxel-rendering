use super::*;

fn max_error(a: &[Probe], b: &[Probe]) -> u8 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.expected.iter().zip(b.expected))
        .map(|(a, b)| a.abs_diff(b))
        .max()
        .unwrap_or(0)
}

#[test]
fn explicit_lods_have_distinct_independent_predictions() {
    let zero = reference(&input(Case::Lod0).unwrap()).unwrap();
    let one = reference(&input(Case::Lod1).unwrap()).unwrap();
    let half = reference(&input(Case::Lod15).unwrap()).unwrap();
    let two = reference(&input(Case::Lod2).unwrap()).unwrap();
    assert!(max_error(&zero, &one) > TOLERANCE);
    assert!(max_error(&one, &half) > TOLERANCE);
    assert!(max_error(&half, &two) > TOLERANCE);
}

#[test]
fn wrong_lod_zero_and_non_generated_chain_are_rejected() {
    let data = input(Case::Lod15).unwrap();
    let expected = reference(&data).unwrap();
    let levels = generated_levels(&data.source, [4, 4]);
    let wrong_zero = expected
        .iter()
        .map(|probe| Probe {
            expected: trilinear(&levels, 0., probe.uv),
            ..probe.clone()
        })
        .collect::<Vec<_>>();
    let wrong_unfiltered = expected
        .iter()
        .map(|probe| Probe {
            expected: nearest(&levels[1].1, [2, 2], probe.uv),
            ..probe.clone()
        })
        .collect::<Vec<_>>();
    assert!(max_error(&expected, &wrong_zero) > TOLERANCE);
    assert!(max_error(&expected, &wrong_unfiltered) > TOLERANCE);
}

#[test]
fn source_only_change_is_visible_at_same_lod() {
    let baseline = reference(&input(Case::Lod1).unwrap()).unwrap();
    let changed = reference(&input(Case::SourceOnly).unwrap()).unwrap();
    assert!(max_error(&baseline, &changed) > TOLERANCE);
}

#[test]
fn actual_consumer_declares_full_chain_and_explicit_lod_shader() {
    let data = input(Case::Lod15).unwrap();
    assert_eq!(data.assets.textures[0].value().mip_levels, MIP_LEVELS);
    assert_eq!(data.assets.textures[0].value().rgba8.len(), 4 * 4 * 4);
    assert_eq!(data.geometry_bytes.len(), 6 * 20);
    assert!(format!("{:?}", data.material).contains("textureSampleLevel"));
}

fn nearest(bytes: &[u8], extent: [u32; 2], uv: [f64; 2]) -> [u8; 4] {
    let x = (uv[0] * f64::from(extent[0]))
        .floor()
        .clamp(0., f64::from(extent[0] - 1)) as u32;
    let y = (uv[1] * f64::from(extent[1]))
        .floor()
        .clamp(0., f64::from(extent[1] - 1)) as u32;
    let offset = ((y * extent[0] + x) * 4) as usize;
    bytes[offset..offset + 4].try_into().unwrap()
}

#[test]
fn downsampled_texels_match_hand_computed_integer_averages() {
    let levels = generated_levels(&source_bytes(false), [4, 4]);
    assert_eq!(&levels[1].1[..4], &[112, 82, 108, 255]);
    assert_eq!(&levels[2].1[..], &[97, 139, 112, 255]);
}

#[test]
fn nearest_mip_and_uninitialized_tail_fail_the_same_acceptance_oracle() {
    let input = input(Case::Lod15).unwrap();
    let expected = reference(&input).unwrap();
    let levels = generated_levels(&input.source, [4, 4]);
    let nearest_mip: Vec<_> = expected
        .iter()
        .map(|probe| Probe {
            expected: trilinear(&levels, 2., probe.uv),
            ..probe.clone()
        })
        .collect();
    assert!(max_error(&expected, &nearest_mip) > TOLERANCE);
    let mut missing = levels;
    for (_, pixels) in missing.iter_mut().skip(1) {
        pixels.fill(0);
    }
    let uninitialized: Vec<_> = expected
        .iter()
        .map(|probe| Probe {
            expected: trilinear(&missing, 1.5, probe.uv),
            ..probe.clone()
        })
        .collect();
    assert!(max_error(&expected, &uninitialized) > TOLERANCE);
}

#[test]
fn actual_explicit_lod_material_prepares_through_the_public_forward_consumer() {
    let input = input(Case::Lod15).unwrap();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let selected = FramePipeline::prepare(
        &pipeline,
        &input.snapshot,
        input.view,
        RenderBounds::new([-5.; 3], [5.; 3]).unwrap(),
    )
    .unwrap();
    let draws = prepare_forward(
        &selected,
        &input.assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .unwrap();
    assert_eq!(draws.len(), 1);
    assert_eq!(draws[0].tint(), [1.5, 1., 1., 1.]);
}
