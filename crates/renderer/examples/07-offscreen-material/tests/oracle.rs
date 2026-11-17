use renderer_07_offscreen_material::{oracle, scene};

#[test]
fn oracle_requires_producer_color_at_interior_and_clear_at_background() {
    let dragon = scene::mirror_geometry();
    let expected = oracle::build(&dragon, &receiver_plane()).expect("observable fixture");
    assert_eq!(expected.color_probes.len(), 4);

    let bytes = (256 * 256 * 4) as usize;
    let mut primary = vec![0; bytes];
    let mut control = vec![0; bytes];
    for probe in &expected.color_probes {
        write(&mut primary, probe.final_pixel, oracle::PRIMARY_ALBEDO);
        write(&mut control, probe.final_pixel, oracle::CONTROL_ALBEDO);
    }
    oracle::verify_pair(&primary, &control, &expected).expect("independent oracle accepts pair");

    write(
        &mut control,
        expected.color_probes[0].final_pixel,
        oracle::CLEAR_RGBA,
    );
    assert!(oracle::verify_pair(&primary, &control, &expected).is_err());
}

fn receiver_plane() -> fluxel_renderer::GeometryData {
    let mut plane = scene::mirror_geometry();
    for vertex in plane.streams[0].chunks_exact_mut(32) {
        let x = f32::from_le_bytes(vertex[0..4].try_into().expect("x"));
        let y = f32::from_le_bytes(vertex[4..8].try_into().expect("y"));
        vertex[0..4].copy_from_slice(&(x * 5.0).to_le_bytes());
        vertex[4..8].copy_from_slice(&0.0_f32.to_le_bytes());
        vertex[8..12].copy_from_slice(&(y * 5.0).to_le_bytes());
    }
    plane
}

fn write(image: &mut [u8], pixel: [u32; 2], value: [u8; 4]) {
    let at = ((pixel[1] * 256 + pixel[0]) * 4) as usize;
    image[at..at + 4].copy_from_slice(&value);
}

#[test]
#[ignore = "requires unredistributed local RHI reference assets"]
fn selected_rhi_assets_have_four_interior_probes_and_a_clear_probe() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let assets = root.join("../../../../../fluxel-rhi/crates/rhi/examples/assets/models");
    let dragon = renderer_07_offscreen_material::gltf::load(
        &std::fs::read(assets.join("chinesedragon.gltf")).expect("dragon asset"),
    )
    .expect("decode dragon");
    let plane = renderer_07_offscreen_material::gltf::load(
        &std::fs::read(assets.join("plane.gltf")).expect("plane asset"),
    )
    .expect("decode plane");
    let inputs =
        renderer_07_offscreen_material::inputs::build_capture_scene(dragon.clone(), plane.clone())
            .expect("scene inputs");
    let producer = inputs
        .capture_snapshot
        .record(
            inputs
                .capture_snapshot
                .record_ids()
                .next()
                .expect("producer"),
        )
        .expect("producer record");
    assert_eq!(
        oracle::clip_depth_range(&dragon, producer.transform).expect("dragon depth"),
        [0.5, 0.5]
    );
    assert_eq!(
        oracle::clip_depth_range(&plane, inputs.receiver_transform).expect("plane depth"),
        [0.5, 0.5]
    );
    let oracle = oracle::build(&dragon, &plane).expect("four interior and clear probes");
    assert_eq!(oracle.color_probes.len(), 4);
}
