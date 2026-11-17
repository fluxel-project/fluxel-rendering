use super::*;

#[test]
fn box_chain_uses_actual_rgba8_bytes() {
    let base = [0, 0, 0, 255, 4, 0, 0, 255, 8, 0, 0, 255, 12, 0, 0, 255];
    let chain = mips(&base, [2, 2]).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[1], [6, 0, 0, 255]);
}

#[test]
fn mirror_sampling_is_not_clamp_sampling() {
    let bytes = [0, 0, 0, 255, 255, 0, 0, 255];
    let mirrored = bilinear(&bytes, [2, 1], [1.75, 0.5], false, false);
    let clamped = bilinear(&bytes, [2, 1], [1.75, 0.5], true, false);
    assert!(mirrored[0] < clamped[0]);
}

#[test]
fn negative_mirror_coordinates_use_positive_fractional_weights() {
    let bytes = [0, 0, 0, 255, 255, 0, 0, 255];
    assert_eq!(
        bilinear(&bytes, [2, 1], [-0.5, 0.5], false, false),
        [0.5, 0.0, 0.0, 1.0]
    );
}

#[test]
#[ignore = "requires the original external RHI tunnel asset pair"]
fn actual_tunnel_assets_have_independent_perspective_sampler_probes() {
    let model = std::env::var_os("FLUXEL_TUNNEL_TEST_MODEL").expect("model fixture path");
    let image = std::env::var_os("FLUXEL_TUNNEL_TEST_IMAGE").expect("image fixture path");
    let input = scene::input(
        std::path::Path::new(&model),
        std::path::Path::new(&image),
        1,
    )
    .unwrap();
    use fluxel_renderer::{
        BoundsVisibility, ForwardCpuPipeline, FramePipeline, RenderBounds, prepare_forward,
    };
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let selected = pipeline
        .prepare(
            &input.snapshot,
            input.view,
            RenderBounds::new([-64.; 3], [64.; 3]).unwrap(),
        )
        .unwrap();
    let prepared = prepare_forward(
        &selected,
        &input.assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .unwrap();
    assert_eq!(
        prepared.len(),
        1,
        "actual public Material-to-Shader preparation"
    );
    let probes = reference(&input).unwrap();
    assert!(probes.len() >= 4);
    println!(
        "geometry vertices={}, image={:?}, probes={probes:?}",
        input.geometry.vertices.end, input.extent
    );
    if let Some(output) = std::env::var_os("FLUXEL_TUNNEL_ORACLE_OUTPUT") {
        let output = std::path::Path::new(&output);
        std::fs::create_dir_all(output).unwrap();
        std::fs::write(output.join("model.gltf"), input.model_bytes).unwrap();
        std::fs::write(output.join("image.ktx"), input.asset_bytes).unwrap();
        std::fs::write(output.join("geometry.bin"), &input.geometry.streams[0]).unwrap();
        std::fs::write(output.join("mip0.rgba"), input.image).unwrap();
        write(output, &probes).unwrap();
    }
}

#[test]
fn isotropic_footprint_uses_the_larger_jacobian_singular_value() {
    let (axis, ideal) = footprint_lods([1.0, 0.0], [1.0, 1.0]);
    assert!((axis - 0.5).abs() < 1e-6);
    let expected = ((3.0 + 5.0_f32.sqrt()) * 0.5).sqrt().log2();
    assert!((ideal - expected).abs() < 1e-6);
    assert!(ideal > axis);
}
