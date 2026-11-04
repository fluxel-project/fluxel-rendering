//! S5 Scene-to-Forward offscreen consumer.
//!
//! The workload deliberately owns normal public assets and scene input. It
//! exercises four mesh layouts, material and texture snapshots, one frame
//! policy, and one offscreen readback. It does not certify presentation: the
//! surface visual oracle is a separate integration workload.

use std::{error::Error, fs, path::Path};

use external_renderer::{ExternalClearPipeline, ExternalOpaquePipeline};
use fluxel_assets::AssetStore;
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryData, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::block_on;
mod fixtures;
mod frames;
mod isolation;
mod metrics;
use fixtures::{
    geometry_snapshot, interleaved_triangle, material_snapshot, mixed_triangle,
    position_only_material, position_triangle, separate_triangle, texture_snapshot,
    textured_material, verify_fixture_inputs, write_input_artifacts,
};
use isolation::{scene as isolation_scene, snapshot_identity_json};

const EXTENT: [u32; 2] = [96, 96];
const CLEAR: [u8; 4] = [0, 0, 0, 0];
const EDGE_EXCLUSION_PIXELS: f32 = 2.0;

/// Executes the first public Scene -> FramePipeline -> Graph -> RHI Forward path.
pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;

    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();

    let geometry = [
        geometry_snapshot(&geometry_store, interleaved_triangle())?,
        geometry_snapshot(&geometry_store, separate_triangle())?,
        geometry_snapshot(&geometry_store, mixed_triangle())?,
        geometry_snapshot(&geometry_store, position_triangle())?,
    ];
    let material = [
        material_snapshot(&material_store, textured_material("textured_color"))?,
        material_snapshot(&material_store, textured_material("textured_color_second"))?,
        material_snapshot(&material_store, textured_material("textured_color_third"))?,
        material_snapshot(&material_store, position_only_material())?,
    ];
    let texture = [
        texture_snapshot(&texture_store, [255, 0, 0, 255])?,
        texture_snapshot(&texture_store, [0, 255, 0, 255])?,
        texture_snapshot(&texture_store, [0, 0, 255, 255])?,
        texture_snapshot(&texture_store, [255, 255, 0, 255])?,
    ];

    verify_fixture_inputs(&[
        geometry[0].value().clone(),
        geometry[1].value().clone(),
        geometry[2].value().clone(),
        geometry[3].value().clone(),
    ])?;
    write_input_artifacts(output, &geometry, &material, &texture)?;
    let expected = expected_image();
    let mask = stable_mask();
    fs::write(output.join("s5-forward-expected.rgba"), &expected)?;
    fs::write(output.join("s5-forward-expected.ppm"), ppm(&expected)?)?;
    fs::write(output.join("s5-forward-stable-mask.bin"), &mask)?;
    fs::write(output.join("s5-forward-oracle.json"), oracle_json(&mask))?;

    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    for index in 0..geometry.len() {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geometry[index]),
                material: SceneAssetRef::from_snapshot(&material[index]),
                texture: SceneAssetRef::from_snapshot(&texture[index]),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let view = receipt.inserted_views()[0];
    let snapshot = scene.snapshot();
    let initial_frame_version = snapshot.version();
    let assets = ForwardAssets {
        geometry: geometry.to_vec(),
        materials: material.to_vec(),
        textures: texture.to_vec(),
    };
    verify_fixture_inputs(&[
        geometry[0].value().clone(),
        geometry[1].value().clone(),
        geometry[2].value().clone(),
        geometry[3].value().clone(),
    ])?;
    write_input_artifacts(output, &geometry, &material, &texture)?;

    // This executable consumes the policy from a separate crate. The renderer
    // receives only the public, statically selected FramePipeline contract.
    let pipeline = ExternalOpaquePipeline::new();
    let mut renderer = Renderer::new(device)?;
    let initial_stats = renderer.forward_cache_stats();
    let initial_compilations = renderer.compilations();
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3])?;
    let submission = block_on(renderer.submit_forward(&pipeline, snapshot, view, volume, &assets))?;
    let actual = block_on(renderer.readback(&submission))
        .map_err(|error| format!("S5 Forward readback: {error:?}"))?;
    fs::write(output.join("s5-forward-readback.rgba"), &actual)?;
    fs::write(output.join("s5-forward-readback.ppm"), ppm(&actual)?)?;
    verify_stable_image(&actual, &expected, &mask)?;
    if renderer.pending() != 0 {
        return Err("S5 Forward submission remained pending after its readback".into());
    }
    frames::run(
        &mut renderer,
        &mut scene,
        view,
        receipt.inserted(),
        &assets,
        &geometry_store,
        &material_store,
        &texture_store,
        output,
        &expected,
        &mask,
    )?;

    // `frames::run` leaves the scene empty. Exercise the external clear
    // policy through the actual Graph/RHI path as a distinct cached topology.
    let clear = ExternalClearPipeline::new();
    let clear_submission =
        block_on(renderer.submit_forward(&clear, scene.snapshot(), view, volume, &assets))?;
    let clear_actual = block_on(renderer.readback(&clear_submission))?;
    fs::write(
        output.join("s5-forward-external-clear-actual.rgba"),
        &clear_actual,
    )?;
    fs::write(
        output.join("s5-forward-external-clear-actual.ppm"),
        ppm(&clear_actual)?,
    )?;
    if clear_actual != vec![0; expected.len()] {
        return Err("external clear Forward policy did not produce the clear image".into());
    }
    metrics::write(
        output,
        initial_stats,
        renderer.forward_cache_stats(),
        initial_compilations,
        renderer.compilations(),
    )?;

    fs::write(
        output.join("s5-forward-workload.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"workload\": \"s5-scene-forward-four-layouts\",\n",
                "  \"extent\": [{}, {}],\n",
                "  \"layouts\": [\"interleaved\", \"separate\", \"mixed\", \"position-only\"],\n",
                "  \"records\": {},\n",
                "  \"materials\": {},\n",
                "  \"textures\": {},\n",
                "  \"frameVersion\": {},\n",
                "  \"readback\": \"s5-forward-readback.rgba\",\n",
                "  \"expected\": \"s5-forward-expected.rgba\",\n",
                "  \"oracle\": \"all stable interior and background pixels\",\n",
                "  \"externalPolicies\": [\"opaque\", \"clear\"],\n",
                "  \"externalClearReadback\": \"s5-forward-external-clear-actual.rgba\",\n",
                "  \"presentationEvidence\": \"pending separate surface workload\"\n",
                "}}\n"
            ),
            EXTENT[0],
            EXTENT[1],
            receipt.inserted().len(),
            material.len(),
            texture.len(),
            initial_frame_version,
        ),
    )?;
    Ok(())
}

/// Proves that completed Forward products remain owned by their originating
/// device when two Renderers consume identical logical asset snapshots.
///
/// The old renderer stays alive after its completion publishes its resident
/// products. The second renderer receives the same asset IDs and generations,
/// but must allocate and upload its own device-local products.
pub(crate) fn run_device_isolation<B>(
    old_device: Device<B>,
    new_device: Device<B>,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    if old_device.id() == new_device.id() {
        return Err("S5 device isolation requires distinct DeviceId values".into());
    }
    fs::create_dir_all(output)?;

    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry = [
        geometry_snapshot(&geometry_store, interleaved_triangle())?,
        geometry_snapshot(&geometry_store, separate_triangle())?,
        geometry_snapshot(&geometry_store, mixed_triangle())?,
        geometry_snapshot(&geometry_store, position_triangle())?,
    ];
    let material = [
        material_snapshot(&material_store, textured_material("textured_color"))?,
        material_snapshot(&material_store, textured_material("textured_color_second"))?,
        material_snapshot(&material_store, textured_material("textured_color_third"))?,
        material_snapshot(&material_store, position_only_material())?,
    ];
    let texture = [
        texture_snapshot(&texture_store, [255, 0, 0, 255])?,
        texture_snapshot(&texture_store, [0, 255, 0, 255])?,
        texture_snapshot(&texture_store, [0, 0, 255, 255])?,
        texture_snapshot(&texture_store, [255, 255, 0, 255])?,
    ];
    let assets = ForwardAssets {
        geometry: geometry.to_vec(),
        materials: material.to_vec(),
        textures: texture.to_vec(),
    };
    write_input_artifacts(output, &geometry, &material, &texture)?;
    let expected = expected_image();
    let mask = stable_mask();
    fs::write(output.join("s5-device-isolation-expected.rgba"), &expected)?;
    fs::write(
        output.join("s5-device-isolation-expected.ppm"),
        ppm(&expected)?,
    )?;

    let pipeline = ExternalOpaquePipeline::new();
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3])?;
    let (old_scene, old_view) = isolation_scene(&geometry, &material, &texture)?;
    let mut old_renderer = Renderer::new(old_device.clone())?;
    let old_submission = block_on(old_renderer.submit_forward(
        &pipeline,
        old_scene.snapshot(),
        old_view,
        volume,
        &assets,
    ))?;
    let old_actual = block_on(old_renderer.readback(&old_submission))?;
    fs::write(
        output.join("s5-device-isolation-old-actual.rgba"),
        &old_actual,
    )?;
    fs::write(
        output.join("s5-device-isolation-old-actual.ppm"),
        ppm(&old_actual)?,
    )?;
    verify_stable_image(&old_actual, &expected, &mask)?;
    let old_stats = old_renderer.forward_cache_stats();
    if old_stats.resident_copy_bytes == 0 || old_renderer.pending() != 0 {
        return Err("old device did not publish completed Forward resident products".into());
    }

    let (new_scene, new_view) = isolation_scene(&geometry, &material, &texture)?;
    let mut new_renderer = Renderer::new(new_device.clone())?;
    let new_submission = block_on(new_renderer.submit_forward(
        &pipeline,
        new_scene.snapshot(),
        new_view,
        volume,
        &assets,
    ))?;
    let new_actual = block_on(new_renderer.readback(&new_submission))?;
    fs::write(
        output.join("s5-device-isolation-new-actual.rgba"),
        &new_actual,
    )?;
    fs::write(
        output.join("s5-device-isolation-new-actual.ppm"),
        ppm(&new_actual)?,
    )?;
    verify_stable_image(&new_actual, &expected, &mask)?;
    let new_stats = new_renderer.forward_cache_stats();
    if new_stats.geometry_hits != 0
        || new_stats.texture_hits != 0
        || new_stats.resident_copy_bytes == 0
    {
        return Err("new device adopted completed Forward products from the old device".into());
    }
    let old_reuse_before = old_renderer.forward_cache_stats();
    let old_reuse_submission = block_on(old_renderer.submit_forward(
        &pipeline,
        old_scene.snapshot(),
        old_view,
        volume,
        &assets,
    ))?;
    let old_reuse_actual = block_on(old_renderer.readback(&old_reuse_submission))?;
    fs::write(
        output.join("s5-device-isolation-old-reuse-actual.rgba"),
        &old_reuse_actual,
    )?;
    fs::write(
        output.join("s5-device-isolation-old-reuse-actual.ppm"),
        ppm(&old_reuse_actual)?,
    )?;
    verify_stable_image(&old_reuse_actual, &expected, &mask)?;
    let old_reuse_after = old_renderer.forward_cache_stats();
    let old_reuse_geometry_hits = old_reuse_after.geometry_hits - old_reuse_before.geometry_hits;
    let old_reuse_texture_hits = old_reuse_after.texture_hits - old_reuse_before.texture_hits;
    let old_reuse_copy_bytes =
        old_reuse_after.resident_copy_bytes - old_reuse_before.resident_copy_bytes;
    if old_reuse_geometry_hits != 4 || old_reuse_texture_hits != 4 || old_reuse_copy_bytes != 0 {
        return Err("old device did not retain its completed Forward resident cache".into());
    }
    let shared_assets = format!(
        concat!(
            "{{\n",
            "  \"geometry\": {},\n",
            "  \"materials\": {},\n",
            "  \"textures\": {}\n",
            "}}\n"
        ),
        snapshot_identity_json(&geometry),
        snapshot_identity_json(&material),
        snapshot_identity_json(&texture),
    );
    fs::write(
        output.join("s5-device-isolation-shared-assets.json"),
        shared_assets,
    )?;
    fs::write(
        output.join("s5-device-isolation.json"),
        format!(
            concat!(
                "{{\n",
                "  \"oldDeviceId\": {},\n",
                "  \"newDeviceId\": {},\n",
                "  \"sharedLogicalAssetGenerations\": true,\n",
                "  \"oldCompletedProductsRetained\": true,\n",
                "  \"newDeviceGeometryHits\": {},\n",
                "  \"newDeviceTextureHits\": {},\n",
                "  \"newDeviceResidentCopyBytes\": {},\n",
                "  \"oldReuseGeometryHits\": {},\n",
                "  \"oldReuseTextureHits\": {},\n",
                "  \"oldReuseResidentCopyBytes\": {}\n",
                "}}\n"
            ),
            crate::shader_admission::json_string(&format!("{:?}", old_device.id())),
            crate::shader_admission::json_string(&format!("{:?}", new_device.id())),
            new_stats.geometry_hits,
            new_stats.texture_hits,
            new_stats.resident_copy_bytes,
            old_reuse_geometry_hits,
            old_reuse_texture_hits,
            old_reuse_copy_bytes,
        ),
    )?;
    fs::write(
        output.join("s5-device-isolation-workload.json"),
        "{\n  \"schemaVersion\": 1,\n  \"workload\": \"s5-forward-two-device-cache-isolation\",\n  \"sharedAssets\": \"s5-device-isolation-shared-assets.json\",\n  \"oldCompletedProductsRetained\": true,\n  \"oldReadback\": \"s5-device-isolation-old-actual.rgba\",\n  \"newReadback\": \"s5-device-isolation-new-actual.rgba\",\n  \"oldReuseReadback\": \"s5-device-isolation-old-reuse-actual.rgba\"\n}\n",
    )?;
    Ok(())
}

fn expected_image() -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(EXTENT[0] * EXTENT[1] * 4).unwrap()];
    for pixel in bytes.chunks_exact_mut(4) {
        pixel.copy_from_slice(&CLEAR);
    }
    for (triangle, color) in triangle_pixels().into_iter().zip([
        [255, 0, 0, 255],
        [255, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]) {
        paint_triangle(&mut bytes, triangle, color);
    }
    bytes
}

fn triangle_pixels() -> [[[f32; 2]; 3]; 4] {
    [
        [[8.64, 38.4], [29.76, 38.4], [19.2, 11.52]],
        [[34.56, 38.4], [55.68, 38.4], [45.12, 11.52]],
        [[60.48, 38.4], [81.6, 38.4], [71.04, 11.52]],
        [[36.48, 84.48], [59.52, 84.48], [48.0, 57.6]],
    ]
}

fn paint_triangle(bytes: &mut [u8], triangle: [[f32; 2]; 3], color: [u8; 4]) {
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            if contains(triangle, point) {
                let offset = usize::try_from((y * EXTENT[0] + x) * 4).unwrap();
                bytes[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }
}

fn contains(vertices: [[f32; 2]; 3], point: [f32; 2]) -> bool {
    let signs = [
        edge(vertices[0], vertices[1], point),
        edge(vertices[1], vertices[2], point),
        edge(vertices[2], vertices[0], point),
    ];
    signs.iter().all(|sign| *sign >= 0.0) || signs.iter().all(|sign| *sign <= 0.0)
}

fn edge(a: [f32; 2], b: [f32; 2], point: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}

fn stable_mask() -> Vec<u8> {
    let mut mask = Vec::with_capacity(usize::try_from(EXTENT[0] * EXTENT[1]).unwrap());
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let stable = triangle_pixels().iter().all(|triangle| {
                (0..3).all(|edge_index| {
                    distance_to_segment(triangle[edge_index], triangle[(edge_index + 1) % 3], point)
                        > EDGE_EXCLUSION_PIXELS
                })
            });
            mask.push(u8::from(stable));
        }
    }
    mask
}

fn distance_to_segment(a: [f32; 2], b: [f32; 2], point: [f32; 2]) -> f32 {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let length_squared = delta[0] * delta[0] + delta[1] * delta[1];
    let parameter = if length_squared == 0.0 {
        0.0
    } else {
        (((point[0] - a[0]) * delta[0] + (point[1] - a[1]) * delta[1]) / length_squared)
            .clamp(0.0, 1.0)
    };
    let closest = [a[0] + parameter * delta[0], a[1] + parameter * delta[1]];
    let dx = point[0] - closest[0];
    let dy = point[1] - closest[1];
    (dx * dx + dy * dy).sqrt()
}

fn verify_stable_image(actual: &[u8], expected: &[u8], mask: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected_len = usize::try_from(EXTENT[0] * EXTENT[1] * 4)?;
    let expected_mask_len = usize::try_from(EXTENT[0] * EXTENT[1])?;
    if actual.len() != expected_len
        || expected.len() != expected_len
        || mask.len() != expected_mask_len
    {
        return Err(format!(
            "S5 Forward oracle shape was actual={}, expected={}, mask={}",
            actual.len(),
            expected.len(),
            mask.len(),
        )
        .into());
    }
    let mut interior = 0;
    let mut background = 0;
    for (pixel, stable) in mask.iter().copied().enumerate() {
        if stable == 0 {
            continue;
        }
        let offset = pixel * 4;
        let expected_pixel = &expected[offset..offset + 4];
        if expected_pixel == CLEAR.as_slice() {
            background += 1;
        } else {
            interior += 1;
        }
        if actual[offset..offset + 4] != expected[offset..offset + 4] {
            let x = pixel % usize::try_from(EXTENT[0])?;
            let y = pixel / usize::try_from(EXTENT[0])?;
            return Err(format!(
                "S5 Forward stable pixel ({x}, {y}) was {:?}, expected {expected_pixel:?}",
                &actual[offset..offset + 4],
            )
            .into());
        }
    }
    if interior == 0 || background == 0 {
        return Err(
            "S5 Forward stable mask did not cover both triangle interiors and background".into(),
        );
    }
    Ok(())
}

fn oracle_json(mask: &[u8]) -> String {
    let stable = mask.iter().filter(|value| **value != 0).count();
    format!(
        concat!(
            "{{\n",
            "  \"schemaVersion\": 1,\n",
            "  \"edgeExclusionPixels\": {},\n",
            "  \"mask\": \"s5-forward-stable-mask.bin\",\n",
            "  \"stablePixelCount\": {},\n",
            "  \"expected\": \"s5-forward-expected.rgba\",\n",
            "  \"actual\": \"s5-forward-readback.rgba\"\n",
            "}}\n"
        ),
        EDGE_EXCLUSION_PIXELS, stable,
    )
}

fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let expected_len = usize::try_from(EXTENT[0] * EXTENT[1] * 4)?;
    if rgba.len() != expected_len {
        return Err("cannot write S5 Forward PPM from malformed RGBA data".into());
    }
    let mut result = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        result.extend_from_slice(&pixel[..3]);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "tests/s5_forward.rs"]
mod tests;
