//! Multi-frame Forward cache and snapshot workload.

use std::{error::Error, fs, path::Path};

use external_renderer::{ExternalClearPipeline, ExternalOpaquePipeline};
use fluxel_assets::{AssetStore, Production, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryData, RenderBounds, RenderChange, RenderChangeSet, RenderId, RenderScene,
    RenderTransform, RenderViewId, Renderer, RendererError, SceneAssetRef,
};
use fluxel_rhi::backend::{
    ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
    ResourceBackend, ShaderBackend,
};

use super::{ppm, textured_material, triangle_pixels};
#[path = "evidence.rs"]
mod evidence;
use evidence::{frame_expected, read_and_verify, read_replacement, write_frame_inputs};

/// Submits immutable scene snapshots until Forward's completion-owned slots fill.
///
/// The caller has already completed the first single-frame workload, so this
/// workload begins with warm resident geometry and texture products.
pub(super) fn run<B>(
    renderer: &mut Renderer<B>,
    scene: &mut RenderScene<ForwardResources>,
    view: RenderViewId,
    ids: &[RenderId],
    assets: &ForwardAssets,
    geometry_store: &AssetStore<GeometryAsset, GeometryData, String>,
    material_store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    texture_store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    output: &Path,
    expected: &[u8],
    mask: &[u8],
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
    if ids.len() != 4 {
        return Err(format!(
            "S5 multi-frame fixture requires four records, got {}",
            ids.len()
        )
        .into());
    }
    let unchanged = frame_expected([true, true, true, true]);
    if expected != unchanged {
        return Err("S5 multi-frame input expected image does not match the frozen fixture".into());
    }
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3])?;
    let pipeline = ExternalOpaquePipeline::new();
    let initial_stats = renderer.forward_cache_stats();
    let initial_compilations = renderer.compilations();

    let expected0 = frame_expected([true, true, true, true]);
    write_frame_inputs(
        output,
        0,
        scene.snapshot().version(),
        false,
        [1.0; 4],
        &expected0,
    )?;
    let first = crate::block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        assets,
    ))?;
    let cached_compilations = renderer.compilations();
    if cached_compilations != initial_compilations + 1 {
        return Err(
            "S5 completed imports did not select the expected new upload-free topology".into(),
        );
    }

    let mut first_record = scene.record(ids[0])?.clone();
    first_record.transform = translated_away()?;
    first_record.bounds = RenderBounds::new([-1.0, -1.0, 2.0], [1.0, 1.0, 2.0])?;
    let mut move_first = RenderChangeSet::new();
    move_first.push(RenderChange::Update(ids[0], first_record));
    scene.apply(&move_first)?;
    let expected1 = frame_expected([false, true, true, true]);
    write_frame_inputs(
        output,
        1,
        scene.snapshot().version(),
        true,
        [1.0; 4],
        &expected1,
    )?;
    let second = crate::block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        assets,
    ))?;
    let fresh_compilations = renderer.compilations();
    if fresh_compilations != cached_compilations + 1 {
        return Err(
            "S5 first fresh slot did not create the expected incoming-state plan variant".into(),
        );
    }

    let mut second_record = scene.record(ids[1])?.clone();
    second_record.resources.tint = [0.0; 4];
    let mut darken_second = RenderChangeSet::new();
    darken_second.push(RenderChange::Update(ids[1], second_record));
    scene.apply(&darken_second)?;
    let expected2 = frame_expected([false, false, true, true]);
    write_frame_inputs(
        output,
        2,
        scene.snapshot().version(),
        true,
        [0.0; 4],
        &expected2,
    )?;
    let third = crate::block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        assets,
    ))?;

    if renderer.pending() != 3 {
        return Err(format!(
            "S5 multi-frame burst retained {} submissions, expected 3",
            renderer.pending()
        )
        .into());
    }
    if !matches!(
        crate::block_on(
            renderer.submit_forward(&pipeline, scene.snapshot(), view, volume, assets,)
        ),
        Err(RendererError::Backpressure)
    ) {
        return Err("S5 multi-frame fourth submission was not refused with Backpressure".into());
    }
    if renderer.pending() != 3 {
        return Err("S5 multi-frame backpressure refusal changed pending ownership".into());
    }
    let burst_stats = renderer.forward_cache_stats();
    let opaque_candidate_delta = burst_stats.opaque_candidates - initial_stats.opaque_candidates;
    let opaque_batch_delta = burst_stats.opaque_batches - initial_stats.opaque_batches;
    let opaque_draw_delta = burst_stats.opaque_draws - initial_stats.opaque_draws;
    fs::write(
        output.join("s5-forward-cache-facts.json"),
        format!(
            "{{\"residentCopyDelta\":{},\"geometryHitDelta\":{},\"textureHitDelta\":{},\"frameReuseDelta\":{},\"opaqueCandidateDelta\":{},\"opaqueBatchDelta\":{},\"opaqueDrawDelta\":{},\"pending\":{},\"warmPlanCompilations\":{},\"freshStateCompilations\":{},\"finalCompilations\":{}}}\n",
            burst_stats.resident_copy_bytes - initial_stats.resident_copy_bytes,
            burst_stats.geometry_hits - initial_stats.geometry_hits,
            burst_stats.texture_hits - initial_stats.texture_hits,
            burst_stats.frame_reuses - initial_stats.frame_reuses,
            opaque_candidate_delta,
            opaque_batch_delta,
            opaque_draw_delta,
            renderer.pending(),
            cached_compilations,
            fresh_compilations,
            renderer.compilations()
        ),
    )?;
    if burst_stats.resident_copy_bytes != initial_stats.resident_copy_bytes {
        return Err("S5 multi-frame warm cache unexpectedly copied static resident assets".into());
    }
    if burst_stats.geometry_hits - initial_stats.geometry_hits != 10
        || burst_stats.texture_hits - initial_stats.texture_hits != 10
    {
        return Err(
            "S5 multi-frame burst did not reuse completed geometry and texture products".into(),
        );
    }
    if burst_stats.frame_reuses - initial_stats.frame_reuses != 1 {
        return Err(
            "S5 reused a pending frame allocation or failed to reuse the completed slot".into(),
        );
    }
    // The three submitted snapshots contain four, then three, then three
    // visible opaque records. Every fixture material/layout pair is distinct,
    // so each preserved-order draw forms one contiguous compatibility batch.
    if (
        opaque_candidate_delta,
        opaque_batch_delta,
        opaque_draw_delta,
    ) != (10, 10, 10)
    {
        return Err(format!(
            "S5 opaque statistics were ({opaque_candidate_delta}, {opaque_batch_delta}, {opaque_draw_delta}), expected (10, 10, 10)"
        )
        .into());
    }
    if renderer.compilations() != fresh_compilations {
        return Err(
            "S5 multi-frame transform/tint updates recompiled a compatible Forward plan".into(),
        );
    }

    read_and_verify(renderer, &first, output, 0, &expected0, mask)?;
    read_and_verify(renderer, &second, output, 1, &expected1, mask)?;
    read_and_verify(renderer, &third, output, 2, &expected2, mask)?;
    if renderer.pending() != 0 {
        return Err("S5 multi-frame readbacks did not retire every submission".into());
    }

    replacement_frames(
        renderer,
        scene,
        view,
        ids,
        assets,
        geometry_store,
        material_store,
        texture_store,
        output,
        mask,
    )?;

    let mut clear = RenderChangeSet::new();
    for id in ids {
        clear.push(RenderChange::Remove(*id));
    }
    scene.apply(&clear)?;
    let empty_expected = vec![0; expected.len()];
    fs::write(
        output.join("s5-forward-empty-expected.rgba"),
        &empty_expected,
    )?;
    fs::write(
        output.join("s5-forward-empty-expected.ppm"),
        ppm(&empty_expected)?,
    )?;
    let clear = ExternalClearPipeline::new();
    let empty =
        crate::block_on(renderer.submit_forward(&clear, scene.snapshot(), view, volume, assets))?;
    let empty_actual = crate::block_on(renderer.readback(&empty))?;
    fs::write(output.join("s5-forward-empty-actual.rgba"), &empty_actual)?;
    fs::write(
        output.join("s5-forward-empty-actual.ppm"),
        ppm(&empty_actual)?,
    )?;
    if empty_actual != empty_expected {
        return Err("S5 empty Forward frame was not an all-zero clear".into());
    }
    Ok(())
}

/// Replaces three logical asset generations while the previous snapshot remains pending.
///
/// The old submission retains its scene snapshot and exact asset generations. The new
/// submission pins replacements with the same logical identities, then both readbacks
/// are checked against separately persisted expected images.
#[allow(clippy::too_many_arguments)]
fn replacement_frames<B>(
    renderer: &mut Renderer<B>,
    scene: &mut RenderScene<ForwardResources>,
    view: RenderViewId,
    ids: &[RenderId],
    assets: &ForwardAssets,
    geometry_store: &AssetStore<GeometryAsset, GeometryData, String>,
    material_store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    texture_store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    output: &Path,
    mask: &[u8],
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
    let volume = RenderBounds::new([-2.0; 3], [2.0; 3])?;
    let pipeline = ExternalOpaquePipeline::new();

    // The previous burst changed transform and tint. Restore its scene records to
    // immutable original inputs before retaining the old generation submission.
    let mut restore = RenderChangeSet::new();
    for (index, id) in ids.iter().copied().enumerate() {
        let mut record = scene.record(id)?.clone();
        record.resources = original_resources(assets, index)?;
        record.transform = RenderTransform::identity();
        record.bounds = RenderBounds::new([-1.0; 3], [1.0; 3])?;
        restore.push(RenderChange::Update(id, record));
    }
    scene.apply(&restore)?;

    let old_expected = frame_expected([true, true, true, true]);
    write_replacement_input(
        output,
        "old",
        scene.snapshot().version(),
        assets,
        &old_expected,
    )?;
    let old = crate::block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        assets,
    ))?;

    let replacement_geometry_data = clip_geometry(assets.geometry[0].value().clone())?;
    let replacement_geometry = replace_geometry(
        geometry_store,
        &assets.geometry[0],
        replacement_geometry_data,
    )?;
    let replacement_material = replace_material(
        material_store,
        &assets.materials[2],
        textured_material("textured_color_second"),
    )?;
    let replacement_texture = replace_texture(
        texture_store,
        &assets.textures[1],
        ColorTextureData {
            extent: [1, 1],
            rgba8: vec![255, 0, 0, 255],
        },
    )?;
    if replacement_geometry.id() != assets.geometry[0].id()
        || replacement_material.id() != assets.materials[2].id()
        || replacement_texture.id() != assets.textures[1].id()
        || replacement_geometry.generation() <= assets.geometry[0].generation()
        || replacement_material.generation() <= assets.materials[2].generation()
        || replacement_texture.generation() <= assets.textures[1].generation()
    {
        return Err("S5 replacement did not retain each logical asset identity and advance content generation".into());
    }

    let replacement_assets = ForwardAssets {
        geometry: vec![
            replacement_geometry.clone(),
            assets.geometry[1].clone(),
            assets.geometry[2].clone(),
            assets.geometry[3].clone(),
        ],
        materials: vec![
            assets.materials[0].clone(),
            assets.materials[1].clone(),
            replacement_material.clone(),
            assets.materials[3].clone(),
        ],
        textures: vec![
            assets.textures[0].clone(),
            replacement_texture.clone(),
            assets.textures[2].clone(),
            assets.textures[3].clone(),
        ],
    };
    let mut update = RenderChangeSet::new();
    for (index, id) in ids.iter().copied().enumerate() {
        let mut record = scene.record(id)?.clone();
        record.resources = original_resources(&replacement_assets, index)?;
        update.push(RenderChange::Update(id, record));
    }
    scene.apply(&update)?;

    let new_expected = replacement_expected();
    write_replacement_input(
        output,
        "new",
        scene.snapshot().version(),
        &replacement_assets,
        &new_expected,
    )?;
    let new = crate::block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        &replacement_assets,
    ))?;
    if renderer.pending() != 2 {
        return Err("S5 replacement frames did not retain old and new submissions together".into());
    }
    read_replacement(renderer, &old, output, "old", &old_expected, mask)?;
    read_replacement(renderer, &new, output, "new", &new_expected, mask)?;
    if renderer.pending() != 0 {
        return Err("S5 replacement readbacks did not retire old and new submissions".into());
    }
    Ok(())
}

fn original_resources(
    assets: &ForwardAssets,
    index: usize,
) -> Result<ForwardResources, Box<dyn Error>> {
    let geometry = assets
        .geometry
        .get(index)
        .ok_or("S5 missing geometry asset")?;
    let material = assets
        .materials
        .get(index)
        .ok_or("S5 missing material asset")?;
    let texture = assets
        .textures
        .get(index)
        .ok_or("S5 missing texture asset")?;
    Ok(ForwardResources {
        geometry: SceneAssetRef::from_snapshot(geometry),
        material: SceneAssetRef::from_snapshot(material),
        texture: SceneAssetRef::from_snapshot(texture),
        tint: [1.0; 4],
    })
}

fn replacement_expected() -> Vec<u8> {
    let mut bytes = frame_expected([false, true, true, true]);
    super::paint_triangle(&mut bytes, triangle_pixels()[1], [0, 255, 255, 255]);
    super::paint_triangle(&mut bytes, triangle_pixels()[2], [255, 255, 0, 255]);
    bytes
}

fn clip_geometry(mut data: GeometryData) -> Result<GeometryData, Box<dyn Error>> {
    let (stream_index, position_offset) = data
        .layout
        .streams
        .iter()
        .enumerate()
        .find_map(|(index, stream)| {
            stream
                .attributes
                .iter()
                .find(|attribute| attribute.semantic == "POSITION")
                .map(|attribute| (index, attribute.offset))
        })
        .ok_or("S5 replacement geometry has no POSITION attribute")?;
    let stream = data
        .layout
        .streams
        .get(stream_index)
        .ok_or("S5 replacement geometry stream is missing")?;
    let bytes = data
        .streams
        .get_mut(stream_index)
        .ok_or("S5 replacement geometry payload is missing")?;
    for vertex in data.vertices.clone() {
        let offset = u64::from(vertex)
            .checked_mul(stream.stride)
            .and_then(|offset| offset.checked_add(position_offset))
            .and_then(|offset| offset.checked_add(8))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or("S5 replacement geometry position offset overflow")?;
        let z = bytes
            .get_mut(offset..offset + 4)
            .ok_or("S5 replacement geometry POSITION payload is truncated")?;
        z.copy_from_slice(&2.0_f32.to_le_bytes());
    }
    Ok(data)
}

fn geometry_bytes(data: &GeometryData) -> Result<ResidentBytes, Box<dyn Error>> {
    let lengths: Vec<u64> = data
        .streams
        .iter()
        .map(|stream| u64::try_from(stream.len()))
        .collect::<Result<_, _>>()?;
    let bytes = lengths.iter().try_fold(0_u64, |total, length| {
        total
            .checked_add(*length)
            .ok_or("S5 replacement geometry byte count overflow")
    })?;
    Ok(ResidentBytes::new(bytes))
}

fn replace_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    previous: &fluxel_assets::AssetSnapshot<GeometryAsset, GeometryData>,
    data: GeometryData,
) -> Result<fluxel_assets::AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    match store.request_replacement(previous.handle())? {
        Production::Producer(permit) => Ok(permit.commit(data.clone(), geometry_bytes(&data)?)?),
        Production::Waiter(_) => {
            Err("S5 geometry replacement unexpectedly joined a producer".into())
        }
    }
}

fn replace_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    previous: &fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>,
    data: MaterialDefinition,
) -> Result<fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    match store.request_replacement(previous.handle())? {
        Production::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(0))?),
        Production::Waiter(_) => {
            Err("S5 material replacement unexpectedly joined a producer".into())
        }
    }
}

fn replace_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    previous: &fluxel_assets::AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    data: ColorTextureData,
) -> Result<fluxel_assets::AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let resident_bytes = ResidentBytes::new(u64::try_from(data.rgba8.len())?);
    match store.request_replacement(previous.handle())? {
        Production::Producer(permit) => Ok(permit.commit(data, resident_bytes)?),
        Production::Waiter(_) => {
            Err("S5 texture replacement unexpectedly joined a producer".into())
        }
    }
}

fn write_replacement_input(
    output: &Path,
    name: &str,
    scene_version: u64,
    assets: &ForwardAssets,
    expected: &[u8],
) -> Result<(), Box<dyn Error>> {
    for (stream, bytes) in assets.geometry[0].value().streams.iter().enumerate() {
        fs::write(
            output.join(format!(
                "s5-forward-replacement-{name}-geometry-stream-{stream}.bin"
            )),
            bytes,
        )?;
    }
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-geometry.txt")),
        format!("{:#?}\n", assets.geometry[0].value()),
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-material.txt")),
        format!("{:#?}\n", assets.materials[2].value()),
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-texture.rgba")),
        &assets.textures[1].value().rgba8,
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-input.json")),
        format!(
            concat!(
                "{{\n",
                "  \"sceneVersion\": {},\n",
                "  \"geometry\": {{\"slot\": {}, \"generation\": {}}},\n",
                "  \"material\": {{\"slot\": {}, \"generation\": {}}},\n",
                "  \"texture\": {{\"slot\": {}, \"generation\": {}}}\n",
                "}}\n"
            ),
            scene_version,
            assets.geometry[0].id().slot(),
            assets.geometry[0].generation().get(),
            assets.materials[2].id().slot(),
            assets.materials[2].generation().get(),
            assets.textures[1].id().slot(),
            assets.textures[1].generation().get(),
        ),
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-expected.ppm")),
        ppm(expected)?,
    )?;
    Ok(())
}

fn translated_away() -> Result<RenderTransform, Box<dyn Error>> {
    Ok(RenderTransform::from_columns([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 1.0],
    ])?)
}
