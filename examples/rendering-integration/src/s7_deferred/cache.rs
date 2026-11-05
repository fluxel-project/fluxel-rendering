//! Native completion-owned Deferred cache evidence.
//!
//! Each frame uses the renderer's real public entry point. The analytic oracle
//! supplies colors independently of the stored material source.

use std::{error::Error, fs, path::Path};

use external_renderer::{ChannelMapVariant, ExternalDeferredPipeline};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, DeferredCacheStats, ForwardAssets, ForwardResources,
    FramePipeline, GeometryAsset, GeometryData, RenderBounds, RenderChange, RenderChangeSet,
    RenderRecord, RenderScene, RenderTransform, RenderView, Renderer, SceneAssetRef,
    prepare_forward,
};
use fluxel_rhi::backend::{
    ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
    ResourceBackend, ShaderBackend,
};
use fluxel_shader::ShaderComposer;

use super::{
    asset_json,
    fixtures::{Assets, EXTENT, flat_positions},
    oracle, verify_facts,
};
use crate::block_on;

const COLD: [u8; 4] = [16, 80, 144, 255];
const TINT: [u8; 4] = [16, 80, 72, 255];
const TEXTURE_REPLACED: [u8; 4] = [112, 160, 48, 255];
const SOURCE_REPLACED: [u8; 4] = [224, 80, 48, 255];

/// Runs a cold frame, two settled repeats, a dynamic-only update, and two
/// same-identity asset replacements. A first warm frame may need a distinct
/// plan for its completed incoming state, so only the second settled repeat is
/// required to hit the Graph cache.
pub(super) fn run<B>(
    renderer: &mut Renderer<B>,
    fixtures: &Assets,
    visibility: RenderBounds,
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
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let material = initial_material(&material_store)?;
    let texture = initial_texture(&texture_store)?;
    let geometry = fixtures.flat.clone();
    let mut assets = make_assets(geometry.clone(), material.clone(), texture.clone());
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record(
        &geometry, &material, &texture, [1.0; 4],
    )));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let id = receipt.inserted()[0];
    let view = receipt.inserted_views()[0];
    let pipeline = ExternalDeferredPipeline::new();
    let start = renderer.deferred_cache_stats();
    let start_compilations = renderer.compilations();

    let cold = frame(
        renderer, &pipeline, &scene, view, visibility, &assets, output, "cold", COLD,
    )?;
    assert_progress(cold.before, cold.after, "cold")?;

    let warm = frame(
        renderer,
        &pipeline,
        &scene,
        view,
        visibility,
        &assets,
        output,
        "warm-state",
        COLD,
    )?;
    require_hits(warm.before, warm.after, "warm-state")?;
    if warm.after.resident_upload_bytes != warm.before.resident_upload_bytes {
        return Err("D1 warm-state copied completed static Deferred assets".into());
    }

    let settled = frame(
        renderer,
        &pipeline,
        &scene,
        view,
        visibility,
        &assets,
        output,
        "warm-settled",
        COLD,
    )?;
    require_hits(settled.before, settled.after, "warm-settled")?;
    if settled.after.graph_hits <= settled.before.graph_hits {
        return Err(
            "D1 settled Deferred frame did not reuse its fresh real Graph declaration".into(),
        );
    }
    if settled.after.frame_reuses <= settled.before.frame_reuses {
        return Err("D1 settled Deferred frame did not reuse a completed frame allocation".into());
    }
    if settled.after.resident_upload_bytes != settled.before.resident_upload_bytes {
        return Err("D1 settled Deferred frame copied static resident assets".into());
    }

    update_record(
        &mut scene,
        id,
        &geometry,
        &material,
        &texture,
        [0.5, 1.0, 1.0, 1.0],
    )?;
    let tint = frame(
        renderer, &pipeline, &scene, view, visibility, &assets, output, "tint", TINT,
    )?;
    require_hits(tint.before, tint.after, "dynamic tint")?;
    if tint.after.resident_upload_bytes != tint.before.resident_upload_bytes {
        return Err("D1 dynamic tint copied static Deferred assets".into());
    }

    let new_texture = replace_texture(&texture_store, &texture, [48, 160, 224, 255])?;
    assert_replacement(&texture, &new_texture, "texture")?;
    assets = make_assets(geometry.clone(), material.clone(), new_texture.clone());
    update_record(&mut scene, id, &geometry, &material, &new_texture, [1.0; 4])?;
    let texture_replacement = frame(
        renderer,
        &pipeline,
        &scene,
        view,
        visibility,
        &assets,
        output,
        "texture-replacement",
        TEXTURE_REPLACED,
    )?;
    if texture_replacement.after.texture_hits != texture_replacement.before.texture_hits {
        return Err(
            "D1 replacement texture reused a completed product from its old generation".into(),
        );
    }
    if texture_replacement.after.resident_upload_bytes
        <= texture_replacement.before.resident_upload_bytes
    {
        return Err("D1 replacement texture did not prepare fresh static upload bytes".into());
    }

    let new_material = replace_material(
        &material_store,
        &material,
        ChannelMapVariant::Second.material(),
    )?;
    assert_replacement(&material, &new_material, "material source")?;
    assets = make_assets(geometry.clone(), new_material.clone(), new_texture.clone());
    update_record(
        &mut scene,
        id,
        &geometry,
        &new_material,
        &new_texture,
        [1.0; 4],
    )?;
    let source_replacement = frame(
        renderer,
        &pipeline,
        &scene,
        view,
        visibility,
        &assets,
        output,
        "source-replacement",
        SOURCE_REPLACED,
    )?;
    if source_replacement.after.gbuffer_program_hits
        != source_replacement.before.gbuffer_program_hits
    {
        return Err("D1 replacement material source reused its old G-buffer program".into());
    }

    write_summary(
        output,
        start,
        renderer.deferred_cache_stats(),
        start_compilations,
        renderer.compilations(),
    )
}

struct FrameFacts {
    before: DeferredCacheStats,
    after: DeferredCacheStats,
}

#[allow(clippy::too_many_arguments)]
fn frame<B>(
    renderer: &mut Renderer<B>,
    pipeline: &ExternalDeferredPipeline,
    scene: &RenderScene<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    visibility: RenderBounds,
    assets: &ForwardAssets,
    output: &Path,
    name: &str,
    color: [u8; 4],
) -> Result<FrameFacts, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let snapshot = scene.snapshot();
    let record = snapshot
        .record(
            snapshot
                .record_ids()
                .next()
                .ok_or("D1 cache scene has no record")?,
        )?
        .clone();
    let (expected, mask) =
        oracle::image_for_color(flat_positions(), oracle::Route::Deferred, color);
    let shader = write_shader_route(output, name, pipeline, &snapshot, view, visibility, assets)?;
    write_inputs(
        output, name, &snapshot, &record, assets, &shader, &expected, &mask, color,
    )?;
    let before = renderer.deferred_cache_stats();
    let pending = block_on(renderer.submit_deferred(pipeline, snapshot, view, visibility, assets))?;
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("D1 cache submission did not retain Deferred Graph facts")?;
    verify_facts(facts)?;
    let actual = block_on(renderer.readback(&pending))?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-actual.rgba")),
        &actual,
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-actual.ppm")),
        oracle::ppm(&actual)?,
    )?;
    oracle::verify(
        &actual,
        &expected,
        &mask,
        &format!("D1 cache {name}"),
        oracle::Route::Deferred,
    )?;
    if renderer.pending() != 0 {
        return Err("D1 cache readback retained a completed submission".into());
    }
    let after = renderer.deferred_cache_stats();
    write_facts(output, name, before, after, renderer.compilations(), facts)?;
    Ok(FrameFacts { before, after })
}

struct ShaderEvidence {
    variant_key: String,
    source_units: String,
}

fn write_shader_route(
    output: &Path,
    name: &str,
    pipeline: &ExternalDeferredPipeline,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    visibility: RenderBounds,
    assets: &ForwardAssets,
) -> Result<ShaderEvidence, Box<dyn Error>> {
    let prepared = pipeline.prepare(snapshot, view, visibility)?;
    let draws = prepare_forward(&prepared, assets, &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("D1 cache source evidence requires exactly one prepared draw".into());
    };
    let shader = draw.prepared().shader();
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-composed.wgsl")),
        shader.source(),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-source-map.txt")),
        format!("{:#?}\n", shader.source_map()),
    )?;
    let material = draw.prepared().material();
    Ok(ShaderEvidence {
        variant_key: material.key().as_str().into(),
        source_units: format!("{:#?}", material.contribution().source_units()),
    })
}

fn make_assets(
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
) -> ForwardAssets {
    ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    }
}

fn record(
    geometry: &AssetSnapshot<GeometryAsset, GeometryData>,
    material: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    tint: [f32; 4],
) -> RenderRecord<ForwardResources> {
    RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(geometry),
            material: SceneAssetRef::from_snapshot(material),
            texture: SceneAssetRef::from_snapshot(texture),
            tint,
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-0.62, -0.56, 0.15], [0.58, 0.66, 0.82])
            .expect("fixed finite D1 cache bounds"),
    }
}

fn update_record(
    scene: &mut RenderScene<ForwardResources>,
    id: fluxel_renderer::RenderId,
    geometry: &AssetSnapshot<GeometryAsset, GeometryData>,
    material: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    tint: [f32; 4],
) -> Result<(), Box<dyn Error>> {
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Update(
        id,
        record(geometry, material, texture, tint),
    ));
    scene.apply(&changes)?;
    Ok(())
}

fn initial_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => {
            Ok(producer.commit(ChannelMapVariant::First.material(), ResidentBytes::new(0))?)
        }
        _ => Err("D1 cache initial material producer missing".into()),
    }
}

fn initial_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => {
            Ok(producer.commit(texture([144, 80, 32, 255]), ResidentBytes::new(4))?)
        }
        _ => Err("D1 cache initial texture producer missing".into()),
    }
}

fn replace_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    previous: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    value: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    match store.request_replacement(previous.handle())? {
        Production::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(0))?),
        Production::Waiter(_) => Err("D1 cache material replacement waiter".into()),
    }
}

fn replace_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    previous: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    rgba8: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    match store.request_replacement(previous.handle())? {
        Production::Producer(producer) => {
            Ok(producer.commit(texture(rgba8), ResidentBytes::new(4))?)
        }
        Production::Waiter(_) => Err("D1 cache texture replacement waiter".into()),
    }
}

fn texture(rgba8: [u8; 4]) -> ColorTextureData {
    ColorTextureData {
        extent: [1, 1],
        rgba8: rgba8.to_vec(),
    }
}

fn assert_replacement<K, V>(
    old: &AssetSnapshot<K, V>,
    new: &AssetSnapshot<K, V>,
    label: &str,
) -> Result<(), Box<dyn Error>>
where
    K: fluxel_assets::AssetKind,
{
    if old.id() != new.id() || old.generation() >= new.generation() {
        return Err(format!(
            "D1 {label} replacement did not preserve identity and advance generation"
        )
        .into());
    }
    Ok(())
}

fn require_hits(
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if after.geometry_hits <= before.geometry_hits
        || after.texture_hits <= before.texture_hits
        || after.gbuffer_program_hits <= before.gbuffer_program_hits
        || after.lighting_program_hits <= before.lighting_program_hits
    {
        return Err(format!(
            "D1 {label} did not select all completed resident/program cache products"
        )
        .into());
    }
    Ok(())
}

fn assert_progress(
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if after.buffer_allocations <= before.buffer_allocations
        || after.texture_allocations <= before.texture_allocations
    {
        return Err(format!("D1 {label} did not assemble a Deferred frame").into());
    }
    Ok(())
}

fn write_inputs(
    output: &Path,
    name: &str,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    assets: &ForwardAssets,
    shader: &ShaderEvidence,
    expected: &[u8],
    mask: &[u8],
    color: [u8; 4],
) -> Result<(), Box<dyn Error>> {
    let geometry = assets
        .geometry
        .first()
        .ok_or("D1 cache input lacks submitted geometry")?;
    let material = assets
        .materials
        .first()
        .ok_or("D1 cache input lacks submitted material")?;
    let texture = assets
        .textures
        .first()
        .ok_or("D1 cache input lacks submitted texture")?;
    let stream = geometry
        .value()
        .streams
        .first()
        .ok_or("D1 cache geometry lacks its submitted stream")?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"route\":\"deferred\",",
                "\"factorPolicy\":\"analytic geometric normal/depth, independent of WGSL\",",
                "\"expectedAlbedo\":{:?},\"tint\":{:?},\"geometryStreamBytes\":{:?},",
                "\"textureRgba8\":{:?},\"materialVariantKey\":{},\"materialSourceUnits\":{},",
                "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            color,
            record.resources.tint,
            stream,
            texture.value().rgba8,
            json_string(&shader.variant_key),
            json_string(&shader.source_units),
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture),
        ),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-material.txt")),
        format!("{:#?}\n", material.value()),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-stable-mask.bin")),
        mask,
    )?;
    Ok(())
}

fn json_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn write_facts(
    output: &Path,
    name: &str,
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    compilations: usize,
    facts: fluxel_renderer::DeferredProbeFacts,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-facts.json")),
        format!(
            "{{\"schemaVersion\":1,\"before\":{},\"after\":{},\"compilations\":{},\"graphPasses\":{}}}\n",
            stats_json(before),
            stats_json(after),
            compilations,
            facts.passes,
        ),
    )?;
    Ok(())
}

fn write_summary(
    output: &Path,
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    before_compilations: usize,
    after_compilations: usize,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("s7-deferred-cache-facts.json"),
        format!(
            "{{\"schemaVersion\":1,\"start\":{},\"end\":{},\"compilationsBefore\":{},\"compilationsAfter\":{},\"frames\":[\"cold\",\"warm-state\",\"warm-settled\",\"tint\",\"texture-replacement\",\"source-replacement\"]}}\n",
            stats_json(before),
            stats_json(after),
            before_compilations,
            after_compilations,
        ),
    )?;
    Ok(())
}

fn stats_json(stats: DeferredCacheStats) -> String {
    format!(
        "{{\"geometryHits\":{},\"textureHits\":{},\"gbufferProgramHits\":{},\"lightingProgramHits\":{},\"graphHits\":{},\"frameReuses\":{},\"bufferAllocations\":{},\"textureAllocations\":{},\"residentUploadBytes\":{},\"frameUploadBytes\":{}}}",
        stats.geometry_hits,
        stats.texture_hits,
        stats.gbuffer_program_hits,
        stats.lighting_program_hits,
        stats.graph_hits,
        stats.frame_reuses,
        stats.buffer_allocations,
        stats.texture_allocations,
        stats.resident_upload_bytes,
        stats.frame_upload_bytes,
    )
}
