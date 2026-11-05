//! Native Deferred two-device cache-isolation workload.
//!
//! The exact immutable CPU scene is submitted on two concurrently live RHI
//! devices.  Completion publishes a product only to its owning Renderer.

mod evidence;

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, Renderer,
    RendererError, SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, TextureFormat, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::{block_on, json_string};

const EXTENT: [u32; 2] = [96, 96];
const FIRST: [u8; 4] = [144, 80, 32, 255];
const REPLACED: [u8; 4] = [48, 160, 224, 255];

pub(crate) fn run<B>(
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
        return Err("S7 Deferred isolation requires distinct DeviceId values".into());
    }
    fs::create_dir_all(output)?;
    let inputs = Inputs::new()?;
    inputs.write_shared(output)?;
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(inputs.record(&inputs.texture)));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let record = receipt.inserted()[0];
    let view = receipt.inserted_views()[0];
    let bounds = RenderBounds::new([-1.0; 3], [1.0; 3])?;
    let first_snapshot = scene.snapshot();
    let first_assets = inputs.assets(inputs.texture.clone());
    let pipeline = ExternalDeferredPipeline::new();
    let mut old = Renderer::new(old_device.clone())?;
    let mut new = Renderer::new(new_device.clone())?;

    let old_cold = frame(
        &mut old,
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
        FIRST,
        "old-cold",
        output,
        true,
    )?;
    require_cold(old_cold, "old cold")?;
    let old_warm = frame(
        &mut old,
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
        FIRST,
        "old-warm",
        output,
        true,
    )?;
    require_warm(old_warm, "old warm")?;
    let new_cold = frame(
        &mut new,
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
        FIRST,
        "new-cold",
        output,
        true,
    )?;
    require_cold(new_cold, "new cold")?;
    let new_warm = frame(
        &mut new,
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
        FIRST,
        "new-warm",
        output,
        true,
    )?;
    require_warm(new_warm, "new warm")?;

    let old_return = frame(
        &mut old,
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
        FIRST,
        "old-return",
        output,
        true,
    )?;
    require_warm(old_return, "old return")?;
    require_graph_hit(old_return, "old return")?;

    let pending = block_on(old.submit_deferred(
        &pipeline,
        first_snapshot.clone(),
        view,
        bounds,
        &first_assets,
    ))?;
    let owners = old
        .deferred_probe_diagnostic(&pending)?
        .ok_or("old pending submission lacks owner diagnostic")?;
    fs::write(output.join("old-pending-owners.txt"), owners)?;
    let foreign = matches!(
        block_on(new.readback(&pending)),
        Err(RendererError::ForeignSubmission)
    );
    if !foreign || old.pending() != 1 || new.pending() != 0 {
        return Err("S7 Deferred cross-device submission was not fail-closed".into());
    }
    fs::write(
        output.join("foreign-submission-rejected-input.json"),
        "{\"schemaVersion\":1,\"attempt\":\"new-renderer-readback-old-submission\"}\n",
    )?;
    fs::write(
        output.join("foreign-submission-rejected.json"),
        "{\"schemaVersion\":1,\"error\":\"ForeignSubmission\",\"oldPending\":1,\"newPending\":0}\n",
    )?;
    fs::write(
        output.join("foreign-submission-rejected-facts.json"),
        "{\"schemaVersion\":1,\"submission\":\"rejected\",\"completion\":\"not-consumed\",\"oldOwnerPending\":true}\n",
    )?;

    let replacement = inputs.replace_texture()?;
    let replacement_assets = inputs.assets(replacement.clone());
    let mut update = RenderChangeSet::new();
    update.push(RenderChange::Update(record, inputs.record(&replacement)));
    scene.apply(&update)?;
    let replacement_snapshot = scene.snapshot();
    let new_replacement = frame(
        &mut new,
        &pipeline,
        replacement_snapshot.clone(),
        view,
        bounds,
        &replacement_assets,
        REPLACED,
        "new-replacement",
        output,
        false,
    )?;
    require_replacement(new_replacement, "new replacement")?;
    let old_pending_actual = block_on(old.readback(&pending))?;
    fs::write(
        output.join("old-pending-completed.rgba"),
        &old_pending_actual,
    )?;
    evidence::verify(&old_pending_actual, FIRST)?;
    if old.pending() != 0 {
        return Err("old pending owner did not retire after its completion".into());
    }
    let old_replacement = frame(
        &mut old,
        &pipeline,
        replacement_snapshot,
        view,
        bounds,
        &replacement_assets,
        REPLACED,
        "old-replacement",
        output,
        false,
    )?;
    require_replacement(old_replacement, "old replacement")?;

    let facts = format!(
        concat!(
            "{{\"schemaVersion\":1,\"oldDeviceId\":{},\"newDeviceId\":{},",
            "\"sameCpuSceneAndAssets\":true,\"oldPendingRetired\":true,",
            "\"oldPendingInput\":\"old-cold-input.json\",",
            "\"foreignSubmissionRejected\":true,\"oldFinalStats\":{},\"newFinalStats\":{}}}\n"
        ),
        json_string(&format!("{:?}", old_device.id())),
        json_string(&format!("{:?}", new_device.id())),
        evidence::stats_json(old.deferred_cache_stats()),
        evidence::stats_json(new.deferred_cache_stats()),
    );
    fs::write(output.join("device-isolation-facts.json"), facts)?;
    drop(old);
    drop(new);
    evidence::write_manifest(output)
}

#[allow(clippy::too_many_arguments)]
fn frame<B>(
    renderer: &mut Renderer<B>,
    pipeline: &ExternalDeferredPipeline,
    snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    bounds: RenderBounds,
    assets: &ForwardAssets,
    color: [u8; 4],
    name: &str,
    output: &Path,
    require_cold: bool,
) -> Result<
    (
        fluxel_renderer::DeferredCacheStats,
        fluxel_renderer::DeferredCacheStats,
    ),
    Box<dyn Error>,
>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let (expected, mask) = evidence::expected(color);
    fs::write(
        output.join(format!("{name}-input.json")),
        input_json(&snapshot, assets)?,
    )?;
    fs::write(output.join(format!("{name}-expected.rgba")), expected)?;
    fs::write(output.join(format!("{name}-mask.bin")), mask)?;
    let before = renderer.deferred_cache_stats();
    let submission = block_on(renderer.submit_deferred(pipeline, snapshot, view, bounds, assets))?;
    let probe = renderer
        .deferred_probe_facts()
        .ok_or("Deferred submission lacks Graph facts")?;
    if probe.passes != 4
        || probe.gbuffer_formats
            != [
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Depth32Float,
            ]
    {
        return Err("Deferred submission violated the G-buffer Graph contract".into());
    }
    let owners = renderer
        .deferred_probe_diagnostic(&submission)?
        .ok_or("Deferred submission lacks owner facts")?;
    fs::write(output.join(format!("{name}-owners.txt")), owners)?;
    let actual = block_on(renderer.readback(&submission))?;
    fs::write(output.join(format!("{name}-actual.rgba")), &actual)?;
    evidence::verify(&actual, color)?;
    let after = renderer.deferred_cache_stats();
    if renderer.pending() != 0 {
        return Err("completed Deferred frame remained pending".into());
    }
    if require_cold
        && after.resident_upload_bytes == before.resident_upload_bytes
        && before.resident_upload_bytes == 0
    {
        return Err("cold Deferred frame did not prepare resident products".into());
    }
    fs::write(
        output.join(format!("{name}-facts.json")),
        format!(
            "{{\"schemaVersion\":1,\"plan\":{},\"passes\":{},\"deviceStatsBefore\":{},\"deviceStatsAfter\":{},\"pendingAfterReadback\":0}}\n",
            json_string(&format!("{:?}", probe.plan)),
            probe.passes,
            evidence::stats_json(before),
            evidence::stats_json(after)
        ),
    )?;
    Ok((before, after))
}

fn require_warm(
    (before, after): (
        fluxel_renderer::DeferredCacheStats,
        fluxel_renderer::DeferredCacheStats,
    ),
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if after.geometry_hits <= before.geometry_hits
        || after.texture_hits <= before.texture_hits
        || after.gbuffer_program_hits <= before.gbuffer_program_hits
        || after.resident_upload_bytes != before.resident_upload_bytes
    {
        return Err(format!("{name} did not reuse device-local Deferred products").into());
    }
    Ok(())
}

fn require_graph_hit(
    (before, after): (
        fluxel_renderer::DeferredCacheStats,
        fluxel_renderer::DeferredCacheStats,
    ),
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if after.graph_hits <= before.graph_hits {
        return Err(format!("{name} did not reuse its declared Deferred Graph plan").into());
    }
    Ok(())
}

fn require_cold(
    (before, after): (
        fluxel_renderer::DeferredCacheStats,
        fluxel_renderer::DeferredCacheStats,
    ),
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if before.geometry_hits != after.geometry_hits
        || before.texture_hits != after.texture_hits
        || before.gbuffer_program_hits != after.gbuffer_program_hits
        || after.resident_upload_bytes <= before.resident_upload_bytes
    {
        return Err(format!("{name} adopted another device's Deferred cache").into());
    }
    Ok(())
}

fn require_replacement(
    (before, after): (
        fluxel_renderer::DeferredCacheStats,
        fluxel_renderer::DeferredCacheStats,
    ),
    name: &str,
) -> Result<(), Box<dyn Error>> {
    if after.texture_hits != before.texture_hits
        || after.resident_upload_bytes <= before.resident_upload_bytes
    {
        return Err(format!("{name} reused a stale texture generation").into());
    }
    Ok(())
}

struct Inputs {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    textures: AssetStore<ColorTextureAsset, ColorTextureData, String>,
}

impl Inputs {
    fn new() -> Result<Self, Box<dyn Error>> {
        let geometries = AssetStore::new();
        let materials = AssetStore::new();
        let textures = AssetStore::new();
        Ok(Self {
            geometry: commit_geometry(&geometries)?,
            material: commit_material(&materials)?,
            texture: commit_texture(&textures, FIRST)?,
            textures,
        })
    }
    fn assets(&self, texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: vec![self.material.clone()],
            textures: vec![texture],
        }
    }
    fn record(
        &self,
        texture: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    ) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(texture),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-0.62, -0.56, 0.15], [0.58, 0.66, 0.82])
                .expect("fixture bounds"),
        }
    }
    fn replace_texture(
        &self,
    ) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
        match self.textures.request_replacement(self.texture.handle())? {
            Production::Producer(p) => Ok(p.commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: REPLACED.to_vec(),
                },
                ResidentBytes::new(4),
            )?),
            Production::Waiter(_) => Err("texture replacement unexpectedly waited".into()),
        }
    }
    fn write_shared(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            output.join("source-position.bin"),
            &self.geometry.value().streams[0],
        )?;
        fs::write(
            output.join("source-geometry.txt"),
            format!("{:#?}\n", self.geometry.value()),
        )?;
        fs::write(
            output.join("source-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        fs::write(
            output.join("source-first.rgba"),
            &self.texture.value().rgba8,
        )?;
        fs::write(output.join("source-replaced.rgba"), REPLACED)?;
        fs::write(
            output.join("source-identities.json"),
            format!(
                "{{\"geometry\":{},\"material\":{},\"texture\":{}}}\n",
                identity(&self.geometry),
                identity(&self.material),
                identity(&self.texture)
            ),
        )?;
        Ok(())
    }
}

fn commit_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let value = GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                }],
            }],
        },
        streams: vec![
            [-0.62f32, -0.56, 0.50, 0.58, -0.42, 0.50, -0.12, 0.66, 0.50]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        ],
        vertices: 0..3,
    };
    let h = store.create()?;
    match store.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(36))?),
        _ => Err("geometry producer missing".into()),
    }
}
fn commit_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let value = MaterialDefinition::new(vec![MaterialFragment { name: "device_isolation_albedo".into(), inputs: vec![], output: MaterialType::Vec4, calls: vec![], wgsl: "fn device_isolation_albedo() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into() }], "device_isolation_albedo", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }], vec![MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } }, MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } }], BTreeMap::new(), SurfaceOutput::Color);
    let h = store.create()?;
    match store.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(0))?),
        _ => Err("material producer missing".into()),
    }
}
fn commit_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    rgba8: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let h = store.create()?;
    match store.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                extent: [1, 1],
                rgba8: rgba8.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("texture producer missing".into()),
    }
}

fn identity<K: fluxel_assets::AssetKind, V>(value: &AssetSnapshot<K, V>) -> String {
    let id = value.id();
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        id.slot(),
        id.slot_generation(),
        value.generation().get()
    )
}
fn input_json(
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    assets: &ForwardAssets,
) -> Result<String, Box<dyn Error>> {
    let id = snapshot
        .record_ids()
        .next()
        .ok_or("Deferred input snapshot has no record")?;
    let record = snapshot.record(id)?;
    let geometry = assets
        .geometry
        .iter()
        .find(|value| {
            value.id() == record.resources.geometry.id()
                && value.generation() == record.resources.geometry.generation()
        })
        .ok_or("Deferred input geometry is absent from supplied assets")?;
    let texture = assets
        .textures
        .iter()
        .find(|value| {
            value.id() == record.resources.texture.id()
                && value.generation() == record.resources.texture.generation()
        })
        .ok_or("Deferred input texture is absent from supplied assets")?;
    Ok(format!(
        concat!(
            "{{\"schemaVersion\":1,\"sceneVersion\":{},\"recordId\":{},",
            "\"assetRefs\":{{\"geometry\":{},\"material\":{},\"texture\":{}}},",
            "\"geometryBytes\":{:?},\"textureRgba8\":{:?},",
            "\"tint\":{:?},\"transformColumns\":{:?},\"extent\":{:?}}}\n"
        ),
        snapshot.version(),
        json_string(&format!("{id:?}")),
        reference_identity(record.resources.geometry),
        reference_identity(record.resources.material),
        reference_identity(record.resources.texture),
        geometry.value().streams[0],
        texture.value().rgba8,
        record.resources.tint,
        record.transform.columns(),
        EXTENT,
    ))
}

fn reference_identity<K: fluxel_assets::AssetKind>(value: SceneAssetRef<K>) -> String {
    let id = value.id();
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        id.slot(),
        id.slot_generation(),
        value.generation().get()
    )
}
