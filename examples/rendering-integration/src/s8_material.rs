//! Material-heavy Forward workload for measured S8 strategies.
use external_renderer::ExternalOpaquePipeline;
use fluxel_assets::{AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCacheStats, ForwardResources,
    GeometryAsset, GeometryData, RenderBounds, RenderChange, RenderChangeSet, RenderId,
    RenderRecord, RenderScene, RenderTransform, RenderView, RenderViewId, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use std::{error::Error, fs, path::Path, time::Instant};
mod lifecycle;
mod s8_material_fixture;
pub(crate) use lifecycle::run_lifecycle;
use s8_material_fixture::{
    geometry_snapshot, material, material_snapshot, texture_snapshot, tiled_triangle,
};

const DRAWS: usize = 10_000;
const TEXTURES: usize = 1_000;
const SMOKE_DRAWS: usize = 100;
const SMOKE_TEXTURES: usize = 10;
const EXTENT: [u32; 2] = [512, 512];
const VARIANTS: usize = 4;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Scenario {
    Cold,
    Warm,
    Replacement10Pct,
}
impl Scenario {
    pub(crate) fn parse(v: &str) -> Result<Self, Box<dyn Error>> {
        match v {
            "cold" => Ok(Self::Cold),
            "warm" => Ok(Self::Warm),
            "replacement-10pct" => Ok(Self::Replacement10Pct),
            _ => Err("--s8-scenario must be cold, warm, or replacement-10pct".into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
            Self::Replacement10Pct => "replacement-10pct",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Config {
    pub scenario: Scenario,
    pub smoke: bool,
}
impl Config {
    pub(crate) fn new(scenario: Scenario, smoke: bool) -> Self {
        Self { scenario, smoke }
    }
    fn dimensions(&self) -> (usize, usize) {
        if self.smoke {
            (SMOKE_DRAWS, SMOKE_TEXTURES)
        } else {
            (DRAWS, TEXTURES)
        }
    }
}

pub(crate) fn run<B>(device: Device<B>, output: &Path, config: Config) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let (draws, textures) = config.dimensions();
    fs::create_dir_all(output)?;
    let fixture = Fixture::new(draws, textures)?;
    fixture.write_inputs(output, &config)?;
    let mut scene = RenderScene::new()?;
    let receipt = scene.apply(&fixture.initial_changes())?;
    let view = receipt.inserted_views()[0];
    let records = receipt.inserted().to_vec();
    let mut renderer = Renderer::new(device)?;
    let pipeline = ExternalOpaquePipeline::new();
    let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
    let facts = match config.scenario {
        Scenario::Cold => frame(
            &mut renderer,
            &pipeline,
            scene.snapshot(),
            view,
            bounds,
            &fixture.assets(),
            output,
            "cold",
        )?,
        Scenario::Warm => {
            let prime = frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &fixture.assets(),
                output,
                "warm-prime",
            )?;
            let _ = prime;
            let settle = frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &fixture.assets(),
                output,
                "warm-settle",
            )?;
            let _ = settle;
            frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &fixture.assets(),
                output,
                "warm",
            )?
        }
        Scenario::Replacement10Pct => {
            let prime = frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &fixture.assets(),
                output,
                "replacement-prime",
            )?;
            let _ = prime;
            let settle = frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &fixture.assets(),
                output,
                "replacement-settle",
            )?;
            let _ = settle;
            let assets = fixture.assets_with_textures(fixture.replace_tenth()?);
            scene.apply(&fixture.replacement_changes(&records, &assets.textures))?;
            fixture.write_replacement_input(output, &assets.textures)?;
            frame(
                &mut renderer,
                &pipeline,
                scene.snapshot(),
                view,
                bounds,
                &assets,
                output,
                "replacement-10pct",
            )?
        }
    };
    if renderer.pending() != 0 {
        return Err("S8 readback did not retire the selected frame".into());
    }
    verify(&config, &facts, draws, textures)?;
    write_artifacts(output, &config, &fixture, facts)
}
fn verify(c: &Config, f: &Facts, draws: usize, textures: usize) -> Result<(), Box<dyn Error>> {
    if f.draws != u64::try_from(draws)? || f.visible != u64::try_from(draws)? {
        return Err("S8 did not consume every tiled visible draw".into());
    }
    match c.scenario {
        Scenario::Cold if f.resident == 0 => Err("S8 cold did not upload residents".into()),
        Scenario::Warm if f.texture_hits != u64::try_from(textures)? => {
            Err("S8 warm did not hit every unique texture resident".into())
        }
        Scenario::Replacement10Pct => {
            let replaced = u64::try_from(textures.div_ceil(10))?;
            if f.resident == 0
                || f.texture_hits != u64::try_from(textures)?.saturating_sub(replaced)
            {
                Err("S8 replacement did not distinguish generation-refreshed textures".into())
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}
struct Facts {
    submit: u128,
    prep: u128,
    graph: u128,
    buffers: u64,
    buffer_bytes: u64,
    textures: u64,
    peak: u64,
    resident: u64,
    frame: u64,
    geometry_hits: u64,
    texture_hits: u64,
    reuse: u64,
    plans: u64,
    visible: u64,
    batches: u64,
    draws: u64,
}
fn frame<B>(
    r: &mut Renderer<B>,
    p: &ExternalOpaquePipeline,
    s: fluxel_renderer::FrameSnapshot<ForwardResources>,
    v: RenderViewId,
    b: RenderBounds,
    a: &ForwardAssets,
    output: &Path,
    name: &str,
) -> Result<Facts, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let before = r.forward_cache_stats();
    let before_plans = r.compilations();
    let started = Instant::now();
    let submission = crate::block_on(r.submit_forward(p, s, v, b, a))?;
    let submit = started.elapsed().as_nanos();
    let raw = crate::block_on(r.readback(&submission))?;
    fs::write(output.join(format!("s8-material-{name}-raw.rgba")), &raw)?;
    let after = r.forward_cache_stats();
    Ok(Facts::delta(
        submit,
        before,
        after,
        before_plans,
        r.compilations(),
    ))
}
impl Facts {
    fn delta(
        submit: u128,
        b: ForwardCacheStats,
        a: ForwardCacheStats,
        bp: usize,
        ap: usize,
    ) -> Self {
        Self {
            submit,
            prep: a
                .preparation_elapsed_ns
                .saturating_sub(b.preparation_elapsed_ns),
            graph: a
                .graph_compile_elapsed_ns
                .saturating_sub(b.graph_compile_elapsed_ns),
            buffers: a
                .accepted_buffer_allocations
                .saturating_sub(b.accepted_buffer_allocations),
            buffer_bytes: a
                .accepted_buffer_allocation_bytes
                .saturating_sub(b.accepted_buffer_allocation_bytes),
            textures: a
                .accepted_texture_allocations
                .saturating_sub(b.accepted_texture_allocations),
            peak: a.pending_peak.max(b.pending_peak),
            resident: a.resident_copy_bytes.saturating_sub(b.resident_copy_bytes),
            frame: a.frame_upload_bytes.saturating_sub(b.frame_upload_bytes),
            geometry_hits: a.geometry_hits.saturating_sub(b.geometry_hits),
            texture_hits: a.texture_hits.saturating_sub(b.texture_hits),
            reuse: a.frame_reuses.saturating_sub(b.frame_reuses),
            plans: u64::try_from(ap.saturating_sub(bp)).unwrap_or(u64::MAX),
            visible: a.opaque_candidates.saturating_sub(b.opaque_candidates),
            batches: a.opaque_batches.saturating_sub(b.opaque_batches),
            draws: a.opaque_draws.saturating_sub(b.opaque_draws),
        }
    }
}

struct Fixture {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    materials: Vec<AssetSnapshot<MaterialAsset, MaterialDefinition>>,
    store: AssetStore<ColorTextureAsset, ColorTextureData, String>,
    initial: Vec<AssetSnapshot<ColorTextureAsset, ColorTextureData>>,
    draws: usize,
}
impl Fixture {
    fn new(draws: usize, textures: usize) -> Result<Self, Box<dyn Error>> {
        if draws == 0 || textures == 0 || draws < textures {
            return Err("S8 requires at least one draw per texture".into());
        };
        let gs = AssetStore::new();
        let ms = AssetStore::new();
        let ts = AssetStore::new();
        let geometry = geometry_snapshot(&gs, tiled_triangle())?;
        let materials = (0..VARIANTS)
            .map(|n| material_snapshot(&ms, material(n)))
            .collect::<Result<Vec<_>, _>>()?;
        let initial = (0..textures)
            .map(|n| texture_snapshot(&ts, rgba(n, false)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            geometry,
            materials,
            store: ts,
            initial,
            draws,
        })
    }
    fn assets(&self) -> ForwardAssets {
        self.assets_with_textures(self.initial.clone())
    }
    fn assets_with_textures(
        &self,
        textures: Vec<AssetSnapshot<ColorTextureAsset, ColorTextureData>>,
    ) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: self.materials.clone(),
            textures,
        }
    }
    fn initial_changes(&self) -> RenderChangeSet<ForwardResources> {
        self.changes(&self.initial)
    }
    fn changes(
        &self,
        t: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>],
    ) -> RenderChangeSet<ForwardResources> {
        let mut c: RenderChangeSet<_> = (0..self.draws)
            .map(|n| RenderChange::Insert(self.record(n, t)))
            .collect();
        c.push(RenderChange::InsertView(
            RenderView::new(RenderTransform::identity(), EXTENT).expect("fixed extent"),
        ));
        c
    }
    fn replacement_changes(
        &self,
        ids: &[RenderId],
        t: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>],
    ) -> RenderChangeSet<ForwardResources> {
        ids.iter()
            .enumerate()
            .map(|(n, id)| RenderChange::Update(*id, self.record(n, t)))
            .collect()
    }
    fn record(
        &self,
        n: usize,
        t: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>],
    ) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.materials[n % VARIANTS]),
                texture: SceneAssetRef::from_snapshot(&t[n % t.len()]),
                tint: tint(n),
            },
            transform: tile_transform(n, self.draws),
            bounds: RenderBounds::new([-1.; 3], [1.; 3]).expect("fixed bounds"),
        }
    }
    fn replace_tenth(
        &self,
    ) -> Result<Vec<AssetSnapshot<ColorTextureAsset, ColorTextureData>>, Box<dyn Error>> {
        self.initial
            .iter()
            .enumerate()
            .map(|(n, old)| {
                if n % 10 != 0 {
                    Ok(old.clone())
                } else {
                    match self.store.request_replacement(old.handle())? {
                        Production::Producer(p) => Ok(p.commit(
                            ColorTextureData {
                                extent: [1, 1],
                                rgba8: rgba(n, true).to_vec(),
                            },
                            ResidentBytes::new(4),
                        )?),
                        Production::Waiter(_) => Err("S8 replacement unexpectedly waited".into()),
                    }
                }
            })
            .collect()
    }
    fn write_inputs(&self, out: &Path, c: &Config) -> Result<(), Box<dyn Error>> {
        fs::write(
            out.join("s8-material-geometry.bin"),
            &self.geometry.value().streams[0],
        )?;
        for (index, material) in self.materials.iter().enumerate() {
            fs::write(
                out.join(format!("s8-material-variant-{index}.txt")),
                format!("{:#?}\n", material.value()),
            )?;
        }
        let one: Vec<u8> = self
            .initial
            .iter()
            .flat_map(|x| x.value().rgba8.clone())
            .collect();
        fs::write(out.join("s8-material-textures-generation-1.rgba"), one)?;
        let two = self.replacement_bytes();
        fs::write(
            out.join("s8-material-textures-generation-2-replaced.rgba"),
            &two,
        )?;
        fs::write(
            out.join("s8-material-input.json"),
            input_json(self, c, two.len() / 4),
        )?;
        Ok(())
    }
    fn replacement_bytes(&self) -> Vec<u8> {
        (0..self.initial.len())
            .filter(|n| n % 10 == 0)
            .flat_map(|n| rgba(n, true))
            .collect()
    }
    fn write_replacement_input(
        &self,
        out: &Path,
        textures: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>],
    ) -> Result<(), Box<dyn Error>> {
        let identities = textures
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 10 == 0)
            .map(|(index, snapshot)| {
                format!(
                    "{{\"index\":{index},\"id\":{},\"generation\":{},\"rgba\":{:?}}}",
                    json(&format!("{:?}", snapshot.id())),
                    json(&format!("{:?}", snapshot.generation())),
                    snapshot.value().rgba8,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        fs::write(
            out.join("s8-material-replacement-input.json"),
            format!("{{\"schemaVersion\":1,\"replacedTextures\":[{identities}]}}\n"),
        )?;
        Ok(())
    }
}
fn tile_transform(n: usize, draws: usize) -> RenderTransform {
    let cols = (draws as f64).sqrt().ceil() as usize;
    let rows = draws.div_ceil(cols);
    let x = n % cols;
    let y = n / cols;
    let sx = 1. / cols as f32;
    let sy = 1. / rows as f32;
    let tx = -1. + (2. * x as f32 + 1.) / cols as f32;
    let ty = 1. - (2. * y as f32 + 1.) / rows as f32;
    RenderTransform::from_columns([
        [sx, 0., 0., 0.],
        [0., sy, 0., 0.],
        [0., 0., 1., 0.],
        [tx, ty, 0., 1.],
    ])
    .expect("finite tiles")
}
fn tint(n: usize) -> [f32; 4] {
    [
        [1., 1., 1., 1.],
        [1., 0., 1., 1.],
        [0., 1., 1., 1.],
        [1., 1., 0., 1.],
    ][n % VARIANTS]
}
fn rgba(n: usize, replacement: bool) -> [u8; 4] {
    let x = n as u8;
    if replacement {
        [
            x.wrapping_mul(37).wrapping_add(17),
            x.wrapping_mul(73).wrapping_add(31),
            x.wrapping_mul(109).wrapping_add(53),
            255,
        ]
    } else {
        [
            x.wrapping_mul(29),
            x.wrapping_mul(47),
            x.wrapping_mul(83),
            255,
        ]
    }
}
fn write_artifacts(out: &Path, c: &Config, x: &Fixture, f: Facts) -> Result<(), Box<dyn Error>> {
    let name = format!("s8-material-{}-raw.rgba", c.scenario.name());
    fs::write(
        out.join(format!("s8-material-{}-facts.json", c.scenario.name())),
        facts_json(&f),
    )?;
    fs::write(
        out.join("s8-material-baseline.json"),
        summary_json(c, x, &f, &name),
    )?;
    Ok(())
}
fn input_json(x: &Fixture, c: &Config, replaced: usize) -> String {
    let textures = x
        .initial
        .iter()
        .enumerate()
        .map(|(n, s)| {
            format!(
                "{{\"index\":{n},\"id\":{},\"generation\":{},\"rgbaOffset\":{}}}",
                json(&format!("{:?}", s.id())),
                json(&format!("{:?}", s.generation())),
                n * 4
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let draws=(0..x.draws).map(|n|format!("{{\"index\":{n},\"geometry\":{},\"materialVariant\":{},\"textureIndex\":{},\"tint\":{:?},\"transformColumns\":{:?}}}",json(&format!("{:?}",x.geometry.id())),n%VARIANTS,n%x.initial.len(),tint(n),tile_transform(n,x.draws).columns())).collect::<Vec<_>>().join(",");
    format!(
        "{{\"schemaVersion\":1,\"scenario\":{},\"draws\":{},\"textures\":{},\"extent\":[512,512],\"sharedGeometry\":{{\"id\":{},\"generation\":{},\"bytes\":\"s8-material-geometry.bin\"}},\"textureGeneration1\":{{\"bytes\":\"s8-material-textures-generation-1.rgba\",\"textures\":[{textures}]}},\"textureGeneration2Replacement\":{{\"bytes\":\"s8-material-textures-generation-2-replaced.rgba\",\"count\":{replaced}}},\"drawRecords\":[{draws}],\"allTexturesReferenced\":true,\"smoke\":{}}}\n",
        json(c.scenario.name()),
        x.draws,
        x.initial.len(),
        json(&format!("{:?}", x.geometry.id())),
        json(&format!("{:?}", x.geometry.generation())),
        c.smoke
    )
}
fn facts_json(f: &Facts) -> String {
    format!("{{\"schemaVersion\":1,{}}}\n", metrics(f))
}
fn summary_json(c: &Config, x: &Fixture, f: &Facts, raw: &str) -> String {
    format!(
        "{{\"schemaVersion\":1,\"workload\":\"s8-material-baseline\",\"role\":\"representative\",\"scenario\":{},\"primaryMetric\":\"cpu-preparation-submit\",\"gpuTiming\":\"unobserved\",\"requestedDraws\":{},\"textures\":{},\"materialVariants\":{},\"sharedGeometryAssets\":1,\"allTexturesReferenced\":true,\"smoke\":{},\"rawReadback\":{},{} }}\n",
        json(c.scenario.name()),
        x.draws,
        x.initial.len(),
        x.materials.len(),
        c.smoke,
        json(raw),
        metrics(f)
    )
}
fn metrics(f: &Facts) -> String {
    format!(
        "\"submitElapsedNs\":{},\"preparationElapsedNs\":{},\"graphCompileElapsedNs\":{},\"acceptedBufferAllocations\":{},\"acceptedBufferAllocationBytes\":{},\"acceptedTextureAllocations\":{},\"pendingPeak\":{},\"residentStaticUploadBytes\":{},\"frameUploadBytes\":{},\"geometryCacheHits\":{},\"textureCacheHits\":{},\"frameAllocationReuses\":{},\"planCompilations\":{},\"visibleCandidates\":{},\"batches\":{},\"draws\":{}",
        f.submit,
        f.prep,
        f.graph,
        f.buffers,
        f.buffer_bytes,
        f.textures,
        f.peak,
        f.resident,
        f.frame,
        f.geometry_hits,
        f.texture_hits,
        f.reuse,
        f.plans,
        f.visible,
        f.batches,
        f.draws
    )
}
fn json(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
