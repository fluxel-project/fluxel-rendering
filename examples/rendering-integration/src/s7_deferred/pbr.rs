//! Native external metallic-roughness consumer evidence.

use std::{error::Error, fs, path::Path};

use external_renderer::{ExternalDeferredPipeline, PBR_PROPERTIES, pbr_definition};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, FramePipeline, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, Renderer, SceneAssetRef, prepare_forward,
};
use fluxel_rhi::{
    api::{VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::ShaderComposer;

use super::{
    asset_json,
    fixtures::EXTENT,
    pbr_oracle::{self, Input, Route},
    verify_facts, write_facts,
};
use crate::block_on;

const POSITIONS: [[f64; 3]; 3] = [[-0.4, -0.4, 0.5], [0.2, -0.4, 0.05], [-0.4, 0.35, 0.5]];
const NORMALS: [[f64; 3]; 3] = [[0.6, 0.0, 0.8]; 3];
const IDENTITY: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const SCALED_ROTATED: [[f64; 4]; 4] = [
    [1.039_230_484_5, 0.0, -0.6, 0.0],
    [0.0, 0.7, 0.0, 0.0],
    [0.25, 0.0, 0.433_012_701_9, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) fn run<B>(
    renderer: &mut Renderer<B>,
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
    let assets = Assets::new()?;
    assets.write(output)?;
    let cases = [
        Case::new("pbr-dielectric", Texture::Dielectric, 0.20, IDENTITY),
        Case::new("pbr-metal", Texture::Metal, 0.20, IDENTITY),
        Case::new("pbr-rough", Texture::Metal, 0.85, IDENTITY),
        Case::new("pbr-base-color", Texture::BaseColor, 0.85, IDENTITY),
        Case::new("pbr-normal-transform", Texture::Metal, 0.85, SCALED_ROTATED),
    ];
    let expected: Vec<_> = cases
        .iter()
        .map(|case| pbr_oracle::image(case.input(), Route::Deferred).0)
        .collect();
    if expected.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("S7 PBR cases do not independently change the expected image".into());
    }
    let mut key = None;
    for case in cases {
        let observed = run_case(renderer, &assets, visibility, output, case)?;
        if let Some(expected) = &key {
            if expected != &observed {
                return Err("S7 PBR dynamic tint/texture changed the compiled variant key".into());
            }
        } else {
            key = Some(observed);
        }
    }
    let key = key
        .as_deref()
        .ok_or("S7 PBR cases were unexpectedly empty")?;
    fs::write(
        output.join("s7-pbr-variants.json"),
        format!(
            "{{\"schemaVersion\":1,\"compiledVariantKey\":{},\"materialSnapshotShared\":true,\"propertyNode\":{}}}\n",
            crate::json_string(key),
            crate::json_string(PBR_PROPERTIES),
        ),
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    texture: Texture,
    roughness: f64,
    transform: [[f64; 4]; 4],
}
impl Case {
    const fn new(
        name: &'static str,
        texture: Texture,
        roughness: f64,
        transform: [[f64; 4]; 4],
    ) -> Self {
        Self {
            name,
            texture,
            roughness,
            transform,
        }
    }
    fn input(self) -> Input {
        Input {
            positions: POSITIONS.map(|v| v.map(|x| f64::from(x as f32))),
            normals: NORMALS.map(|v| v.map(|x| f64::from(x as f32))),
            transform: self.transform.map(|v| v.map(|x| f64::from(x as f32))),
            color: self.texture.color(),
            roughness: f64::from(self.roughness as f32),
        }
    }
}
#[derive(Clone, Copy)]
enum Texture {
    Dielectric,
    Metal,
    BaseColor,
}
impl Texture {
    const fn color(self) -> [u8; 4] {
        match self {
            Self::Dielectric => [180, 100, 40, 0],
            Self::Metal => [180, 100, 40, 255],
            Self::BaseColor => [96, 160, 60, 255],
        }
    }
}

fn run_case<B>(
    renderer: &mut Renderer<B>,
    assets: &Assets,
    visibility: RenderBounds,
    output: &Path,
    case: Case,
) -> Result<String, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let record = assets.record(case);
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let view = receipt.inserted_views()[0];
    let snapshot = scene.snapshot();
    let pipeline = ExternalDeferredPipeline::new();
    let frame = pipeline.prepare(&snapshot, view, visibility)?;
    let draws = prepare_forward(&frame, &assets.forward(), &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("S7 PBR expected exactly one draw".into());
    };
    let shader = draw.prepared().shader();
    if !shader.source().contains(PBR_PROPERTIES) {
        return Err("S7 PBR composed source omitted properties node".into());
    }
    let (expected, mask) = pbr_oracle::image(case.input(), Route::Deferred);
    let (forward_expected, forward_mask) = pbr_oracle::image(case.input(), Route::Forward);
    let variant_key = draw.prepared().material().key().as_str().to_owned();
    write_inputs(
        output,
        case,
        &snapshot,
        &record,
        &expected,
        &forward_expected,
        &mask,
        shader.source(),
        shader.source_map(),
        &variant_key,
    )?;
    if renderer.pending() != 0 {
        return Err("S7 PBR began with pending work".into());
    }
    let forward_assets = assets.forward();
    let deferred = block_on(renderer.submit_deferred(
        &pipeline,
        snapshot.clone(),
        view,
        visibility,
        &forward_assets,
    ))?;
    let after_deferred = renderer.pending();
    fs::write(
        output.join(format!("s7-{}-graph-owners.txt", case.name)),
        renderer
            .deferred_probe_diagnostic(&deferred)?
            .ok_or("S7 PBR has no Graph owners")?,
    )?;
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("S7 PBR lacks facts")?;
    verify_facts(facts)?;
    write_facts(
        output,
        case.name,
        0,
        after_deferred,
        None,
        [None, None],
        facts,
    )?;
    let actual = block_on(renderer.readback(&deferred))?;
    fs::write(
        output.join(format!("s7-{}-deferred.rgba", case.name)),
        &actual,
    )?;
    fs::write(
        output.join(format!("s7-{}-deferred.ppm", case.name)),
        pbr_oracle::ppm(&actual)?,
    )?;
    let after_deferred_readback = renderer.pending();
    write_facts(
        output,
        case.name,
        0,
        after_deferred,
        Some(after_deferred_readback),
        [None, None],
        facts,
    )?;
    if after_deferred != 1 || after_deferred_readback != 0 {
        return Err("S7 PBR Deferred pending transition differs from 0 -> 1 -> 0".into());
    }
    pbr_oracle::verify(&actual, &expected, &mask, case.name, Route::Deferred)?;
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        visibility,
        &forward_assets,
    ))?;
    let after_forward = renderer.pending();
    let forward_actual = block_on(renderer.readback(&forward))?;
    fs::write(
        output.join(format!("s7-{}-forward.rgba", case.name)),
        &forward_actual,
    )?;
    fs::write(
        output.join(format!("s7-{}-forward.ppm", case.name)),
        pbr_oracle::ppm(&forward_actual)?,
    )?;
    let after_forward_readback = renderer.pending();
    write_facts(
        output,
        case.name,
        0,
        after_deferred,
        Some(after_deferred_readback),
        [Some(after_forward), Some(after_forward_readback)],
        facts,
    )?;
    if after_forward != 1 || after_forward_readback != 0 {
        return Err("S7 PBR Forward pending transition differs from 0 -> 1 -> 0".into());
    }
    pbr_oracle::verify(
        &forward_actual,
        &forward_expected,
        &forward_mask,
        &format!("{} Forward", case.name),
        Route::Forward,
    )?;
    Ok(variant_key)
}

#[allow(clippy::too_many_arguments)]
fn write_inputs(
    output: &Path,
    case: Case,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    expected: &[u8],
    forward_expected: &[u8],
    mask: &[u8],
    source: &str,
    source_map: &[fluxel_shader::SourceSpan],
    key: &str,
) -> Result<(), Box<dyn Error>> {
    let input = case.input();
    fs::write(
        output.join(format!("s7-{}-input.json", case.name)),
        format!(
            "{{\"schemaVersion\":1,\"sceneVersion\":{},\"textureRgba\":{:?},\"tint\":{:?},\"roughness\":{},\"positions\":{:?},\"normals\":{:?},\"transformColumns\":{:?},\"savedShaderRoute\":\"forward-shared-preparation\",\"sameSnapshotForDeferredAndForward\":true,\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n",
            snapshot.version(),
            case.texture.color(),
            record.resources.tint.map(f64::from),
            input.roughness,
            input.positions,
            input.normals,
            input.transform,
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture)
        ),
    )?;
    for (suffix, value) in [
        ("expected.rgba", expected),
        ("forward-expected.rgba", forward_expected),
    ] {
        fs::write(output.join(format!("s7-{}-{suffix}", case.name)), value)?;
        fs::write(
            output.join(format!(
                "s7-{}-{}.ppm",
                case.name,
                suffix.trim_end_matches(".rgba")
            )),
            pbr_oracle::ppm(value)?,
        )?;
    }
    fs::write(
        output.join(format!("s7-{}-stable-mask.bin", case.name)),
        mask,
    )?;
    fs::write(
        output.join(format!("s7-{}-composed.wgsl", case.name)),
        source,
    )?;
    fs::write(
        output.join(format!("s7-{}-source-map.txt", case.name)),
        format!("{source_map:#?}\nvariant key: {key}\n"),
    )?;
    Ok(())
}

struct Assets {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    dielectric: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    metal: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    base: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}
impl Assets {
    fn new() -> Result<Self, Box<dyn Error>> {
        let geometries = AssetStore::new();
        let materials = AssetStore::new();
        let textures = AssetStore::new();
        Ok(Self {
            geometry: geometry_snapshot(&geometries)?,
            material: material_snapshot(&materials)?,
            dielectric: texture_snapshot(&textures, Texture::Dielectric)?,
            metal: texture_snapshot(&textures, Texture::Metal)?,
            base: texture_snapshot(&textures, Texture::BaseColor)?,
        })
    }
    fn texture(&self, kind: Texture) -> &AssetSnapshot<ColorTextureAsset, ColorTextureData> {
        match kind {
            Texture::Dielectric => &self.dielectric,
            Texture::Metal => &self.metal,
            Texture::BaseColor => &self.base,
        }
    }
    fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: vec![self.material.clone()],
            textures: vec![
                self.dielectric.clone(),
                self.metal.clone(),
                self.base.clone(),
            ],
        }
    }
    fn record(&self, case: Case) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(self.texture(case.texture)),
                tint: [1.0, 1.0, 1.0, case.roughness as f32],
            },
            transform: RenderTransform::from_columns(
                case.transform
                    .map(|column| column.map(|value| value as f32)),
            )
            .expect("finite PBR transform"),
            bounds: RenderBounds::new([-0.8; 3], [0.8; 3]).expect("finite PBR bounds"),
        }
    }
    fn write(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            output.join("s7-pbr-geometry.txt"),
            format!("{:#?}\n", self.geometry.value()),
        )?;
        fs::write(
            output.join("s7-pbr-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        for (name, texture) in [
            ("dielectric", &self.dielectric),
            ("metal", &self.metal),
            ("base-color", &self.base),
        ] {
            fs::write(
                output.join(format!("s7-pbr-{name}.rgba")),
                &texture.value().rgba8,
            )?;
        }
        Ok(())
    }
}
fn geometry_snapshot(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let value = GeometryData {
        layout: GeometryLayout {
            streams: vec![
                GeometryStream {
                    stride: 12,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    }],
                },
                GeometryStream {
                    stride: 12,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![GeometryAttribute {
                        semantic: "NORMAL".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    }],
                },
            ],
        },
        streams: vec![bytes(POSITIONS), bytes(NORMALS)],
        vertices: 0..3,
    };
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(72))?),
        _ => Err("S7 PBR geometry producer missing".into()),
    }
}
fn material_snapshot(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(pbr_definition(), ResidentBytes::new(0))?),
        _ => Err("S7 PBR material producer missing".into()),
    }
}
fn texture_snapshot(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    texture: Texture,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let value = ColorTextureData {
        extent: [1, 1],
        rgba8: texture.color().to_vec(),
    };
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(value, ResidentBytes::new(4))?),
        _ => Err("S7 PBR texture producer missing".into()),
    }
}
fn bytes(values: [[f64; 3]; 3]) -> Vec<u8> {
    let mut result = Vec::with_capacity(36);
    for value in values.into_iter().flatten() {
        result.extend_from_slice(&(value as f32).to_le_bytes());
    }
    result
}
