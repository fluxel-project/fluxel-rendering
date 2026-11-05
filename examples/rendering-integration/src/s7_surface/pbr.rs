//! PBR Surface cases sharing the real Deferred Surface route with Color.

use std::{error::Error, fs, path::Path};

use external_renderer::{ExternalDeferredPipeline, PBR_PROPERTIES, pbr_definition};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, FramePipeline,
    GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds,
    RenderChange, RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView,
    Renderer, SceneAssetRef, prepare_forward,
};
use fluxel_rhi::{
    api::{Surface, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};
use fluxel_shader::ShaderComposer;

use crate::{
    block_on,
    s7_deferred::pbr_oracle::{self, Input, Route},
};

use super::{COMPANION_EXTENT, asset_json};

const POSITIONS: [[f64; 3]; 3] = [[-0.4, -0.4, 0.5], [0.2, -0.4, 0.05], [-0.4, 0.35, 0.5]];
const NORMALS: [[f64; 3]; 3] = [[0.6, 0.0, 0.8]; 3];
const IDENTITY: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const METAL: [u8; 4] = [180, 100, 40, 255];

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    roughness: f64,
}

impl Case {
    fn input(self) -> Input {
        Input {
            positions: POSITIONS.map(|value| value.map(|component| f64::from(component as f32))),
            normals: NORMALS.map(|value| value.map(|component| f64::from(component as f32))),
            transform: IDENTITY.map(|value| value.map(|component| f64::from(component as f32))),
            color: METAL,
            roughness: f64::from(self.roughness as f32),
        }
    }
}

pub(super) fn run<B, Capture, Pump>(
    renderer: &mut Renderer<B>,
    target: &mut Surface<B>,
    surface_extent: [u32; 2],
    volume: RenderBounds,
    output: &Path,
    capture: &Capture,
    pump: &Pump,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
    Capture: Fn(usize, [u32; 2], [u8; 4]) -> Result<(), Box<dyn Error>>,
    Pump: Fn() -> Result<(), Box<dyn Error>>,
{
    let assets = Assets::new()?;
    assets.write(output)?;
    let forward_assets = assets.forward();
    let pipeline = ExternalDeferredPipeline::new();
    let cases = [
        Case {
            name: "metal",
            roughness: 0.20,
        },
        Case {
            name: "rough",
            roughness: 0.85,
        },
    ];
    let images: Vec<_> = cases
        .iter()
        .map(|case| pbr_oracle::image(case.input(), Route::Deferred).0)
        .collect();
    if images[0] == images[1] {
        return Err(
            "S7 Deferred Surface PBR metal and rough cases have identical independent pixels"
                .into(),
        );
    }
    for (index, case) in cases.into_iter().enumerate() {
        let record = assets.record(case)?;
        let mut scene = RenderScene::<ForwardResources>::new()?;
        let mut changes = RenderChangeSet::new();
        changes.push(RenderChange::Insert(record.clone()));
        changes.push(RenderChange::InsertView(RenderView::new(
            RenderTransform::identity(),
            surface_extent,
        )?));
        changes.push(RenderChange::InsertView(RenderView::new(
            RenderTransform::identity(),
            COMPANION_EXTENT,
        )?));
        let receipt = scene.apply(&changes)?;
        let surface_view = receipt.inserted_views()[0];
        let companion_view = receipt.inserted_views()[1];
        let snapshot = scene.snapshot();
        let (expected, mask) = pbr_oracle::image(case.input(), Route::Deferred);
        let interior = expected
            .chunks_exact(4)
            .find(|pixel| *pixel != [0; 4])
            .ok_or("S7 Deferred Surface PBR oracle lacks foreground")?;
        let surface_color: [u8; 4] = interior
            .try_into()
            .map_err(|_| "S7 PBR oracle pixel shape")?;
        write_input(
            output,
            case,
            &snapshot,
            &record,
            &forward_assets,
            &pipeline,
            surface_view,
            surface_extent,
            surface_color,
            &expected,
            &mask,
        )?;
        let image = block_on(target.acquire())?;
        if image.extent() != surface_extent {
            return Err("S7 Deferred Surface PBR acquired wrong extent".into());
        }
        let surface = block_on(renderer.submit_deferred_surface(
            &pipeline,
            snapshot.clone(),
            surface_view,
            volume,
            &forward_assets,
            image,
        ))?;
        let facts = renderer
            .deferred_probe_facts()
            .ok_or("S7 Deferred Surface PBR lacks current Graph facts")?;
        if facts.passes != 3 {
            return Err(
                "S7 Deferred Surface PBR must declare upload, geometry and lighting passes".into(),
            );
        }
        block_on(renderer.complete(&surface))?;
        capture(index + 2, surface_extent, surface_color)?;
        crate::surface::await_visual_release(output, index + 2, surface_extent, pump)?;
        let companion = block_on(renderer.submit_deferred(
            &pipeline,
            snapshot,
            companion_view,
            volume,
            &forward_assets,
        ))?;
        let raw = block_on(renderer.readback(&companion))?;
        fs::write(
            output.join(format!(
                "s7-deferred-surface-pbr-{}-readback.rgba",
                case.name
            )),
            &raw,
        )?;
        fs::write(
            output.join(format!(
                "s7-deferred-surface-pbr-{}-readback.ppm",
                case.name
            )),
            pbr_oracle::ppm(&raw)?,
        )?;
        pbr_oracle::verify(
            &raw,
            &expected,
            &mask,
            &format!("Surface PBR {} companion", case.name),
            Route::Deferred,
        )?;
        fs::write(
            output.join(format!("s7-deferred-surface-pbr-{}-facts.json", case.name)),
            format!(
                "{{\"schemaVersion\":1,\"graphPasses\":{},\"pendingAfterComplete\":0,\"surfaceCapture\":\"surface-client-frame-{}.bmp\",\"companionRaw\":\"s7-deferred-surface-pbr-{}-readback.rgba\"}}\n",
                facts.passes,
                index + 2,
                case.name
            ),
        )?;
    }
    fs::write(
        output.join("s7-deferred-surface-pbr-workload.json"),
        "{\"schemaVersion\":1,\"workload\":\"s7-deferred-pbr-surface\",\"sharedArtifacts\":[\"s7-deferred-surface-pbr-geometry.txt\",\"s7-deferred-surface-pbr-position.bin\",\"s7-deferred-surface-pbr-normal.bin\",\"s7-deferred-surface-pbr-material.txt\",\"s7-deferred-surface-pbr-texture.rgba\"],\"cases\":[{\"name\":\"metal\",\"artifacts\":[\"s7-deferred-surface-pbr-metal-input.json\",\"s7-deferred-surface-pbr-metal-expected.rgba\",\"s7-deferred-surface-pbr-metal-expected.ppm\",\"s7-deferred-surface-pbr-metal-stable-mask.bin\",\"s7-deferred-surface-pbr-metal-composed.wgsl\",\"s7-deferred-surface-pbr-metal-source-map.txt\",\"s7-deferred-surface-pbr-metal-readback.rgba\",\"s7-deferred-surface-pbr-metal-readback.ppm\",\"s7-deferred-surface-pbr-metal-facts.json\",\"surface-client-frame-2.bmp\",\"surface-client-frame-2.capture.json\",\"surface-client-frame-2.verdict.json\"]},{\"name\":\"rough\",\"artifacts\":[\"s7-deferred-surface-pbr-rough-input.json\",\"s7-deferred-surface-pbr-rough-expected.rgba\",\"s7-deferred-surface-pbr-rough-expected.ppm\",\"s7-deferred-surface-pbr-rough-stable-mask.bin\",\"s7-deferred-surface-pbr-rough-composed.wgsl\",\"s7-deferred-surface-pbr-rough-source-map.txt\",\"s7-deferred-surface-pbr-rough-readback.rgba\",\"s7-deferred-surface-pbr-rough-readback.ppm\",\"s7-deferred-surface-pbr-rough-facts.json\",\"surface-client-frame-3.bmp\",\"surface-client-frame-3.capture.json\",\"surface-client-frame-3.verdict.json\"]}],\"surfaceProof\":\"native-client-capture\",\"companionProof\":\"same-snapshot-offscreen-readback\"}\n",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_input(
    output: &Path,
    case: Case,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    assets: &ForwardAssets,
    pipeline: &ExternalDeferredPipeline,
    surface_view: fluxel_renderer::RenderViewId,
    surface_extent: [u32; 2],
    expected_surface_interior: [u8; 4],
    expected: &[u8],
    mask: &[u8],
) -> Result<(), Box<dyn Error>> {
    let prepared = pipeline.prepare(
        snapshot,
        surface_view,
        RenderBounds::new([-1.0; 3], [1.0; 3])?,
    )?;
    let draws = prepare_forward(&prepared, assets, &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("S7 PBR Surface expected one prepared draw".into());
    };
    let shader = draw.prepared().shader();
    if !shader.source().contains(PBR_PROPERTIES) {
        return Err("S7 PBR Surface composed source omitted PBR properties".into());
    }
    let input = case.input();
    fs::write(
        output.join(format!("s7-deferred-surface-pbr-{}-input.json", case.name)),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"surfaceExtent\":{:?},\"companionExtent\":{:?},",
                "\"textureRgba\":{:?},\"tint\":{:?},\"roughness\":{},\"positions\":{:?},\"normals\":{:?},\"transformColumns\":{:?},",
                "\"expectedSurfaceInteriorRgba\":{:?},\"triangleRgbTolerance\":1,\"backgroundRgbTolerance\":0,",
                "\"sharedInputFiles\":[\"s7-deferred-surface-pbr-geometry.txt\",\"s7-deferred-surface-pbr-position.bin\",\"s7-deferred-surface-pbr-normal.bin\",\"s7-deferred-surface-pbr-material.txt\",\"s7-deferred-surface-pbr-texture.rgba\"],",
                "\"sourceUnitsFile\":\"s7-deferred-surface-pbr-{}-source-map.txt\",",
                "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            surface_extent,
            COMPANION_EXTENT,
            input.color,
            record.resources.tint,
            input.roughness,
            input.positions,
            input.normals,
            input.transform,
            expected_surface_interior,
            case.name,
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture),
        ),
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-pbr-{}-expected.rgba",
            case.name
        )),
        expected,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-pbr-{}-expected.ppm",
            case.name
        )),
        pbr_oracle::ppm(expected)?,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-pbr-{}-stable-mask.bin",
            case.name
        )),
        mask,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-pbr-{}-composed.wgsl",
            case.name
        )),
        shader.source(),
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-pbr-{}-source-map.txt",
            case.name
        )),
        format!(
            "{:#?}\nsource units:\n{:#?}\n",
            shader.source_map(),
            draw.prepared().material().contribution().source_units()
        ),
    )?;
    Ok(())
}

struct Assets {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}
impl Assets {
    fn new() -> Result<Self, Box<dyn Error>> {
        let geometries = AssetStore::new();
        let materials = AssetStore::new();
        let textures = AssetStore::new();
        Ok(Self {
            geometry: geometry(&geometries)?,
            material: material(&materials)?,
            texture: texture(&textures)?,
        })
    }
    fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: vec![self.material.clone()],
            textures: vec![self.texture.clone()],
        }
    }
    fn record(&self, case: Case) -> Result<RenderRecord<ForwardResources>, Box<dyn Error>> {
        Ok(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(&self.texture),
                tint: [1.0, 1.0, 1.0, case.roughness as f32],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-0.8; 3], [0.8; 3])?,
        })
    }
    fn write(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            output.join("s7-deferred-surface-pbr-geometry.txt"),
            format!("{:#?}\n", self.geometry.value()),
        )?;
        fs::write(
            output.join("s7-deferred-surface-pbr-position.bin"),
            &self.geometry.value().streams[0],
        )?;
        fs::write(
            output.join("s7-deferred-surface-pbr-normal.bin"),
            &self.geometry.value().streams[1],
        )?;
        fs::write(
            output.join("s7-deferred-surface-pbr-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        fs::write(
            output.join("s7-deferred-surface-pbr-texture.rgba"),
            &self.texture.value().rgba8,
        )?;
        Ok(())
    }
}
fn geometry(
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
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(72))?),
        _ => Err("S7 PBR Surface geometry producer missing".into()),
    }
}
fn material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => {
            Ok(producer.commit(pbr_definition(), ResidentBytes::new(0))?)
        }
        _ => Err("S7 PBR Surface material producer missing".into()),
    }
}
fn texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(
            ColorTextureData {
                extent: [1, 1],
                rgba8: METAL.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("S7 PBR Surface texture producer missing".into()),
    }
}
fn bytes(values: [[f64; 3]; 3]) -> Vec<u8> {
    values
        .into_iter()
        .flatten()
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect()
}
