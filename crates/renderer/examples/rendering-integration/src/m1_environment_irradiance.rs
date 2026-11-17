//! Forward diffuse-irradiance device workload.
//!
//! This fixture proves only the first environment subset: a linear RGBA8
//! lat-long map containing `E / pi`, sampled for Lambert diffuse. It does not
//! claim prefiltered radiance, a BRDF LUT, or specular IBL.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialCodeNode, MaterialDefinition, MaterialNode,
    MaterialNodeGraph, MaterialNodeId, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, EnvironmentIrradianceAsset,
    EnvironmentIrradianceData, ForwardAssets, ForwardCpuPipeline, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderChange,
    RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView, Renderer,
    SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::SourceUnit;

use crate::block_on;

const EXTENT: [u32; 2] = [64, 64];
pub(crate) const BASE: [f64; 3] = [0.5, 0.5, 0.5];
// Two identical latitude rows make equatorial sampling independent of the
// vertical filter while preserving a genuine 2D lat-long texture.
pub(crate) const OLD_ENV: [u8; 32] = [
    64, 0, 0, 17, 0, 64, 0, 31, 0, 0, 64, 47, 64, 64, 0, 63, 64, 0, 0, 17, 0, 64, 0, 31, 0, 0, 64,
    47, 64, 64, 0, 63,
];
pub(crate) const NEW_ENV: [u8; 32] = [
    8, 96, 32, 255, 24, 12, 96, 255, 96, 24, 12, 255, 96, 48, 8, 255, 8, 96, 32, 255, 24, 12, 96,
    255, 96, 24, 12, 255, 96, 48, 8, 255,
];

#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) normal: [f64; 3],
    pub(crate) metallic: f64,
    pub(crate) environment: Environment,
}

#[derive(Clone, Copy)]
pub(crate) enum Environment {
    None,
    Old,
    Replacement,
}

pub(crate) fn run<B>(device: Device<B>, out: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(out)?;
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let environments =
        AssetStore::<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>::new();
    let texture = snapshot_texture(&textures)?;
    let environment_handle = environments.create()?;
    let old = produce_environment(&environments, &environment_handle, OLD_ENV)?;
    let replacement = replace_environment(&environments, &environment_handle, NEW_ENV)?;
    let cases = [
        Case {
            name: "direction-green",
            normal: direction(-std::f64::consts::FRAC_PI_4),
            metallic: 0.0,
            environment: Environment::Old,
        },
        Case {
            name: "direction-blue",
            normal: direction(std::f64::consts::FRAC_PI_4),
            metallic: 0.0,
            environment: Environment::Old,
        },
        Case {
            name: "metal",
            normal: direction(std::f64::consts::FRAC_PI_4),
            metallic: 1.0,
            environment: Environment::Old,
        },
        Case {
            name: "metal-none",
            normal: direction(std::f64::consts::FRAC_PI_4),
            metallic: 1.0,
            environment: Environment::None,
        },
        Case {
            name: "none",
            normal: direction(-std::f64::consts::FRAC_PI_4),
            metallic: 0.0,
            environment: Environment::None,
        },
        // Both normals lie on opposite sides of the longitude seam. Repeat-U
        // must yield the same bilinear blend of the last and first texels.
        Case {
            name: "seam-negative",
            normal: direction(-std::f64::consts::PI + 0.01),
            metallic: 0.0,
            environment: Environment::Old,
        },
        Case {
            name: "seam-positive",
            normal: direction(std::f64::consts::PI - 0.01),
            metallic: 0.0,
            environment: Environment::Old,
        },
        // `replacement` is a new generation for the same logical asset
        // handle. Selecting it after `old` tests the exact setter snapshot.
        Case {
            name: "replacement",
            normal: direction(-std::f64::consts::FRAC_PI_4),
            metallic: 0.0,
            environment: Environment::Replacement,
        },
    ];
    // Persist all source bytes and each independent expected result before a
    // submission can allocate, upload, or execute GPU work.
    fs::write(out.join("m1-environment-old-e-over-pi.rgba"), OLD_ENV)?;
    fs::write(
        out.join("m1-environment-replacement-e-over-pi.rgba"),
        NEW_ENV,
    )?;
    fs::write(out.join("m1-environment-input.json"), input_json(&cases))?;
    let mut renderer = Renderer::new(device)?;
    let mut results = Vec::new();
    for case in cases {
        let forward_expected = crate::m1_environment_oracle::forward_expected(case);
        let deferred_expected = crate::m1_environment_oracle::deferred_expected(case);
        fs::write(
            out.join(format!("m1-environment-{}-oracle.json", case.name)),
            oracle_json(case, forward_expected, deferred_expected),
        )?;
        let geometry = snapshot_geometry(&geometries, case.normal)?;
        let material = snapshot_material(&materials, case.metallic)?;
        fs::write(
            out.join(format!("m1-environment-{}-geometry.bin", case.name)),
            &geometry.value().streams[0],
        )?;
        select_environment(&mut renderer, case.environment, &old, &replacement)?;
        let deferred_actual = submit_deferred(&mut renderer, &geometry, &material, &texture)?;
        fs::write(
            out.join(format!(
                "m1-environment-{}-deferred-readback.rgba",
                case.name
            )),
            &deferred_actual,
        )?;
        fs::write(
            out.join(format!(
                "m1-environment-{}-deferred-readback.ppm",
                case.name
            )),
            ppm(&deferred_actual)?,
        )?;
        verify_center(
            &deferred_actual,
            deferred_expected,
            9,
            case.name,
            "Deferred",
        )?;
        let actual = submit_forward(&mut renderer, &geometry, &material, &texture)?;
        fs::write(
            out.join(format!("m1-environment-{}-readback.rgba", case.name)),
            &actual,
        )?;
        fs::write(
            out.join(format!("m1-environment-{}-readback.ppm", case.name)),
            ppm(&actual)?,
        )?;
        verify_center(&actual, forward_expected, 5, case.name, "Forward")?;
        results.push((case.name, center(&actual)?));
    }
    require_distinct(&results)?;
    fs::write(
        out.join("m1-environment-result.json"),
        "{\"schemaVersion\":1,\"result\":\"forward-and-deferred-diffuse-irradiance-cases-matched-independent-cpu-oracles\",\"scope\":\"linear RGBA8 lat-long E/pi diffuse only; no specular IBL claim\"}\n",
    )?;
    Ok(())
}

fn select_environment<B>(
    renderer: &mut Renderer<B>,
    selected: Environment,
    old: &AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>,
    replacement: &AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    renderer.set_environment_irradiance(match selected {
        Environment::None => None,
        Environment::Old => Some(old.clone()),
        Environment::Replacement => Some(replacement.clone()),
    })?;
    Ok(())
}

fn submit_forward<B>(
    renderer: &mut Renderer<B>,
    geometry: &AssetSnapshot<GeometryAsset, GeometryData>,
    material: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
) -> Result<Vec<u8>, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let record = RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(geometry),
            material: SceneAssetRef::from_snapshot(material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(texture),
            )],
            sampler: None,
            tint: [0.5, 0.5, 0.5, 1.0],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
    };
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let assets = ForwardAssets {
        geometry: vec![geometry.clone()],
        materials: vec![material.clone()],
        textures: vec![texture.clone()],
    };
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot(),
        receipt.inserted_views()[0],
        RenderBounds::new([-1.0; 3], [1.0; 3])?,
        &assets,
    ))?;
    block_on(renderer.readback(&submission)).map_err(Into::into)
}

fn submit_deferred<B>(
    renderer: &mut Renderer<B>,
    geometry: &AssetSnapshot<GeometryAsset, GeometryData>,
    material: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
) -> Result<Vec<u8>, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let record = RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(geometry),
            material: SceneAssetRef::from_snapshot(material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(texture),
            )],
            sampler: None,
            tint: [0.5, 0.5, 0.5, 1.0],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
    };
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let assets = ForwardAssets {
        geometry: vec![geometry.clone()],
        materials: vec![material.clone()],
        textures: vec![texture.clone()],
    };
    let submission = block_on(renderer.submit_deferred(
        &ExternalDeferredPipeline::new(),
        scene.snapshot(),
        receipt.inserted_views()[0],
        RenderBounds::new([-1.0; 3], [1.0; 3])?,
        &assets,
    ))?;
    block_on(renderer.readback(&submission)).map_err(Into::into)
}

fn material(metallic: f64) -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(),
        "unused",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "tint_u".into(),
        }],
        vec![
            MaterialResource {
                name: "tint_u".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::MetallicRoughness {
            properties: "properties".into(),
        },
    )
    .with_code_node(code(
        "base",
        MaterialType::Vec4,
        "fn base() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }",
    ))
    .with_code_node(code(
        "properties",
        MaterialType::Vec2,
        &format!("fn properties() -> vec2<f32> {{ return vec2<f32>({metallic}, 0.5); }}"),
    ))
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(71),
            callable: "base".into(),
            inputs: Vec::new(),
        }],
        output: MaterialNodeId(71),
    })
}

fn code(name: &str, output: MaterialType, source: &str) -> MaterialCodeNode {
    MaterialCodeNode {
        name: name.into(),
        inputs: Vec::new(),
        output,
        calls: Vec::new(),
        source: SourceUnit::new(format!("m1-environment-{name}"), source),
        resources: Vec::new(),
    }
}

fn geometry(normal: [f64; 3]) -> GeometryData {
    let mut bytes = Vec::new();
    for position in [
        [-1., -1., 0.5],
        [1., -1., 0.5],
        [-1., 1., 0.5],
        [-1., 1., 0.5],
        [1., -1., 0.5],
        [1., 1., 0.5],
    ] {
        for value in position.into_iter().chain(normal.map(|value| value as f32)) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 24,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "NORMAL".into(),
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}

fn snapshot_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    normal: [f64; 3],
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(geometry(normal), ResidentBytes::new(144))?),
        _ => Err("environment geometry producer missing".into()),
    }
}
fn snapshot_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    metallic: f64,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(material(metallic), ResidentBytes::new(0))?),
        _ => Err("environment material producer missing".into()),
    }
}
fn snapshot_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                encoding: fluxel_renderer::TextureEncoding::Linear,
                mip_levels: 1,
                extent: [1, 1],
                rgba8: vec![255; 4],
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("environment texture producer missing".into()),
    }
}
fn produce_environment(
    store: &AssetStore<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>,
    handle: &fluxel_assets::AssetHandle<EnvironmentIrradianceAsset>,
    bytes: [u8; 32],
) -> Result<AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>, Box<dyn Error>> {
    match store.acquire(handle)? {
        Acquire::Producer(p) => Ok(p.commit(
            EnvironmentIrradianceData {
                extent: [4, 2],
                rgba8: bytes.to_vec(),
            },
            ResidentBytes::new(32),
        )?),
        _ => Err("environment producer missing".into()),
    }
}
fn replace_environment(
    store: &AssetStore<EnvironmentIrradianceAsset, EnvironmentIrradianceData, String>,
    handle: &fluxel_assets::AssetHandle<EnvironmentIrradianceAsset>,
    bytes: [u8; 32],
) -> Result<AssetSnapshot<EnvironmentIrradianceAsset, EnvironmentIrradianceData>, Box<dyn Error>> {
    match store.request_replacement(handle)? {
        Production::Producer(p) => Ok(p.commit(
            EnvironmentIrradianceData {
                extent: [4, 2],
                rgba8: bytes.to_vec(),
            },
            ResidentBytes::new(32),
        )?),
        Production::Waiter(_) => Err("environment replacement waiter unexpected".into()),
    }
}

fn direction(angle: f64) -> [f64; 3] {
    [angle.cos(), 0.0, angle.sin()]
}
fn center(bytes: &[u8]) -> Result<[u8; 4], Box<dyn Error>> {
    Ok(
        bytes[(((EXTENT[1] / 2 * EXTENT[0] + EXTENT[0] / 2) * 4) as usize)
            ..(((EXTENT[1] / 2 * EXTENT[0] + EXTENT[0] / 2) * 4 + 4) as usize)]
            .try_into()?,
    )
}
fn verify_center(
    bytes: &[u8],
    expected: [u8; 4],
    tolerance: u8,
    name: &str,
    route: &str,
) -> Result<(), Box<dyn Error>> {
    let found = center(bytes)?;
    if found[3] != 255
        || found[..3]
            .iter()
            .zip(expected[..3].iter())
            .any(|(a, b)| a.abs_diff(*b) > tolerance)
    {
        return Err(format!(
            "environment {name} {route} centre {found:?} differs from independent CPU oracle {expected:?}"
        )
        .into());
    }
    Ok(())
}
fn require_distinct(results: &[(&str, [u8; 4])]) -> Result<(), Box<dyn Error>> {
    let find = |name| {
        results
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, value)| *value)
            .ok_or("missing result")
    };
    if find("direction-green")? == find("direction-blue")?
        || find("none")? == find("direction-green")?
        || find("replacement")? == find("direction-green")?
        || find("seam-negative")? != find("seam-positive")?
        || find("metal")? != find("metal-none")?
    {
        return Err(
            "environment direction, replacement, None, or seam behavior was not observable".into(),
        );
    }
    Ok(())
}
fn input_json(cases: &[Case]) -> String {
    format!(
        "{{\"schemaVersion\":1,\"encoding\":\"linear non-premultiplied RGBA8; RGB is E/pi; alpha ignored\",\"latlong\":\"u=fract(atan2(z,x)/TAU+0.5), v=acos(clamp(y,-1,1))/PI; +Y north\",\"sampler\":\"linear repeat-U clamp-V\",\"extent\":[4,2],\"cases\":[{}]}}\n",
        cases
            .iter()
            .map(|case| format!(
                "{{\"name\":\"{}\",\"normal\":{:?},\"metallic\":{},\"environment\":\"{}\"}}",
                case.name,
                case.normal,
                case.metallic,
                match case.environment {
                    Environment::None => "none",
                    Environment::Old => "old",
                    Environment::Replacement => "replacement",
                }
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn oracle_json(case: Case, forward_expected: [u8; 4], deferred_expected: [u8; 4]) -> String {
    format!(
        "{{\"schemaVersion\":1,\"case\":\"{}\",\"forwardExpectedCenter\":{:?},\"deferredExpectedCenter\":{:?},\"forwardTolerance\":5,\"deferredTolerance\":9,\"forwardOracle\":\"independent direct GGX plus bilinear E/pi lat-long Lambert diffuse\",\"deferredOracle\":\"independent RGBA8 G-buffer UNORM encode/decode, normal normalize, then direct GGX plus bilinear E/pi lat-long Lambert diffuse\"}}\n",
        case.name, forward_expected, deferred_expected
    )
}
fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("environment image byte count invalid".into());
    }
    let mut bytes = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        bytes.extend_from_slice(&pixel[..3]);
    }
    Ok(bytes)
}
