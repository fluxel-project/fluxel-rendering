//! M0 real-device proof for one node-authored ordinary sampled material.
//!
//! This intentionally tests only the smallest texture-sampling path: a
//! node-authored surface callable samples a one-pixel color texture through an
//! ordinary sampler, then the same frame snapshot is submitted through the
//! Forward and Deferred paths and read back. It is not a full PBR,
//! transparency, or resolve test.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialNode,
    MaterialNodeGraph, MaterialNodeId, MaterialParameter, MaterialResource, MaterialResourceKind,
    MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use crate::block_on;

const EXTENT: [u32; 2] = [64, 64];
const SAMPLE: [u8; 4] = [43, 173, 221, 255];

/// Runs the sampled-texture Forward and Deferred workload on one device.
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
    let geometry = snapshot_geometry(&geometry_store, fullscreen_geometry())?;
    let material = snapshot_material(&material_store, sampled_node_material())?;
    let texture = snapshot_texture(
        &texture_store,
        ColorTextureData {
            encoding: fluxel_renderer::TextureEncoding::Linear,
            mip_levels: 1,
            extent: [1, 1],
            rgba8: SAMPLE.to_vec(),
        },
    )?;

    fs::write(
        output.join("m0-sampled-forward-geometry.bin"),
        &geometry.value().streams[0],
    )?;
    fs::write(
        output.join("m0-sampled-forward-material.txt"),
        format!("{:#?}\n", material.value()),
    )?;
    fs::write(
        output.join("m0-sampled-forward-texture.rgba"),
        &texture.value().rgba8,
    )?;
    fs::write(
        output.join("m0-sampled-forward-input.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"workload\":\"m0-node-texture-sample-forward-deferred\",",
                "\"extent\":{:?},\"textureRgba\":{:?},\"sampleCoordinates\":[0.5,0.5],",
                "\"materialGraph\":{{\"nodes\":[7],\"output\":7}},",
                "\"bindingLayout\":[\"g0b0 uniform\",\"g0b1 texture_2d<f32>\",",
                "\"g0b2 sampler\"]}}\n"
            ),
            EXTENT, SAMPLE
        ),
    )?;

    let forward_expected = std::iter::repeat_n(SAMPLE, (EXTENT[0] * EXTENT[1]) as usize)
        .flatten()
        .collect::<Vec<_>>();
    // The private Deferred probe encodes/decodes the normal through RGBA8,
    // then applies its documented 0.9 flat-plane lighting factor. The two
    // UNORM conversions permit one RGB code difference from this ideal.
    let deferred_sample =
        SAMPLE.map(|component| (f32::from(component) / 255.0 * 0.9 * 255.0).round() as u8);
    let deferred_expected = std::iter::repeat_n(
        [
            deferred_sample[0],
            deferred_sample[1],
            deferred_sample[2],
            SAMPLE[3],
        ],
        (EXTENT[0] * EXTENT[1]) as usize,
    )
    .flatten()
    .collect::<Vec<_>>();
    fs::write(
        output.join("m0-sampled-forward-expected.rgba"),
        &forward_expected,
    )?;
    fs::write(
        output.join("m0-sampled-forward-expected.ppm"),
        ppm(&forward_expected)?,
    )?;
    fs::write(
        output.join("m0-sampled-deferred-expected.rgba"),
        &deferred_expected,
    )?;
    fs::write(
        output.join("m0-sampled-deferred-expected.ppm"),
        ppm(&deferred_expected)?,
    )?;
    fs::write(
        output.join("m0-sampled-oracle.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"forward\":{{\"rgbTolerance\":0,\"alphaTolerance\":0}},",
                "\"deferred\":{{\"rgbTolerance\":1,\"alphaTolerance\":0,",
                "\"idealFlatPlaneFactor\":0.9}},\"sampleRgba\":{:?}}}\n"
            ),
            SAMPLE
        ),
    )?;

    let record = RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&texture),
            )],
            sampler: None,
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-2.0; 3], [2.0; 3])?,
    };
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let assets = ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    };
    let mut renderer = Renderer::new(device)?;
    let snapshot = scene.snapshot();
    let view = receipt.inserted_views()[0];
    let visibility = RenderBounds::new([-2.0; 3], [2.0; 3])?;
    let deferred = block_on(renderer.submit_deferred(
        &ExternalDeferredPipeline::new(),
        snapshot.clone(),
        view,
        visibility,
        &assets,
    ))?;
    if renderer.pending() != 1 {
        return Err("M0 sampled Deferred submission did not retain its pending owner".into());
    }
    let deferred_actual = block_on(renderer.readback(&deferred))?;
    fs::write(
        output.join("m0-sampled-deferred-readback.rgba"),
        &deferred_actual,
    )?;
    fs::write(
        output.join("m0-sampled-deferred-readback.ppm"),
        ppm(&deferred_actual)?,
    )?;
    if renderer.pending() != 0 {
        return Err("M0 sampled Deferred readback did not retire its submission owner".into());
    }
    verify_deferred(&deferred_actual, &deferred_expected)?;

    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        visibility,
        &assets,
    ))?;
    if renderer.pending() != 1 {
        return Err("M0 sampled Forward submission did not retain its pending owner".into());
    }
    let actual = block_on(renderer.readback(&submission))?;
    fs::write(output.join("m0-sampled-forward-readback.rgba"), &actual)?;
    fs::write(
        output.join("m0-sampled-forward-readback.ppm"),
        ppm(&actual)?,
    )?;
    if renderer.pending() != 0 {
        return Err("M0 sampled Forward readback did not retire its submission owner".into());
    }
    if actual != forward_expected {
        return Err(
            "M0 sampled Forward readback differs from independent one-pixel expectation".into(),
        );
    }
    fs::write(
        output.join("m0-sampled-forward-result.json"),
        "{\"schemaVersion\":1,\"result\":\"forward-and-deferred-readbacks-matched-independent-expected\",\"scope\":\"ordinary sampled texture; no full PBR, transparency, or resolve claim\"}\n",
    )?;
    Ok(())
}

fn verify_deferred(actual: &[u8], expected: &[u8]) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err("M0 sampled Deferred readback has an unexpected byte count".into());
    }
    for (index, (actual, expected)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        if actual[3] != expected[3]
            || actual[..3]
                .iter()
                .zip(&expected[..3])
                .any(|(actual, expected)| actual.abs_diff(*expected) > 1)
        {
            return Err(format!(
                "M0 sampled Deferred differs at pixel {index}: actual {actual:?}, expected {expected:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn sampled_node_material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "sampled_color".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn sampled_color() -> vec4<f32> { return textureSample(material_texture, material_sampler, vec2<f32>(0.5, 0.5)) * tint; }".into(),
        }],
        "sampled_color",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
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
            MaterialResource {
                name: "material_sampler".into(),
                kind: MaterialResourceKind::Sampler { comparison: false },
                location: BindingLocation {
                    group: 0,
                    binding: 2,
                },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode {
            id: MaterialNodeId(7),
            callable: "sampled_color".into(),
            inputs: Vec::new(),
        }],
        output: MaterialNodeId(7),
    })
}

fn fullscreen_geometry() -> GeometryData {
    let positions = [[-1.0_f32, -1.0, 0.5], [3.0, -1.0, 0.5], [-1.0, 3.0, 0.5]];
    let mut bytes = Vec::with_capacity(36);
    for position in positions {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
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
        streams: vec![bytes],
        vertices: 0..3,
    }
}

fn snapshot_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    value: GeometryData,
) -> Result<fluxel_assets::AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(36))?),
        _ => Err("M0 sampled geometry producer missing".into()),
    }
}

fn snapshot_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    value: MaterialDefinition,
) -> Result<fluxel_assets::AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(0))?),
        _ => Err("M0 sampled material producer missing".into()),
    }
}

fn snapshot_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    value: ColorTextureData,
) -> Result<fluxel_assets::AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(4))?),
        _ => Err("M0 sampled texture producer missing".into()),
    }
}

fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (EXTENT[0] * EXTENT[1] * 4) as usize {
        return Err("M0 sampled Forward image has an unexpected byte count".into());
    }
    let mut output = format!("P6\n{} {}\n255\n", EXTENT[0], EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        output.extend_from_slice(&pixel[..3]);
    }
    Ok(output)
}
