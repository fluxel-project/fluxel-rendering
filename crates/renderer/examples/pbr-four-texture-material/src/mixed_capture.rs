//! Forward-only consumer of a same-frame producer target and named PBR assets.
//!
//! The producer remains an ordinary two-record PBR scene.  The receiver uses
//! the same PBR definition, but its `emissive` binding is the producer target;
//! base color, normal, and ORM remain immutable assets.  No CPU image is made
//! from the capture product.

use crate::scene::{self, Case, ROLE_NAMES};
use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    CaptureTextureSlot, ColorTextureAsset, ColorTextureData, ForwardAssets,
    ForwardCaptureResources, ForwardDrawSource, ForwardTextureBinding, GeometryAsset, GeometryData,
    RenderBounds, RenderScene, RenderTransform, RenderView, SceneAssetRef,
    prepare_forward_capture_receiver,
};
use fluxel_shader::ShaderComposer;
use std::{error::Error, fs, path::Path};

/// Logical name supplied independently by the producer submission.
pub const SLOT: &str = "mixed_producer_color";

/// Actual acceptance controls. Every non-baseline row changes only the named
/// producer or receiver source, never a shared asset identity.
pub const CONTROLS: [(&str, Case, Case); 8] = [
    ("baseline", Case::Baseline, Case::Baseline),
    ("producer-base", Case::Base, Case::Baseline),
    ("producer-normal", Case::Normal, Case::Baseline),
    ("receiver-base", Case::Baseline, Case::Base),
    ("receiver-normal", Case::Baseline, Case::Normal),
    ("receiver-ao", Case::Baseline, Case::Ao),
    ("receiver-roughness", Case::Baseline, Case::Roughness),
    ("receiver-metallic", Case::Baseline, Case::Metallic),
];

/// A producer scene plus its independently prepared PBR receiver.
pub struct Input {
    pub producer: scene::Input,
    /// Includes producer inputs and separately-identifiable receiver assets.
    pub assets: ForwardAssets,
    pub receiver_images: [AssetSnapshot<ColorTextureAsset, ColorTextureData>; 3],
    pub receiver_geometry_bytes: Vec<u8>,
    pub receiver_material: MaterialDefinition,
    pub receiver: fluxel_renderer::PreparedCaptureReceiver,
    pub slot: CaptureTextureSlot,
    pub visibility: RenderBounds,
}

/// Builds one actual mixed-source receiver. `case` changes the producer's
/// nonconstant PBR input; the receiver keeps three physical asset bindings.
pub fn input(case: Case) -> Result<Input, Box<dyn Error>> {
    input_with_receiver(case, Case::Baseline)
}

/// Keeps producer and receiver controls independent. The two scenes have
/// distinct asset identities even when their source bytes initially match.
pub fn input_with_receiver(
    producer_case: Case,
    receiver_case: Case,
) -> Result<Input, Box<dyn Error>> {
    let producer = scene::input(producer_case)?;
    let receiver_source = scene::input(receiver_case)?;
    let producer_geometry = receiver_source
        .assets
        .geometry
        .first()
        .ok_or("missing producer geometry")?;
    let producer_material = receiver_source
        .assets
        .materials
        .first()
        .ok_or("missing producer material")?;
    let sources = receiver_source.sources[0];
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry = commit(
        &geometry_store,
        producer_geometry.value().clone(),
        producer_geometry
            .value()
            .streams
            .iter()
            .map(Vec::len)
            .sum::<usize>() as u64,
    )?;
    let material = commit(&material_store, producer_material.value().clone(), 0)?;
    let receiver_geometry_bytes = producer_geometry.value().streams.concat();
    let receiver_material = producer_material.value().clone();
    let receiver_images = [0, 1, 2]
        .map(|role| {
            let source = &receiver_source.assets.textures[sources[role]];
            commit(
                &texture_store,
                source.value().clone(),
                source.value().rgba8.len() as u64,
            )
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let receiver_images: [AssetSnapshot<ColorTextureAsset, ColorTextureData>; 3] = receiver_images
        .try_into()
        .map_err(|_| "receiver texture source count")?;
    let receiver_textures = [2, 0, 1]
        .map(|role| {
            ForwardTextureBinding::asset(
                ROLE_NAMES[role],
                SceneAssetRef::from_snapshot(&receiver_images[role]),
            )
        })
        .into_iter()
        .chain(std::iter::once(ForwardTextureBinding::capture(
            "emissive", SLOT,
        )))
        .collect();
    let resources = ForwardCaptureResources {
        geometry: SceneAssetRef::from_snapshot(&geometry),
        material: SceneAssetRef::from_snapshot(&material),
        textures: receiver_textures,
        samplers: Vec::new(),
        tint: [0.72, 0.82, 0.92, 1.0],
    };
    let receiver_scene = RenderScene::<ForwardCaptureResources>::new()?;
    let source = ForwardDrawSource::Feature {
        feature: receiver_scene.feature_issuer()?.issue("mixed-capture")?,
        local_id: 0,
    };
    let receiver = prepare_forward_capture_receiver(
        &RenderView::new(RenderTransform::identity(), scene::EXTENT)?,
        source,
        0,
        &resources,
        RenderTransform::identity(),
        &ForwardAssets {
            geometry: vec![geometry.clone()],
            materials: vec![material.clone()],
            textures: receiver_images.to_vec(),
        },
        &mut ShaderComposer::new(),
    )?;
    Ok(Input {
        assets: ForwardAssets {
            geometry: producer
                .assets
                .geometry
                .iter()
                .cloned()
                .chain(std::iter::once(geometry))
                .collect(),
            materials: producer
                .assets
                .materials
                .iter()
                .cloned()
                .chain(std::iter::once(material))
                .collect(),
            textures: producer
                .assets
                .textures
                .iter()
                .cloned()
                .chain(receiver_images.iter().cloned())
                .collect(),
        },
        producer,
        receiver_images,
        receiver_geometry_bytes,
        receiver_material,
        receiver,
        slot: CaptureTextureSlot::new(SLOT)?,
        visibility: RenderBounds::new([-5.; 3], [5.; 3])?,
    })
}

fn commit<K: AssetKind, T: Send + Sync + 'static>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>> {
    match store.acquire(&store.create()?)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new mixed-capture asset has no producer".into()),
    }
}

/// Returns a CPU-only reference image for evidence/oracle code.  It is never
/// placed in `ForwardAssets`, uploaded, or bound by the renderer.
pub fn cpu_capture_reference(input: &Input) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut image = vec![0_u8; (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize];
    for y in 0..scene::EXTENT[1] {
        for x in 0..scene::EXTENT[0] {
            let uv_screen = [
                (f64::from(x) + 0.5) / f64::from(scene::EXTENT[0]),
                (f64::from(y) + 0.5) / f64::from(scene::EXTENT[1]),
            ];
            let mut color = [0; 4];
            for object in 0..2 {
                let model = input.producer.models[object];
                let local = [
                    (uv_screen[0] * 2. - 1. - f64::from(model[3][0])) / f64::from(model[0][0]),
                    (1. - uv_screen[1] * 2. - f64::from(model[3][1])) / f64::from(model[1][1]),
                ];
                if local[0].abs() <= 1. && local[1].abs() <= 1. {
                    let uv = [(local[0] + 1.) * 0.5, (1. - local[1]) * 0.5];
                    color = crate::oracle::shade(&input.producer, object, uv, false);
                }
            }
            let offset = ((y * scene::EXTENT[0] + x) * 4) as usize;
            image[offset..offset + 4].copy_from_slice(&color);
        }
    }
    Ok(image)
}

/// Fixed interior receiver probes, evaluated from independent CPU inputs.
pub fn reference(input: &Input) -> Result<Vec<Probe>, Box<dyn Error>> {
    let captured = cpu_capture_reference(input)?;
    reference_from_capture(input, &captured, false)
}

fn reference_from_capture(
    input: &Input,
    captured: &[u8],
    incorrectly_decode_capture_as_srgb: bool,
) -> Result<Vec<Probe>, Box<dyn Error>> {
    reference_with_receiver_mapping(
        input,
        captured,
        incorrectly_decode_capture_as_srgb,
        [0, 1, 2],
    )
}

fn reference_with_receiver_mapping(
    input: &Input,
    captured: &[u8],
    incorrectly_decode_capture_as_srgb: bool,
    mapping: [usize; 3],
) -> Result<Vec<Probe>, Box<dyn Error>> {
    if captured.len() != (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize {
        return Err("CPU capture reference byte length mismatch".into());
    }
    let base = input.receiver_images[mapping[0]].value();
    let normal = input.receiver_images[mapping[1]].value();
    let orm = input.receiver_images[mapping[2]].value();
    let mut probes = Vec::new();
    for (x, y) in [(97, 121), (223, 251), (359, 211), (431, 391)] {
        let uv = [
            (f64::from(x) + 0.5) / f64::from(scene::EXTENT[0]),
            (f64::from(y) + 0.5) / f64::from(scene::EXTENT[1]),
        ];
        let capture = crate::oracle::sample_image(
            &captured,
            scene::EXTENT,
            if incorrectly_decode_capture_as_srgb {
                fluxel_renderer::TextureEncoding::Srgb
            } else {
                fluxel_renderer::TextureEncoding::Linear
            },
            uv,
        );
        let expected = crate::oracle::shade_samples(
            [0.72, 0.82, 0.92, 1.0],
            &input.producer.diffuse.value().rgba8,
            [
                crate::oracle::sample_image(&base.rgba8, base.extent, base.encoding, uv),
                crate::oracle::sample_image(&normal.rgba8, normal.extent, normal.encoding, uv),
                crate::oracle::sample_image(&orm.rgba8, orm.extent, orm.encoding, uv),
                capture,
            ],
            false,
        );
        probes.push(Probe { x, y, expected });
    }
    Ok(probes)
}

#[derive(Clone, Copy)]
pub struct Probe {
    pub x: u32,
    pub y: u32,
    pub expected: [u8; 4],
}

pub fn verify(raw: &[u8], probes: &[Probe]) -> Result<(), Box<dyn Error>> {
    if raw.len() != (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize {
        return Err("mixed readback size mismatch".into());
    }
    for probe in probes {
        let offset = ((probe.y * scene::EXTENT[0] + probe.x) * 4) as usize;
        if raw[offset..offset + 3]
            .iter()
            .zip(&probe.expected[..3])
            .any(|(actual, expected)| actual.abs_diff(*expected) > crate::oracle::TOLERANCE)
        {
            return Err(format!(
                "mixed probe {},{} differs from independent reference",
                probe.x, probe.y
            )
            .into());
        }
    }
    Ok(())
}

/// Saves the actual, immutable source bytes before any protected GPU work.
pub fn write_inputs(path: &Path, input: &Input, probes: &[Probe]) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(path)?;
    fs::write(
        path.join("producer-geometry.bin"),
        &input.producer.geometry_bytes,
    )?;
    fs::write(
        path.join("producer-material.txt"),
        format!("{:#?}\n", input.producer.material),
    )?;
    fs::write(
        path.join("receiver-geometry.bin"),
        &input.receiver_geometry_bytes,
    )?;
    fs::write(
        path.join("receiver-material.txt"),
        format!("{:#?}\n", input.receiver_material),
    )?;
    fs::write(
        path.join("diffuse-irradiance.rgba"),
        &input.producer.diffuse.value().rgba8,
    )?;
    fs::write(path.join("Cargo.lock"), include_bytes!("../Cargo.lock"))?;
    for (index, texture) in input.producer.assets.textures.iter().enumerate() {
        fs::write(
            path.join(format!("producer-source-{index}.rgba")),
            &texture.value().rgba8,
        )?;
    }
    for (role, texture) in input.receiver_images.iter().enumerate() {
        fs::write(
            path.join(format!("receiver-role-{role}.rgba")),
            &texture.value().rgba8,
        )?;
    }
    let data = probes
        .iter()
        .map(|probe| {
            format!(
                "{{\"x\":{},\"y\":{},\"expected\":[{},{},{},{}]}}",
                probe.x,
                probe.y,
                probe.expected[0],
                probe.expected[1],
                probe.expected[2],
                probe.expected[3]
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let texture_fact = |asset: &AssetSnapshot<ColorTextureAsset, ColorTextureData>| {
        let value = asset.value();
        format!(
            "{{\"id\":{:?},\"generation\":{},\"extent\":{:?},\"encoding\":{:?}}}",
            format!("{:?}", asset.id()),
            asset.generation().get(),
            value.extent,
            format!("{:?}", value.encoding),
        )
    };
    let producer_textures = input
        .producer
        .assets
        .textures
        .iter()
        .map(texture_fact)
        .collect::<Vec<_>>()
        .join(",");
    let receiver_textures = input
        .receiver_images
        .iter()
        .map(texture_fact)
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        path.join("mixed-input.json"),
        format!(
            "{{\"producerSlot\":\"{SLOT}\",\"extent\":{:?},\"view\":\"identity clip transform\",\"producer\":{{\"geometry\":{{\"id\":{:?},\"generation\":{},\"layout\":{:?}}},\"material\":{{\"id\":{:?},\"generation\":{}}},\"sources\":{:?},\"textures\":[{producer_textures}],\"models\":{:?},\"tints\":{:?}}},\"receiver\":{{\"geometry\":{{\"id\":{:?},\"generation\":{},\"layout\":{:?}}},\"material\":{{\"id\":{:?},\"generation\":{}}},\"textures\":[{receiver_textures}],\"tint\":[0.72,0.82,0.92,1.0],\"bindings\":{{\"base\":\"asset[0]\",\"normal\":\"asset[1]\",\"orm\":\"asset[2]\",\"emissive\":\"capture linear target\"}}}},\"diffuse\":{{\"id\":{:?},\"generation\":{},\"extent\":{:?}}},\"probes\":[{data}]}}\n",
            scene::EXTENT,
            format!("{:?}", input.producer.assets.geometry[0].id()),
            input.producer.assets.geometry[0].generation().get(),
            format!("{:?}", input.producer.assets.geometry[0].value().layout),
            format!("{:?}", input.producer.assets.materials[0].id()),
            input.producer.assets.materials[0].generation().get(),
            input.producer.sources,
            input.producer.models,
            scene::TINTS,
            format!("{:?}", input.assets.geometry[1].id()),
            input.assets.geometry[1].generation().get(),
            format!("{:?}", input.assets.geometry[1].value().layout),
            format!("{:?}", input.assets.materials[1].id()),
            input.assets.materials[1].generation().get(),
            format!("{:?}", input.producer.diffuse.id()),
            input.producer.diffuse.generation().get(),
            input.producer.diffuse.value().extent,
        ),
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/mixed_capture.rs"]
mod tests;
