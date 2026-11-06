//! Private S8 asset constructors copied from the S5 Forward fixture shape.

use std::{collections::BTreeMap, error::Error};

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

pub(super) fn tiled_triangle() -> GeometryData {
    let mut bytes = Vec::with_capacity(108);
    for position in [
        [-0.92_f32, -0.92, 0.0],
        [0.92, -0.92, 0.0],
        [-0.92, 0.92, 0.0],
    ] {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0_f32, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1.0_f32; 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 36,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV".into(),
                        offset: 12,
                        format: VertexFormat::Float32(2),
                    },
                    GeometryAttribute {
                        semantic: "COLOR".into(),
                        offset: 20,
                        format: VertexFormat::Float32(4),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    }
}

pub(super) fn material(variant: usize) -> MaterialDefinition {
    let root = format!("s8_color_variant_{variant}");
    let body = if variant == 3 {
        "let c = textureLoad(material_texture, vec2<i32>(0, 0), 0); return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a) * tint;"
    } else {
        "return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint;"
    };
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: root.clone(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: format!("fn {root}() -> vec4<f32> {{ {body} }}"),
        }],
        &root,
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
        ],
        BTreeMap::from([("forward_sample_level".into(), StaticValue::U32(0))]),
        SurfaceOutput::Color,
    )
}

pub(super) fn geometry_snapshot(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    data: GeometryData,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    let bytes = u64::try_from(data.streams.iter().map(Vec::len).sum::<usize>())?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(bytes))?),
        _ => Err("S8 geometry producer missing".into()),
    }
}

pub(super) fn material_snapshot(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    data: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(0))?),
        _ => Err("S8 material producer missing".into()),
    }
}

pub(super) fn texture_snapshot(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    rgba: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(
            ColorTextureData {
                extent: [1, 1],
                rgba8: rgba.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("S8 texture producer missing".into()),
    }
}
