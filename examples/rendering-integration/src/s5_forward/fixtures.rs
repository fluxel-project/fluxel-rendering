//! Assets and independent fixture-input checks for the S5 Forward workload.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

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

pub(super) fn geometry_snapshot(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    data: GeometryData,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let handle = store.create()?;
    let byte_count = data.streams.iter().try_fold(0_u64, |total, stream| {
        let length = u64::try_from(stream.len())
            .map_err(|_| std::io::Error::other("geometry stream length does not fit u64"))?;
        total
            .checked_add(length)
            .ok_or_else(|| std::io::Error::other("geometry byte count overflow"))
    })?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(byte_count))?),
        _ => Err("new geometry asset did not yield a producer".into()),
    }
}

pub(super) fn material_snapshot(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    data: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(0))?),
        _ => Err("new material asset did not yield a producer".into()),
    }
}

pub(super) fn texture_snapshot(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    rgba: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let data = ColorTextureData {
        extent: [1, 1],
        rgba8: rgba.to_vec(),
    };
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(data, ResidentBytes::new(4))?),
        _ => Err("new texture asset did not yield a producer".into()),
    }
}

pub(super) fn textured_material(root: &str) -> MaterialDefinition {
    let body = if root == "textured_color_second" {
        // Green input becomes magenta; the independent image oracle expects
        // this material-code effect in addition to distinct texture bindings.
        "let c = textureLoad(material_texture, vec2<i32>(0, 0), 0); return vec4<f32>(vec3<f32>(1.0) - c.rgb, c.a) * tint;"
    } else {
        "return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint;"
    };
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: root.into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: format!("fn {root}() -> vec4<f32> {{ {body} }}"),
        }],
        root,
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        material_resources(),
        BTreeMap::from([("forward_sample_level".into(), StaticValue::U32(0))]),
        SurfaceOutput::Color,
    )
}

pub(super) fn position_only_material() -> MaterialDefinition {
    textured_material("position_only_color")
}

fn material_resources() -> Vec<MaterialResource> {
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
    ]
}

pub(super) fn interleaved_triangle() -> GeometryData {
    let vertices = triangle_vertices([-0.82, 0.20], [-0.38, 0.20], [-0.60, 0.76]);
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 36,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    attribute("POSITION", 0, VertexFormat::Float32(3)),
                    attribute("UV", 12, VertexFormat::Float32(2)),
                    attribute("COLOR", 20, VertexFormat::Float32(4)),
                ],
            }],
        },
        streams: vec![interleave(&vertices)],
        vertices: 0..3,
    }
}

pub(super) fn separate_triangle() -> GeometryData {
    let vertices = triangle_vertices([-0.28, 0.20], [0.16, 0.20], [-0.06, 0.76]);
    GeometryData {
        layout: GeometryLayout {
            streams: vec![
                stream("POSITION", VertexFormat::Float32(3), 12),
                stream("UV", VertexFormat::Float32(2), 8),
                stream("COLOR", VertexFormat::Float32(4), 16),
            ],
        },
        streams: vec![positions(&vertices), uvs(&vertices), colors(&vertices)],
        vertices: 0..3,
    }
}

pub(super) fn mixed_triangle() -> GeometryData {
    let vertices = triangle_vertices([0.26, 0.20], [0.70, 0.20], [0.48, 0.76]);
    GeometryData {
        layout: GeometryLayout {
            streams: vec![
                GeometryStream {
                    stride: 20,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        attribute("POSITION", 0, VertexFormat::Float32(3)),
                        attribute("UV", 12, VertexFormat::Float32(2)),
                    ],
                },
                stream("COLOR", VertexFormat::Float32(4), 16),
            ],
        },
        streams: vec![position_uvs(&vertices), colors(&vertices)],
        vertices: 0..3,
    }
}

pub(super) fn position_triangle() -> GeometryData {
    let vertices = triangle_vertices([-0.24, -0.76], [0.24, -0.76], [0.0, -0.20]);
    GeometryData {
        layout: GeometryLayout {
            streams: vec![stream("POSITION", VertexFormat::Float32(3), 12)],
        },
        streams: vec![positions(&vertices)],
        vertices: 0..3,
    }
}

fn stream(semantic: &str, format: VertexFormat, stride: u64) -> GeometryStream {
    GeometryStream {
        stride,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![attribute(semantic, 0, format)],
    }
}

fn attribute(semantic: &str, offset: u64, format: VertexFormat) -> GeometryAttribute {
    GeometryAttribute {
        semantic: semantic.into(),
        offset,
        format,
    }
}

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

fn triangle_vertices(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [Vertex; 3] {
    [
        Vertex {
            position: [a[0], a[1], 0.0],
            uv: [0.0; 2],
            color: [1.0; 4],
        },
        Vertex {
            position: [b[0], b[1], 0.0],
            uv: [0.0; 2],
            color: [1.0; 4],
        },
        Vertex {
            position: [c[0], c[1], 0.0],
            uv: [0.0; 2],
            color: [1.0; 4],
        },
    ]
}

fn positions(vertices: &[Vertex; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(36);
    for vertex in vertices {
        floats(&mut bytes, &vertex.position);
    }
    bytes
}

fn uvs(vertices: &[Vertex; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24);
    for vertex in vertices {
        floats(&mut bytes, &vertex.uv);
    }
    bytes
}

fn colors(vertices: &[Vertex; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48);
    for vertex in vertices {
        floats(&mut bytes, &vertex.color);
    }
    bytes
}

fn position_uvs(vertices: &[Vertex; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(60);
    for vertex in vertices {
        floats(&mut bytes, &vertex.position);
        floats(&mut bytes, &vertex.uv);
    }
    bytes
}

fn interleave(vertices: &[Vertex; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(108);
    for vertex in vertices {
        floats(&mut bytes, &vertex.position);
        floats(&mut bytes, &vertex.uv);
        floats(&mut bytes, &vertex.color);
    }
    bytes
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

pub(super) fn write_input_artifacts(
    output: &Path,
    geometry: &[AssetSnapshot<GeometryAsset, GeometryData>; 4],
    material: &[AssetSnapshot<MaterialAsset, MaterialDefinition>; 4],
    texture: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>; 4],
) -> Result<(), Box<dyn Error>> {
    for (index, snapshot) in geometry.iter().enumerate() {
        for (stream, bytes) in snapshot.value().streams.iter().enumerate() {
            fs::write(
                output.join(format!("s5-forward-geometry-{index}-stream-{stream}.bin")),
                bytes,
            )?;
        }
        fs::write(
            output.join(format!("s5-forward-geometry-{index}.txt")),
            format!("{:#?}\n", snapshot.value().layout),
        )?;
    }
    for (index, snapshot) in material.iter().enumerate() {
        fs::write(
            output.join(format!("s5-forward-material-{index}.txt")),
            format!("{:#?}\n", snapshot.value()),
        )?;
    }
    for (index, snapshot) in texture.iter().enumerate() {
        fs::write(
            output.join(format!("s5-forward-texture-{index}.rgba")),
            &snapshot.value().rgba8,
        )?;
    }
    Ok(())
}

pub(super) fn verify_fixture_inputs(geometry: &[GeometryData; 4]) -> Result<(), Box<dyn Error>> {
    let expected = [
        [[-0.82, 0.20, 0.0], [-0.38, 0.20, 0.0], [-0.60, 0.76, 0.0]],
        [[-0.28, 0.20, 0.0], [0.16, 0.20, 0.0], [-0.06, 0.76, 0.0]],
        [[0.26, 0.20, 0.0], [0.70, 0.20, 0.0], [0.48, 0.76, 0.0]],
        [[-0.24, -0.76, 0.0], [0.24, -0.76, 0.0], [0.0, -0.20, 0.0]],
    ];
    for (index, (actual, expected)) in geometry.iter().zip(expected).enumerate() {
        if positions_from_geometry(actual)? != expected {
            return Err(
                format!("S5 geometry input {index} differs from the frozen image oracle").into(),
            );
        }
    }
    Ok(())
}

fn positions_from_geometry(data: &GeometryData) -> Result<[[f32; 3]; 3], Box<dyn Error>> {
    if data.vertices != (0..3) || data.streams.len() != data.layout.streams.len() {
        return Err("S5 geometry fixture has an unexpected stream or vertex range".into());
    }
    let Some((stream_index, (_, attribute))) = data
        .layout
        .streams
        .iter()
        .enumerate()
        .flat_map(|(stream_index, stream)| {
            stream
                .attributes
                .iter()
                .map(move |attribute| (stream_index, (stream, attribute)))
        })
        .find(|(_, (_, attribute))| attribute.semantic == "POSITION")
    else {
        return Err("S5 geometry fixture has no POSITION attribute".into());
    };
    if !matches!(attribute.format, VertexFormat::Float32(3)) {
        return Err("S5 geometry fixture POSITION is not Float32x3".into());
    }
    let stream = &data.layout.streams[stream_index];
    let bytes = &data.streams[stream_index];
    let mut result = [[0.0; 3]; 3];
    for (vertex, position) in result.iter_mut().enumerate() {
        let offset = u64::try_from(vertex)?
            .checked_mul(stream.stride)
            .and_then(|offset| offset.checked_add(attribute.offset))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or("S5 geometry fixture position offset overflow")?;
        let values = bytes
            .get(offset..offset + 12)
            .ok_or("S5 geometry fixture position payload is truncated")?;
        for (component, output) in position.iter_mut().enumerate() {
            *output = f32::from_le_bytes(
                values[component * 4..component * 4 + 4]
                    .try_into()
                    .expect("fixed four-byte component slice"),
            );
        }
    }
    Ok(result)
}
