//! Strict CPU-only decoder for the embedded glTF geometry used by example 07.
//!
//! It supports only data-URI buffers, non-sparse `TRIANGLES` primitives, and
//! float position/normal/UV attributes. Every range is checked before public
//! `GeometryData` is emitted.

use std::{error::Error, fmt};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use fluxel_renderer::{GeometryAttribute, GeometryData, GeometryLayout, GeometryStream};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use serde_json::{Map, Value};

#[path = "gltf/transform.rs"]
mod transform;
use transform::{Transform, node_transform};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GltfError(String);

impl GltfError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GltfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
impl Error for GltfError {}

/// Decodes all primitives into Renderer’s non-indexed position/normal/UV ABI.
///
/// `POSITION` and `NORMAL` are required. The first dragon primitive lacks
/// `TEXCOORD_0`, so only genuinely absent UVs default to `(0, 0)`; malformed
/// UV data always rejects the source.
pub fn load(bytes: &[u8]) -> Result<GeometryData, GltfError> {
    let document: Value = serde_json::from_slice(bytes)
        .map_err(|error| GltfError::new(format!("invalid glTF JSON: {error}")))?;
    let root = object(&document, "document")?;
    require_version(root)?;
    let buffers = decode_buffers(root)?;
    let accessors = array_map(root, "accessors")?;
    let views = array_map(root, "bufferViews")?;
    let meshes = array_map(root, "meshes")?;
    let nodes = array_map(root, "nodes")?;
    let mut bytes = Vec::new();
    for node_index in active_scene_nodes(root)? {
        let node = nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("active scene node is out of range"))?;
        if node.get("children").is_some() || node.get("matrix").is_some() {
            return Err(GltfError::new(
                "only flat node TRS transforms are supported",
            ));
        }
        let Some(mesh_index) = node.get("mesh") else {
            continue;
        };
        let mesh = meshes
            .get(value_index(mesh_index, "node mesh")?)
            .ok_or_else(|| GltfError::new("node mesh is out of range"))?;
        let transform = node_transform(node)?;
        for primitive in array(mesh, "primitives")? {
            append_primitive(primitive, accessors, views, &buffers, transform, &mut bytes)?;
        }
    }
    let count = u32::try_from(bytes.len() / 32)
        .map_err(|_| GltfError::new("expanded vertex count exceeds Renderer range"))?;
    if count == 0 {
        return Err(GltfError::new("glTF has no triangle vertices"));
    }
    Ok(GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 32,
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
                    GeometryAttribute {
                        semantic: "UV0".into(),
                        offset: 24,
                        format: VertexFormat::Float32(2),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..count,
    })
}

fn append_primitive(
    primitive: &Value,
    accessors: &[Value],
    views: &[Value],
    buffers: &[Vec<u8>],
    transform: Transform,
    output: &mut Vec<u8>,
) -> Result<(), GltfError> {
    if let Some(mode) = primitive.get("mode") {
        if number(mode, "primitive mode")? != 4 {
            return Err(GltfError::new("only TRIANGLES primitives are supported"));
        }
    }
    let attributes = object(required(primitive, "attributes")?, "primitive attributes")?;
    let positions = read_f32x3(
        accessors,
        views,
        buffers,
        required_index(attributes, "POSITION")?,
    )?;
    let normals = read_f32x3(
        accessors,
        views,
        buffers,
        required_index(attributes, "NORMAL")?,
    )?;
    if positions.len() != normals.len() {
        return Err(GltfError::new("POSITION and NORMAL counts differ"));
    }
    let uvs = match attributes.get("TEXCOORD_0") {
        Some(value) => Some(read_f32x2(
            accessors,
            views,
            buffers,
            value_index(value, "TEXCOORD_0")?,
        )?),
        None => None,
    };
    if uvs
        .as_ref()
        .is_some_and(|values| values.len() != positions.len())
    {
        return Err(GltfError::new("TEXCOORD_0 count differs from POSITION"));
    }
    let indices = match primitive.get("indices") {
        Some(value) => read_indices(accessors, views, buffers, value_index(value, "indices")?)?,
        None => (0..positions.len())
            .map(|index| u32::try_from(index).map_err(|_| GltfError::new("vertex index overflow")))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if indices.len() % 3 != 0 {
        return Err(GltfError::new(
            "TRIANGLES index count is not divisible by three",
        ));
    }
    for index in indices {
        let index =
            usize::try_from(index).map_err(|_| GltfError::new("index conversion overflow"))?;
        let position = transform.position(
            *positions
                .get(index)
                .ok_or_else(|| GltfError::new("index exceeds POSITION count"))?,
        );
        let normal = transform.normal(
            *normals
                .get(index)
                .ok_or_else(|| GltfError::new("index exceeds NORMAL count"))?,
        );
        let uv = uvs.as_ref().map_or([0.0, 0.0], |values| values[index]);
        for value in position.iter().chain(normal.iter()).chain(&uv) {
            if !value.is_finite() {
                return Err(GltfError::new(
                    "geometry attribute contains non-finite value",
                ));
            }
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn active_scene_nodes(root: &Map<String, Value>) -> Result<Vec<usize>, GltfError> {
    let scene_index = usize_number(required_map(root, "scene")?, "active scene")?;
    let scene = array_map(root, "scenes")?
        .get(scene_index)
        .ok_or_else(|| GltfError::new("active scene is out of range"))?;
    let nodes = array(scene, "nodes")?;
    if nodes.is_empty() {
        return Err(GltfError::new("active scene has no root nodes"));
    }
    nodes
        .iter()
        .map(|node| value_index(node, "scene node"))
        .collect()
}

fn decode_buffers(root: &Map<String, Value>) -> Result<Vec<Vec<u8>>, GltfError> {
    array_map(root, "buffers")?
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            let declared = usize_number(required(buffer, "byteLength")?, "buffer byteLength")?;
            let uri = string(required(buffer, "uri")?, "buffer uri")?;
            let (header, payload) = uri
                .strip_prefix("data:")
                .and_then(|value| value.split_once(','))
                .ok_or_else(|| GltfError::new(format!("buffer {index} must use a data URI")))?;
            if !header.ends_with(";base64") {
                return Err(GltfError::new(format!(
                    "buffer {index} data URI is not base64"
                )));
            }
            let bytes = STANDARD
                .decode(payload)
                .map_err(|_| GltfError::new(format!("buffer {index} base64 decode failed")))?;
            if bytes.len() != declared {
                return Err(GltfError::new(format!(
                    "buffer {index} byteLength differs from decoded bytes"
                )));
            }
            Ok(bytes)
        })
        .collect()
}

fn read_f32x3(
    accessors: &[Value],
    views: &[Value],
    buffers: &[Vec<u8>],
    index: usize,
) -> Result<Vec<[f32; 3]>, GltfError> {
    let raw = raw_accessor(accessors, views, buffers, index)?;
    if raw.component_type != 5126 || raw.components != 3 {
        return Err(GltfError::new("vertex attribute must be FLOAT VEC3"));
    }
    (0..raw.count)
        .map(|element| {
            let bytes = raw.element(element)?;
            Ok([f32_at(bytes, 0)?, f32_at(bytes, 4)?, f32_at(bytes, 8)?])
        })
        .collect()
}

fn read_f32x2(
    accessors: &[Value],
    views: &[Value],
    buffers: &[Vec<u8>],
    index: usize,
) -> Result<Vec<[f32; 2]>, GltfError> {
    let raw = raw_accessor(accessors, views, buffers, index)?;
    if raw.component_type != 5126 || raw.components != 2 {
        return Err(GltfError::new("TEXCOORD_0 must be FLOAT VEC2"));
    }
    (0..raw.count)
        .map(|element| {
            let bytes = raw.element(element)?;
            Ok([f32_at(bytes, 0)?, f32_at(bytes, 4)?])
        })
        .collect()
}

fn read_indices(
    accessors: &[Value],
    views: &[Value],
    buffers: &[Vec<u8>],
    index: usize,
) -> Result<Vec<u32>, GltfError> {
    let raw = raw_accessor(accessors, views, buffers, index)?;
    if raw.components != 1 {
        return Err(GltfError::new("index accessor must be SCALAR"));
    }
    (0..raw.count)
        .map(|element| {
            let bytes = raw.element(element)?;
            match raw.component_type {
                5121 => Ok(u32::from(bytes[0])),
                5123 => Ok(u32::from(u16::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("invalid u16 index"))?,
                ))),
                5125 => Ok(u32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| GltfError::new("invalid u32 index"))?,
                )),
                _ => Err(GltfError::new(
                    "indices must be unsigned byte, short, or int",
                )),
            }
        })
        .collect()
}

struct RawAccessor<'a> {
    bytes: &'a [u8],
    start: usize,
    stride: usize,
    count: usize,
    components: usize,
    component_type: u64,
    component_size: usize,
}
impl RawAccessor<'_> {
    fn element(&self, element: usize) -> Result<&[u8], GltfError> {
        let offset = self
            .start
            .checked_add(
                element
                    .checked_mul(self.stride)
                    .ok_or_else(|| GltfError::new("accessor element offset overflows"))?,
            )
            .ok_or_else(|| GltfError::new("accessor element offset overflows"))?;
        let length = self
            .components
            .checked_mul(self.component_size)
            .ok_or_else(|| GltfError::new("accessor element size overflows"))?;
        self.bytes
            .get(
                offset
                    ..offset
                        .checked_add(length)
                        .ok_or_else(|| GltfError::new("accessor end overflows"))?,
            )
            .ok_or_else(|| GltfError::new("accessor data is truncated"))
    }
}

fn raw_accessor<'a>(
    accessors: &'a [Value],
    views: &'a [Value],
    buffers: &'a [Vec<u8>],
    index: usize,
) -> Result<RawAccessor<'a>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("accessor index is out of range"))?;
    if accessor.get("sparse").is_some() {
        return Err(GltfError::new("sparse accessors are unsupported"));
    }
    let view = views
        .get(value_index(
            required(accessor, "bufferView")?,
            "bufferView",
        )?)
        .ok_or_else(|| GltfError::new("bufferView index is out of range"))?;
    let buffer = buffers
        .get(usize_number(
            required(view, "buffer")?,
            "bufferView buffer",
        )?)
        .ok_or_else(|| GltfError::new("bufferView selects missing buffer"))?;
    let view_offset = optional_usize(view, "byteOffset")?;
    let view_length = usize_number(required(view, "byteLength")?, "bufferView byteLength")?;
    let view_end = view_offset
        .checked_add(view_length)
        .ok_or_else(|| GltfError::new("bufferView range overflows"))?;
    if view_end > buffer.len() {
        return Err(GltfError::new("bufferView exceeds decoded buffer"));
    }
    let components = components(string(required(accessor, "type")?, "accessor type")?)?;
    let component_type = number(
        required(accessor, "componentType")?,
        "accessor componentType",
    )?;
    let component_size = component_size(component_type)?;
    let packed = components
        .checked_mul(component_size)
        .ok_or_else(|| GltfError::new("packed accessor size overflows"))?;
    let stride = match view.get("byteStride") {
        Some(value) => usize_number(value, "bufferView byteStride")?,
        None => packed,
    };
    if stride < packed || stride > 252 || stride % component_size != 0 {
        return Err(GltfError::new(
            "bufferView byteStride is invalid for accessor",
        ));
    }
    let accessor_offset = optional_usize(accessor, "byteOffset")?;
    let count = usize_number(required(accessor, "count")?, "accessor count")?;
    if count == 0 {
        return Err(GltfError::new("accessor count must be nonzero"));
    }
    let used = count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|last| last.checked_add(packed))
        .ok_or_else(|| GltfError::new("accessor range overflows"))?;
    let end = accessor_offset
        .checked_add(used)
        .ok_or_else(|| GltfError::new("accessor range overflows"))?;
    if end > view_length {
        return Err(GltfError::new("accessor exceeds bufferView range"));
    }
    Ok(RawAccessor {
        bytes: buffer,
        start: view_offset
            .checked_add(accessor_offset)
            .ok_or_else(|| GltfError::new("accessor start overflows"))?,
        stride,
        count,
        components,
        component_type,
        component_size,
    })
}

fn f32_at(bytes: &[u8], start: usize) -> Result<f32, GltfError> {
    Ok(f32::from_le_bytes(
        bytes
            .get(start..start + 4)
            .ok_or_else(|| GltfError::new("invalid float attribute"))?
            .try_into()
            .map_err(|_| GltfError::new("invalid float attribute"))?,
    ))
}
fn require_version(root: &Map<String, Value>) -> Result<(), GltfError> {
    let asset = object(required_map(root, "asset")?, "asset")?;
    if string(required_map(asset, "version")?, "asset version")? == "2.0" {
        Ok(())
    } else {
        Err(GltfError::new("only glTF 2.0 is supported"))
    }
}
fn array<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], GltfError> {
    value
        .get(name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| GltfError::new(format!("missing {name} array")))
}
fn array_map<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a [Value], GltfError> {
    map.get(name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| GltfError::new(format!("missing {name} array")))
}
fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, GltfError> {
    value
        .as_object()
        .ok_or_else(|| GltfError::new(format!("{name} must be an object")))
}
fn required<'a>(value: &'a Value, name: &str) -> Result<&'a Value, GltfError> {
    value
        .get(name)
        .ok_or_else(|| GltfError::new(format!("missing {name}")))
}
fn required_map<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a Value, GltfError> {
    map.get(name)
        .ok_or_else(|| GltfError::new(format!("missing {name}")))
}
fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, GltfError> {
    value
        .as_str()
        .ok_or_else(|| GltfError::new(format!("{name} must be a string")))
}
fn number(value: &Value, name: &str) -> Result<u64, GltfError> {
    value
        .as_u64()
        .ok_or_else(|| GltfError::new(format!("{name} must be an unsigned integer")))
}
fn usize_number(value: &Value, name: &str) -> Result<usize, GltfError> {
    usize::try_from(number(value, name)?)
        .map_err(|_| GltfError::new(format!("{name} exceeds platform range")))
}
fn optional_usize(value: &Value, name: &str) -> Result<usize, GltfError> {
    match value.get(name) {
        Some(entry) => usize_number(entry, name),
        None => Ok(0),
    }
}
fn value_index(value: &Value, name: &str) -> Result<usize, GltfError> {
    usize_number(value, name)
}
fn required_index(map: &Map<String, Value>, name: &str) -> Result<usize, GltfError> {
    value_index(
        map.get(name)
            .ok_or_else(|| GltfError::new(format!("missing {name} attribute")))?,
        name,
    )
}
fn components(kind: &str) -> Result<usize, GltfError> {
    match kind {
        "SCALAR" => Ok(1),
        "VEC2" => Ok(2),
        "VEC3" => Ok(3),
        "VEC4" => Ok(4),
        _ => Err(GltfError::new("accessor type must be scalar or vector")),
    }
}
fn component_size(kind: u64) -> Result<usize, GltfError> {
    match kind {
        5121 => Ok(1),
        5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(GltfError::new("unsupported accessor component type")),
    }
}
