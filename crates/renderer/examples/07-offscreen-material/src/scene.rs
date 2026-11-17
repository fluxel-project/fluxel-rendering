//! Material and receiver geometry for the 07 same-Graph capture consumer.
//!
//! `mirror_scene_template` is CPU-only preparation. It cannot allocate a
//! texture or submit commands, so it keeps Graph and device authority in the
//! forthcoming Renderer capture-product entry.

use std::collections::BTreeMap;

use fluxel_material::{
    BindingLocation, FragmentInput, MaterialCodeNode, MaterialDefinition, MaterialNode,
    MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderTransform,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::SourceUnit;

/// The Material resource name reserved for the Renderer-owned capture product.
pub const CAPTURE_TEXTURE_SLOT: &str = "captured_dragon";

/// CPU-only invariant inputs for the RHI 07 camera framing.
pub struct MirrorSceneTemplate {
    /// The exact ordinary texture slot that the capture entry must bind.
    pub capture_texture_slot: &'static str,
    /// Receiver geometry, retaining RHI 07's plane position/UV convention.
    pub mirror_geometry: GeometryData,
    /// The receiver Material that samples the capture product.
    pub mirror_material: MaterialDefinition,
    /// RHI 07's terminal camera transform (positive-Y renderer target form).
    pub camera_from_world: RenderTransform,
}

/// Prepares the mirror side of the two-pass scene without performing GPU work.
pub fn mirror_scene_template() -> MirrorSceneTemplate {
    MirrorSceneTemplate {
        capture_texture_slot: CAPTURE_TEXTURE_SLOT,
        mirror_geometry: mirror_geometry(),
        mirror_material: mirror_material(),
        camera_from_world: rhi07_camera(),
    }
}

/// Non-indexed receiver plane; the public Renderer geometry path is non-indexed.
pub fn mirror_geometry() -> GeometryData {
    let vertices: [([f32; 3], [f32; 2]); 6] = [
        ([-1.0, -1.0, 0.0], [0.0, 1.0]),
        ([1.0, -1.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0, 0.0], [1.0, 0.0]),
        ([-1.0, -1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0, 0.0], [1.0, 0.0]),
        ([-1.0, 1.0, 0.0], [0.0, 0.0]),
    ];
    let mut bytes = Vec::with_capacity(vertices.len() * 32);
    for (position, uv) in vertices {
        for value in position.into_iter().chain([0.0, 0.0, 1.0]).chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
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
        vertices: 0..6,
    }
}

/// Material-side declaration of the capture product. The Renderer must reject
/// an ordinary asset in this slot and bind only its same-frame private product.
pub fn mirror_material() -> MaterialDefinition {
    MaterialDefinition::new(
        Vec::new(),
        "sample_captured_dragon",
        vec![MaterialParameter { name: "mirror_tint".into(), ty: MaterialType::Vec4, resource: "mirror_tint".into() }],
        vec![
            MaterialResource { name: "mirror_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: CAPTURE_TEXTURE_SLOT.into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "capture_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "sample_captured_dragon".into(),
        inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new("renderer-07-capture-sample", "fn sample_captured_dragon(uv: vec2<f32>) -> vec4<f32> { return textureSample(captured_dragon, capture_sampler, uv) * mirror_tint; }"),
        resources: Vec::new(),
    })
    .with_node_graph(MaterialNodeGraph {
        nodes: vec![MaterialNode { id: MaterialNodeId(7), callable: "sample_captured_dragon".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }],
        output: MaterialNodeId(7),
    })
}

fn rhi07_camera() -> RenderTransform {
    RenderTransform::from_columns([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.999_048_23, -0.043_619_387, 0.0],
        [0.0, 0.043_619_387, 0.999_048_23, 0.0],
        [0.0, 1.0, -6.0, 1.0],
    ])
    .expect("RHI 07 camera matrix is finite")
}
