//! An external typed quad-marker Forward feature and its composed pipeline.
//!
//! The application seeds the returned CPU asset data in its own stores, then
//! passes exact `SceneAssetRef`s in [`QuadAssets`].  This module deliberately
//! has no device, Graph, encoder, or submission ownership.

use std::collections::{BTreeMap, HashSet};

use fluxel_material::{
    BindingLocation, MaterialDefinition, MaterialFragment, MaterialParameter, MaterialResource,
    MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    validate_context, BoundsVisibility, FeatureContext, FeatureError, FeatureHandle,
    ForwardCompositionContext, ForwardCpuPipeline, ForwardDrawOrder, ForwardDrawRequest,
    ForwardFeaturePass, ForwardResources, FramePipeline, FrameSnapshot, FrameTopology,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, PipelineError, PreparedFrame,
    RenderBounds, RenderFeature, RenderTransform, RenderViewId, SceneAssetRef,
    TypedFeatureSnapshot,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

/// The exact immutable assets used by one quad marker.
#[derive(Clone, Debug)]
pub struct QuadAssets {
    /// Quad vertex data generation.
    pub geometry: SceneAssetRef<fluxel_renderer::GeometryAsset>,
    /// Normal Material compiler definition generation.
    pub material: SceneAssetRef<fluxel_material::MaterialAsset>,
    /// Sampled color texture generation.
    pub texture: SceneAssetRef<fluxel_renderer::ColorTextureAsset>,
}

/// One application-owned marker expressed in world-space quad terms.
#[derive(Clone, Debug)]
pub struct QuadMarker {
    /// Identity unique within the enclosing [`QuadInput`] frame payload.
    pub id: u64,
    /// Quad centre in world space.
    pub center: [f32; 3],
    /// Positive x/y half-size in world units.
    pub half_extent: [f32; 2],
    /// Dynamic group-0 tint.
    pub tint: [f32; 4],
    /// Exact immutable assets selected by this marker.
    pub assets: QuadAssets,
    /// Explicit layer shared with built-in mesh draws.
    pub order: i64,
}

/// Typed payload for [`QuadFeature`].
#[derive(Clone, Debug, Default)]
pub struct QuadInput {
    /// Markers supplied for one scene-version-pinned feature snapshot.
    pub markers: Vec<QuadMarker>,
}

/// CPU result retained through graph declaration by [`QuadFeature`].
#[derive(Clone, Debug, Default)]
pub struct PreparedQuads {
    draws: Vec<ForwardDrawRequest>,
}

impl PreparedQuads {
    /// Returns the feature-owned Forward requests selected for this frame.
    pub fn draws(&self) -> &[ForwardDrawRequest] {
        &self.draws
    }
}

/// Statically selected external marker feature with runtime enablement.
#[derive(Clone, Debug)]
pub struct QuadFeature {
    handle: FeatureHandle,
    enabled: bool,
}

impl QuadFeature {
    /// Issues a scene-bound feature handle without allocating GPU resources.
    pub fn new(issuer: &fluxel_renderer::FeatureIssuer) -> Result<Self, FeatureError> {
        Ok(Self {
            handle: issuer.issue("external-quad-markers")?,
            enabled: true,
        })
    }

    /// Enables or disables marker production for subsequent preparations.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether this feature currently produces visible markers.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl RenderFeature for QuadFeature {
    type Input = QuadInput;
    type Prepared = PreparedQuads;

    fn handle(&self) -> FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
        _frame: &PreparedFrame<'_, ForwardResources>,
        visibility: RenderBounds,
    ) -> Result<Self::Prepared, FeatureError> {
        validate_context(context, input, self.handle)?;
        if !self.enabled {
            return Ok(PreparedQuads::default());
        }

        let mut draws = Vec::with_capacity(input.data().markers.len());
        let mut identities = HashSet::with_capacity(input.data().markers.len());
        for marker in &input.data().markers {
            if !identities.insert(marker.id) {
                return Err(FeatureError::InvalidPreparedData {
                    feature: "external-quad-markers",
                    detail: "marker IDs must be unique within one payload",
                });
            }
            validate_marker(marker)?;
            if !quad_intersects(marker, visibility) {
                continue;
            }
            draws.push(ForwardDrawRequest {
                local_id: marker.id,
                order: marker.order,
                resources: ForwardResources {
                    geometry: marker.assets.geometry,
                    material: marker.assets.material,
                    texture: marker.assets.texture,
                    tint: marker.tint,
                },
                transform: quad_transform(marker.center, marker.half_extent),
            });
        }
        Ok(PreparedQuads { draws })
    }

    fn draws<'a>(&self, prepared: &'a Self::Prepared) -> &'a [ForwardDrawRequest] {
        prepared.draws()
    }

    fn contribute(
        &self,
        _prepared: &Self::Prepared,
        pass: &mut ForwardFeaturePass<'_, '_>,
    ) -> Result<(), PipelineError> {
        pass.read_vertex_streams()?;
        pass.read_frame_uniforms()?;
        pass.read_sampled_textures()
    }
}

/// An external pipeline that composes mesh and quad draws in one opaque pass.
#[derive(Clone, Debug, Default)]
pub struct ExternalComposedPipeline(ForwardCpuPipeline<BoundsVisibility>);

impl ExternalComposedPipeline {
    /// Creates the composed external Forward policy.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FramePipeline<ForwardResources> for ExternalComposedPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<ForwardResources>,
        view: RenderViewId,
        visibility: RenderBounds,
    ) -> Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        self.0.prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        frame: &PreparedFrame<'_, ForwardResources>,
    ) -> Result<FrameTopology, PipelineError> {
        self.0.topology(frame)
    }

    fn order_forward(
        &self,
        _frame: &PreparedFrame<'_, ForwardResources>,
        draws: &mut [ForwardDrawOrder],
    ) -> Result<(), PipelineError> {
        // The public total key is the explicit cross-source policy. It first
        // orders application layers, then the renderer's namespace-aware
        // source identity, so a mesh and a marker never alias on a numeric ID.
        draws.sort_by_key(|draw| draw.sort_key());
        Ok(())
    }

    fn contribute_composed<B: fluxel_rhi::backend::ResourceBackend, F: RenderFeature>(
        &self,
        _frame: &PreparedFrame<'_, ForwardResources>,
        context: &mut ForwardCompositionContext<'_, '_, '_, B, F>,
    ) -> Result<(), PipelineError> {
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.meshes(|meshes| {
                meshes.read_vertex_streams()?;
                meshes.read_frame_uniforms()?;
                meshes.read_sampled_textures()
            })?;
            pass.feature()?;
            pass.write_main_color()
        })
    }
}

/// Returns interleaved position/UV/color bytes for a unit quad centered at zero.
pub fn quad_geometry_data() -> GeometryData {
    let vertices: [([f32; 3], [f32; 2]); 6] = [
        ([-1.0, -1.0, 0.0], [0.0, 1.0]),
        ([1.0, -1.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0, 0.0], [1.0, 0.0]),
        ([-1.0, -1.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0, 0.0], [1.0, 0.0]),
        ([-1.0, 1.0, 0.0], [0.0, 0.0]),
    ];
    let mut bytes = Vec::with_capacity(vertices.len() * 36);
    for (position, uv) in vertices {
        for value in position.into_iter().chain(uv).chain([1.0; 4]) {
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
        vertices: 0..6,
    }
}

/// Returns a normal Material definition whose fragment visibly remaps tint.
pub fn quad_marker_material_definition() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "quad_marker_color".into(),
            inputs: vec![],
            output: MaterialType::Vec4,
            calls: vec![],
            // This is intentionally unlike the built-in `return tint`: it
            // samples the declared texture, swaps red/blue, scales/biases RGB, and
            // leaves alpha untouched. The texture read also gives group 0,
            // binding 1 real fragment-stage visibility in the final Shader.
            // Binary-exact coefficients avoid UNORM half-step color ties.
            wgsl: "fn quad_marker_color() -> vec4<f32> { let remapped = vec4<f32>(tint.bgr * 0.75 + vec3<f32>(0.125), tint.a); return textureLoad(quad_texture, vec2<i32>(0, 0), 0) * remapped; }".into(),
        }],
        "quad_marker_color",
        vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "quad_tint".into() }],
        vec![
            MaterialResource {
                name: "quad_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "quad_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
}

fn validate_marker(marker: &QuadMarker) -> Result<(), FeatureError> {
    if !marker
        .center
        .iter()
        .chain(marker.half_extent.iter())
        .chain(marker.tint.iter())
        .all(|value| value.is_finite())
        || marker.half_extent.iter().any(|value| *value <= 0.0)
    {
        return Err(FeatureError::InvalidPreparedData {
            feature: "external-quad-markers",
            detail: "marker values must be finite with positive half extent",
        });
    }
    Ok(())
}

fn quad_intersects(marker: &QuadMarker, volume: RenderBounds) -> bool {
    // This is feature-local visibility, not the built-in BoundsVisibility:
    // a marker is a z-flat footprint and touching a volume face stays visible.
    let minimum = [
        marker.center[0] - marker.half_extent[0],
        marker.center[1] - marker.half_extent[1],
        marker.center[2],
    ];
    let maximum = [
        marker.center[0] + marker.half_extent[0],
        marker.center[1] + marker.half_extent[1],
        marker.center[2],
    ];
    (0..3).all(|axis| {
        minimum[axis] <= volume.maximum()[axis] && volume.minimum()[axis] <= maximum[axis]
    })
}

fn quad_transform(center: [f32; 3], half_extent: [f32; 2]) -> RenderTransform {
    RenderTransform::from_columns([
        [half_extent[0], 0.0, 0.0, 0.0],
        [0.0, half_extent[1], 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [center[0], center[1], center[2], 1.0],
    ])
    .expect("validated quad values construct a finite transform")
}
