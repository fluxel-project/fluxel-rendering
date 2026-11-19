//! Complete material authoring definitions.

use std::collections::BTreeMap;

use super::{
    AmbientOcclusionConfiguration, DiffuseRoughnessConfiguration, MaterialCodeNode,
    MaterialFragment, MaterialNodeGraph, MaterialNodeInputSource, MaterialParameter,
    MaterialResource, MaterialTextureEncoding, MaterialTextureEncodingRequirement,
    NormalMapSettings, RefractionEnvironment, StaticValue, SurfacePolicy, TransmissionExtensions,
    TransmissionTextureConfiguration,
};

/// The selected surface output for the minimal material contribution.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SurfaceOutput {
    /// A fragment-color `vec4<f32>` output.
    Color,
    /// A UV0-evaluated fragment-color `vec4<f32>` output.
    ///
    /// This remains an unlit color surface. It does not select a PBR
    /// metallic-roughness evaluation or imply that a Deferred G-buffer route
    /// exists for the material.
    ColorUv0,
    /// A base color plus metallic and perceptual-roughness property callable.
    MetallicRoughness {
        /// Zero-argument `vec2<f32>` callable yielding metallic then roughness.
        properties: String,
    },
    /// A UV0-evaluated base color plus metallic and roughness properties.
    MetallicRoughnessUv0 {
        /// One-argument vec2<f32> callable yielding metallic then roughness.
        properties: String,
    },
    /// A UV0-evaluated metallic-roughness surface with a tangent-space normal.
    MetallicRoughnessUv0Normal {
        /// One-argument vec2<f32> callable yielding metallic then roughness.
        properties: String,
        /// One-argument vec2<f32> callable yielding a tangent-space normal.
        normal: String,
    },
    /// A UV0 PBR surface whose base color is the node-graph root and whose
    /// metallic and roughness values are independently selected graph values.
    MetallicRoughnessUv0Graph {
        /// Node graph source yielding one scalar metallic value.
        metallic: MaterialNodeInputSource,
        /// Node graph source yielding one scalar perceptual-roughness value.
        roughness: MaterialNodeInputSource,
        /// Optional node graph source yielding one scalar occlusion value.
        ///
        /// This must resolve to `F32`. When absent, graph PBR evaluation uses
        /// the neutral unit occlusion value.
        occlusion: Option<MaterialNodeInputSource>,
        /// Optional UV0 tangent-space normal callable.
        normal: Option<String>,
        /// Optional graph source yielding a world-space pre-orientation normal.
        ///
        /// This must resolve to `Vec3` and cannot be combined with `normal`.
        /// Renderer applies its established orientation and sidedness rules
        /// after consuming this graph result.
        world_normal: Option<MaterialNodeInputSource>,
    },
}

/// Optional UV0 normal-surface evaluation channels.
///
/// The selected normal surface always emits stable occlusion and emissive
/// evaluators. Absent callables lower to neutral values: one for occlusion and
/// zero for emissive.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SurfaceExtensions {
    /// Optional `vec2<f32> -> f32` occlusion callable.
    pub occlusion: Option<String>,
    /// Optional `vec2<f32> -> vec3<f32>` emissive callable.
    ///
    /// This remains available for custom emissive evaluation. It cannot be
    /// combined with a definition-level [`EmissiveConfiguration`], which
    /// generates the standard color-texture-strength evaluator itself.
    pub emissive: Option<String>,
}

/// Standard emissive controls for a UV0 normal PBR surface.
///
/// This lowers to Babylon's PBR relation: color multiplied by sampled linear
/// RGB, texture level, and intensity. Missing parameter references lower to
/// neutral constants: black color and unit intensity. A missing texture leaves
/// no sampled term, so the result is color multiplied by intensity.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmissiveConfiguration {
    /// Optional `Vec3` material parameter naming the emissive color.
    pub color_parameter: Option<String>,
    /// Optional `F32` material parameter naming the emissive intensity.
    pub intensity_parameter: Option<String>,
    /// Selects the software sRGB decode used for an
    /// [`MaterialTextureEncoding::SrgbEncodedUnorm`] emissive texture.
    ///
    /// This has no effect for linear or hardware-sRGB texture encodings.
    pub srgb_encoded_unorm_decode: EmissiveSrgbDecode,
    /// Optional named texture, sampler, level, and transfer encoding.
    pub texture: Option<EmissiveTexture>,
}

/// Software sRGB decode selected for an encoded UNORM emissive texture.
///
/// Babylon uses [`Self::BabylonApproximate`] by default. Its engine-level
/// exact-conversion option selects [`Self::Exact`]. Fluxel keeps this choice
/// material-local until a renderer exposes a broader policy deliberately.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EmissiveSrgbDecode {
    /// Babylon's default approximation: `pow(rgb, 2.2)`.
    #[default]
    BabylonApproximate,
    /// The IEC sRGB piecewise transfer-function inverse.
    Exact,
}

/// One declared emissive texture binding.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmissiveTexture {
    /// Declared `Texture2dF32` resource sampled by the evaluator.
    pub texture: String,
    /// Declared non-comparison sampler resource paired with [`Self::texture`].
    pub sampler: String,
    /// Optional `F32` material parameter naming the texture level.
    pub level_parameter: Option<String>,
    /// Transfer encoding required of the bound texture asset.
    ///
    /// `Srgb` selects hardware sRGB decode before filtering, as on Babylon's
    /// sRGB texture path. `SrgbEncodedUnorm` instead samples ordinary UNORM
    /// bytes and performs the selected software decode after filtering.
    pub encoding: MaterialTextureEncoding,
}

/// One sampled base-color texture used by the standard PBR factor route.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PbrBaseTexture {
    /// Declared hardware-sRGB `Texture2dF32` resource.
    pub texture: String,
    /// Declared non-comparison sampler resource.
    pub sampler: String,
}

/// Mesh texture-coordinate set selected by one generated texture evaluator.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TextureUvSet {
    /// The first mesh texture-coordinate set.
    Uv0,
    /// The second mesh texture-coordinate set.
    Uv1,
}

/// One finite affine texture-coordinate transform for a generated texture use.
///
/// The three stored columns encode a homogeneous `mat3x2` transform from the
/// selected mesh UV set. Coefficients retain their exact IEEE-754 bit patterns
/// so variant identity distinguishes signed zero and other authored values.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaterialTextureUv {
    set: TextureUvSet,
    columns: [[u32; 2]; 3],
}

impl MaterialTextureUv {
    /// Creates an affine texture-coordinate transform when every coefficient is finite.
    #[must_use]
    pub fn new(set: TextureUvSet, columns: [[f32; 2]; 3]) -> Option<Self> {
        columns
            .iter()
            .flatten()
            .all(|value| value.is_finite())
            .then(|| Self {
                set,
                columns: columns.map(|column| column.map(f32::to_bits)),
            })
    }

    /// Returns the selected mesh texture-coordinate set.
    #[must_use]
    pub const fn set(self) -> TextureUvSet {
        self.set
    }

    /// Returns the authored affine transform coefficients.
    #[must_use]
    pub fn columns(self) -> [[f32; 2]; 3] {
        self.columns.map(|column| column.map(f32::from_bits))
    }

    pub(crate) const fn bits(self) -> [[u32; 2]; 3] {
        self.columns
    }
}

/// One channel of a packed PBR data texture.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PbrPackedChannel {
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Alpha channel.
    Alpha,
}

/// One linear packed metallic/roughness texture and its channel assignment.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PbrPackedMetallicRoughness {
    /// Declared linear `Texture2dF32` resource.
    pub texture: String,
    /// Declared non-comparison sampler resource.
    pub sampler: String,
    /// Channel multiplied by the authored metallic value and factor.
    pub metallic: PbrPackedChannel,
    /// Channel multiplied by the authored roughness value and factor.
    pub roughness: PbrPackedChannel,
    /// Optional channel mixed into authored occlusion.
    pub occlusion: Option<PbrPackedChannel>,
}

impl PbrPackedMetallicRoughness {
    /// Selects Babylon's traditional metallic-red, roughness-alpha packing.
    #[must_use]
    pub fn babylon_metallic_roughness(
        texture: impl Into<String>,
        sampler: impl Into<String>,
    ) -> Self {
        Self {
            texture: texture.into(),
            sampler: sampler.into(),
            metallic: PbrPackedChannel::Red,
            roughness: PbrPackedChannel::Alpha,
            occlusion: None,
        }
    }

    /// Selects the glTF ORM convention: occlusion-red, roughness-green and metallic-blue.
    #[must_use]
    pub fn gltf_orm(texture: impl Into<String>, sampler: impl Into<String>) -> Self {
        Self {
            texture: texture.into(),
            sampler: sampler.into(),
            metallic: PbrPackedChannel::Blue,
            roughness: PbrPackedChannel::Green,
            occlusion: Some(PbrPackedChannel::Red),
        }
    }
}

/// Optional standard PBR factors layered over the existing authored callables.
///
/// This is deliberately opt-in. The root and metallic/roughness callables
/// retain their established meanings, then this configuration multiplies their
/// values with authored factors and selected texture channels.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PbrSurfaceFactors {
    /// Optional `Vec4` parameter multiplied with the authored base value.
    pub base_factor_parameter: Option<String>,
    /// Optional hardware-sRGB base-color texture.
    pub base_texture: Option<PbrBaseTexture>,
    /// Optional `F32` parameter multiplied with authored metallic.
    pub metallic_factor_parameter: Option<String>,
    /// Optional `F32` parameter multiplied with authored perceptual roughness.
    pub roughness_factor_parameter: Option<String>,
    /// Optional linear packed metallic/roughness/occlusion texture.
    pub packed_metallic_roughness: Option<PbrPackedMetallicRoughness>,
    /// Optional `F32` weight for mixing packed occlusion into authored occlusion.
    pub occlusion_strength_parameter: Option<String>,
}

/// One material's validated, author-facing definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialDefinition {
    fragments: Vec<MaterialFragment>,
    code_nodes: Vec<MaterialCodeNode>,
    root_fragment: String,
    parameters: Vec<MaterialParameter>,
    resources: Vec<MaterialResource>,
    texture_encoding_requirements: Vec<MaterialTextureEncodingRequirement>,
    texture_uv: BTreeMap<String, MaterialTextureUv>,
    static_options: BTreeMap<String, StaticValue>,
    output: SurfaceOutput,
    surface_extensions: SurfaceExtensions,
    emissive_configuration: Option<EmissiveConfiguration>,
    pbr_surface_factors: Option<PbrSurfaceFactors>,
    pub(super) ambient_occlusion_configuration: Option<AmbientOcclusionConfiguration>,
    pub(super) diffuse_roughness_configuration: Option<DiffuseRoughnessConfiguration>,
    normal_map_settings: Option<NormalMapSettings>,
    surface_policy: SurfacePolicy,
    transmission_extensions: Option<TransmissionExtensions>,
    pub(super) transmission_texture_configuration: Option<TransmissionTextureConfiguration>,
    refraction_environment: Option<RefractionEnvironment>,
    node_graph: Option<MaterialNodeGraph>,
    vertex_uv0: Option<VertexUv0>,
    vertex_varyings: Vec<MaterialVertexVarying>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum VertexUv0 {
    Callable(String),
    Graph(MaterialNodeGraph),
}

/// One graph result authored for vertex interpolation into the surface graph.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaterialVertexVarying {
    /// Stable material-local varying identifier.
    pub id: u32,
    /// Typed vertex-graph node result selected for this varying.
    pub source: MaterialNodeInputSource,
}

impl MaterialDefinition {
    /// Creates a material definition without compiling it.
    #[must_use]
    pub fn new(
        fragments: Vec<MaterialFragment>,
        root_fragment: impl Into<String>,
        parameters: Vec<MaterialParameter>,
        resources: Vec<MaterialResource>,
        static_options: BTreeMap<String, StaticValue>,
        output: SurfaceOutput,
    ) -> Self {
        Self {
            fragments,
            code_nodes: Vec::new(),
            root_fragment: root_fragment.into(),
            parameters,
            resources,
            texture_encoding_requirements: Vec::new(),
            texture_uv: BTreeMap::new(),
            static_options,
            output,
            surface_extensions: SurfaceExtensions::default(),
            emissive_configuration: None,
            pbr_surface_factors: None,
            ambient_occlusion_configuration: None,
            diffuse_roughness_configuration: None,
            normal_map_settings: None,
            surface_policy: SurfacePolicy::default(),
            transmission_extensions: None,
            transmission_texture_configuration: None,
            refraction_environment: None,
            node_graph: None,
            vertex_uv0: None,
            vertex_varyings: Vec::new(),
        }
    }

    /// Adds one first-class author WGSL node to this material definition.
    #[must_use]
    pub fn with_code_node(mut self, node: MaterialCodeNode) -> Self {
        self.code_nodes.push(node);
        self
    }

    /// Selects a vertex-stage callable that transforms UV0 for the UV0 PBR node graph.
    ///
    /// The callable takes and returns a `vec2<f32>`. The fragment graph can consume
    /// the interpolated result through `MaterialNodeInputSource::VertexUv0`.
    #[must_use]
    pub fn with_vertex_uv0(mut self, callable: impl Into<String>) -> Self {
        self.vertex_uv0 = Some(VertexUv0::Callable(callable.into()));
        self
    }

    /// Selects a typed vertex graph that produces the interpolated UV0 result.
    ///
    /// The selected output node returns a `vec2<f32>`. The graph can connect
    /// mesh UV0 and other nodes from this vertex graph. This replaces any
    /// previously selected vertex UV0 callable or graph.
    #[must_use]
    pub fn with_vertex_uv0_graph(mut self, graph: MaterialNodeGraph) -> Self {
        self.vertex_uv0 = Some(VertexUv0::Graph(graph));
        self
    }

    /// Selects graph node results that the vertex stage interpolates to the
    /// fragment surface graph.
    ///
    /// This replaces any previously selected vertex varyings. Every source
    /// must be a node or named node output in the selected vertex graph from
    /// [`Self::with_vertex_uv0_graph`].
    #[must_use]
    pub fn with_vertex_varyings(mut self, varyings: Vec<MaterialVertexVarying>) -> Self {
        self.vertex_varyings = varyings;
        self
    }

    /// Selects explicit typed node connections as the surface-output root.
    #[must_use]
    pub fn with_node_graph(mut self, graph: MaterialNodeGraph) -> Self {
        self.node_graph = Some(graph);
        self
    }

    /// Selects optional occlusion and emissive evaluators for a UV0 normal surface.
    #[must_use]
    pub fn with_surface_extensions(mut self, extensions: SurfaceExtensions) -> Self {
        self.surface_extensions = extensions;
        self
    }

    /// Selects standard color-texture-strength emissive evaluation for a UV0
    /// normal PBR surface.
    #[must_use]
    pub fn with_emissive_configuration(mut self, configuration: EmissiveConfiguration) -> Self {
        self.emissive_configuration = Some(configuration);
        self
    }

    /// Selects standard base and packed metallic/roughness factor composition.
    #[must_use]
    pub fn with_pbr_surface_factors(mut self, factors: PbrSurfaceFactors) -> Self {
        self.pbr_surface_factors = Some(factors);
        self
    }

    /// Selects static strength and channel-convention controls for the
    /// tangent-space normal callable of this PBR surface.
    #[must_use]
    pub fn with_normal_map_settings(mut self, settings: NormalMapSettings) -> Self {
        self.normal_map_settings = Some(settings);
        self
    }

    /// Requires transfer encodings for selected texture resources when an
    /// asset source is bound by a renderer consumer.
    #[must_use]
    pub fn with_texture_encoding_requirements(
        mut self,
        requirements: Vec<MaterialTextureEncodingRequirement>,
    ) -> Self {
        self.texture_encoding_requirements = requirements;
        self
    }

    /// Selects one texture-coordinate set and affine transform for a generated
    /// built-in texture evaluator.
    ///
    /// Validation only admits names selected by standard PBR base or packed
    /// factors, or by the configured standard emissive texture.
    #[must_use]
    pub fn with_texture_uv(
        mut self,
        resource: impl Into<String>,
        coordinates: MaterialTextureUv,
    ) -> Self {
        self.texture_uv.insert(resource.into(), coordinates);
        self
    }

    /// Selects static alpha and raster facts for this material surface.
    #[must_use]
    pub fn with_surface_policy(mut self, policy: SurfacePolicy) -> Self {
        self.surface_policy = policy;
        self
    }

    /// Selects transmission and volume-attenuation controls for a UV0 normal PBR surface.
    #[must_use]
    pub fn with_transmission_extensions(mut self, extensions: TransmissionExtensions) -> Self {
        self.transmission_extensions = Some(extensions);
        self
    }

    /// Selects a material-local environment for transmission refraction.
    ///
    /// Transmission controls must also be selected. The named texture and
    /// sampler remain ordinary material resources; Renderer owns their source,
    /// residency, and Graph reads.
    #[must_use]
    pub fn with_refraction_environment(mut self, environment: RefractionEnvironment) -> Self {
        self.refraction_environment = Some(environment);
        self
    }

    /// Returns the explicitly requested optional surface evaluators.
    #[must_use]
    pub fn surface_extensions(&self) -> &SurfaceExtensions {
        &self.surface_extensions
    }

    /// Returns standard emissive evaluation controls, when selected.
    #[must_use]
    pub fn emissive_configuration(&self) -> Option<&EmissiveConfiguration> {
        self.emissive_configuration.as_ref()
    }

    /// Returns standard base and packed metallic/roughness controls, when selected.
    #[must_use]
    pub fn pbr_surface_factors(&self) -> Option<&PbrSurfaceFactors> {
        self.pbr_surface_factors.as_ref()
    }

    /// Returns tangent-space normal-map settings when explicitly selected.
    #[must_use]
    pub const fn normal_map_settings(&self) -> Option<NormalMapSettings> {
        self.normal_map_settings
    }

    /// Returns whether this material needs the renderer effects route.
    ///
    /// The effects route owns the stable occlusion and emissive evaluator ABI
    /// for UV0 normal PBR surfaces.
    #[must_use]
    pub const fn requires_surface_effects(&self) -> bool {
        self.surface_extensions.occlusion.is_some()
            || self.surface_extensions.emissive.is_some()
            || self.emissive_configuration.is_some()
            || self.ambient_occlusion_configuration.is_some()
            || matches!(
                &self.output,
                SurfaceOutput::MetallicRoughnessUv0Graph {
                    occlusion: Some(_),
                    ..
                }
            )
            || match &self.pbr_surface_factors {
                Some(factors) => match &factors.packed_metallic_roughness {
                    Some(packed) => packed.occlusion.is_some(),
                    None => false,
                },
                None => false,
            }
    }

    /// Returns the static alpha and raster facts selected for this surface.
    #[must_use]
    pub const fn surface_policy(&self) -> SurfacePolicy {
        self.surface_policy
    }

    /// Returns requested transmission and volume-attenuation controls.
    #[must_use]
    pub fn transmission_extensions(&self) -> Option<&TransmissionExtensions> {
        self.transmission_extensions.as_ref()
    }

    /// Returns the material-local transmission environment, when selected.
    #[must_use]
    pub fn refraction_environment(&self) -> Option<&RefractionEnvironment> {
        self.refraction_environment.as_ref()
    }

    /// Returns the selected surface domain.
    #[must_use]
    pub fn output(&self) -> &SurfaceOutput {
        &self.output
    }

    pub(crate) fn fragments(&self) -> &[MaterialFragment] {
        &self.fragments
    }
    pub(crate) fn code_nodes(&self) -> &[MaterialCodeNode] {
        &self.code_nodes
    }
    pub(crate) fn root_fragment(&self) -> &str {
        &self.root_fragment
    }
    pub(crate) fn parameters(&self) -> &[MaterialParameter] {
        &self.parameters
    }
    pub(crate) fn resources(&self) -> &[MaterialResource] {
        &self.resources
    }
    pub(crate) fn texture_encoding_requirements(&self) -> &[MaterialTextureEncodingRequirement] {
        &self.texture_encoding_requirements
    }
    pub(crate) fn texture_uv(&self) -> &BTreeMap<String, MaterialTextureUv> {
        &self.texture_uv
    }
    pub(crate) fn static_options(&self) -> &BTreeMap<String, StaticValue> {
        &self.static_options
    }
    /// Whether the material graph requires an explicit world-space camera eye.
    pub fn requires_surface_camera_position(&self) -> bool {
        self.node_graph.as_ref().is_some_and(|graph| {
            graph.nodes.iter().any(|node| {
                node.inputs
                    .iter()
                    .any(|input| input.source == MaterialNodeInputSource::CameraPosition)
            })
        })
    }

    pub(crate) fn vertex_uv0(&self) -> Option<&VertexUv0> {
        self.vertex_uv0.as_ref()
    }

    pub(crate) fn vertex_varyings(&self) -> &[MaterialVertexVarying] {
        &self.vertex_varyings
    }

    pub(crate) fn node_graph(&self) -> Option<&MaterialNodeGraph> {
        self.node_graph.as_ref()
    }
}
