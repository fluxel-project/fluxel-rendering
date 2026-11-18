//! Complete material authoring definitions.

use std::collections::BTreeMap;

use super::{
    MaterialCodeNode, MaterialFragment, MaterialNodeGraph, MaterialParameter, MaterialResource,
    MaterialTextureEncoding, MaterialTextureEncodingRequirement, StaticValue, SurfacePolicy,
    TransmissionExtensions,
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
    /// Optional named texture, sampler, level, and transfer encoding.
    pub texture: Option<EmissiveTexture>,
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
    /// sRGB texture path. The evaluator consumes the resulting linear RGB.
    pub encoding: MaterialTextureEncoding,
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
    static_options: BTreeMap<String, StaticValue>,
    output: SurfaceOutput,
    surface_extensions: SurfaceExtensions,
    emissive_configuration: Option<EmissiveConfiguration>,
    surface_policy: SurfacePolicy,
    transmission_extensions: Option<TransmissionExtensions>,
    node_graph: Option<MaterialNodeGraph>,
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
            static_options,
            output,
            surface_extensions: SurfaceExtensions::default(),
            emissive_configuration: None,
            surface_policy: SurfacePolicy::default(),
            transmission_extensions: None,
            node_graph: None,
        }
    }

    /// Adds one first-class author WGSL node to this material definition.
    #[must_use]
    pub fn with_code_node(mut self, node: MaterialCodeNode) -> Self {
        self.code_nodes.push(node);
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

    /// Returns whether this material needs the renderer effects route.
    ///
    /// The effects route owns the stable occlusion and emissive evaluator ABI
    /// for UV0 normal PBR surfaces.
    #[must_use]
    pub const fn requires_surface_effects(&self) -> bool {
        self.surface_extensions.occlusion.is_some()
            || self.surface_extensions.emissive.is_some()
            || self.emissive_configuration.is_some()
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
    pub(crate) fn static_options(&self) -> &BTreeMap<String, StaticValue> {
        &self.static_options
    }
    pub(crate) fn node_graph(&self) -> Option<&MaterialNodeGraph> {
        self.node_graph.as_ref()
    }
}
