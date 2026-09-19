//! Closed raster artifact descriptors and creation errors.
use super::*;
/// Why creation of a fixed raster artifact was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RasterCreateError {
    /// The selected adapter cannot filter the requested fixed texture format.
    TextureFormatNotFilterable {
        /// The fixed texture format that requires filtering support.
        format: TextureFormat,
    },
    /// The pipeline, attachment, or buffer operation crossed device identities.
    ForeignDevice,
    /// The closed raster uniform recipe received an incompatible object.
    BindingRecipeMismatch,
    /// The uniform buffer is not exactly the fixed 80-byte whole binding.
    InvalidBindingRange,
    /// The buffer's native creation facts do not authorize uniform reads.
    UniformUsageRequired,
    /// The position stream is not the exact closed f32x3 vertex range.
    InvalidPositionStreamRange,
    /// The normal stream is not the exact closed f32x3 vertex range.
    InvalidNormalStreamRange,
    /// The color stream is not the exact closed RGBA8 whole-stream range.
    InvalidColorStreamRange,
    /// The texture-coordinate stream is not the exact closed f32x2 vertex range.
    InvalidTextureCoordinateStreamRange,
    /// A closed vertex stream was created without vertex usage.
    VertexUsageRequired,
    /// The two closed vertex streams do not describe the requested vertex count.
    VertexStreamCountMismatch,
    /// Native creation of a validated fixed raster binding failed.
    NativeFailure(String),
    /// The fixed WGSL artifact failed Naga parsing or validation.
    ShaderValidation(String),
    /// The backend rejected shader lowering or raster-pipeline compilation.
    ShaderCompilation(String),
    /// The backend could not create a fixed layout or another non-shader object.
    NativeObjectCreation(String),
}

impl fmt::Display for RasterCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextureFormatNotFilterable { format } => {
                write!(f, "raster texture format is not filterable: {format:?}")
            }
            Self::ForeignDevice => f.write_str("raster objects belong to different devices"),
            Self::BindingRecipeMismatch => {
                f.write_str("raster bindings do not match the fixed artifact")
            }
            Self::InvalidBindingRange => f.write_str("invalid fixed raster uniform range"),
            Self::UniformUsageRequired => {
                f.write_str("raster uniform binding requires uniform usage")
            }
            Self::InvalidPositionStreamRange => {
                f.write_str("invalid fixed raster position stream range")
            }
            Self::InvalidNormalStreamRange => {
                f.write_str("invalid fixed raster normal stream range")
            }
            Self::InvalidColorStreamRange => f.write_str("invalid fixed raster color stream range"),
            Self::InvalidTextureCoordinateStreamRange => {
                f.write_str("invalid fixed raster texture-coordinate stream range")
            }
            Self::VertexUsageRequired => {
                f.write_str("fixed raster vertex streams require vertex usage")
            }
            Self::VertexStreamCountMismatch => {
                f.write_str("fixed raster vertex stream count mismatch")
            }
            Self::NativeFailure(reason) => {
                write!(f, "native raster binding creation failed: {reason}")
            }
            Self::ShaderValidation(reason) => {
                write!(f, "raster shader validation failed: {reason}")
            }
            Self::ShaderCompilation(reason) => {
                write!(f, "raster shader compilation failed: {reason}")
            }
            Self::NativeObjectCreation(reason) => {
                write!(f, "native raster object creation failed: {reason}")
            }
        }
    }
}
impl std::error::Error for RasterCreateError {}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(super) fn map_raster_pipeline_create_error(
    error: crate::imp::RasterPipelineCreateError,
) -> RasterCreateError {
    match error {
        crate::imp::RasterPipelineCreateError::ShaderValidation(reason) => {
            RasterCreateError::ShaderValidation(reason)
        }
        crate::imp::RasterPipelineCreateError::ShaderCompilation(reason) => {
            RasterCreateError::ShaderCompilation(reason)
        }
        crate::imp::RasterPipelineCreateError::NativeObjectCreation(reason) => {
            RasterCreateError::NativeObjectCreation(reason)
        }
    }
}

/// The deterministic raster artifacts supported by this milestone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterKernel {
    /// A fixed, non-indexed triangle with no resource bindings.
    Triangle,
    /// An indexed mesh using `float32x2 position + unorm8x4 color` vertices.
    IndexedPositionColor,
    /// An indexed clip-space mesh using `float32x3` positions and a fixed
    /// opaque fragment color.
    IndexedPositionFloat32x3,
    /// An indexed model-identity mesh using `float32x3` positions and exactly
    /// one 80-byte view-projection plus base-color uniform binding.
    IndexedPositionFloat32x3CameraMaterial,
    /// An indexed camera/material mesh with one closed `textureLoad` RGBA8 binding.
    IndexedPositionFloat32x3CameraMaterialTexture,
    /// An indexed camera/material mesh with separate f32x3 position and f32x2
    /// UV streams plus one closed `textureLoad` RGBA8 binding.
    IndexedPositionFloat32x3CameraMaterialTextureUv,
    /// An indexed camera/material mesh with explicit UV streams and a fixed
    /// filtering linear-clamp sampler at explicit mip level zero.
    IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
    /// An indexed camera/material mesh with explicit UV streams, fixed
    /// linear-clamp sampling at explicit mip level zero, and an sRGB
    /// base-color texture decoded by the native sampling operation.
    IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
    /// An indexed camera/material mesh with separate f32x3 position and
    /// unit-normal streams plus fixed object-space `+Z` Lambert lighting.
    IndexedPositionFloat32x3CameraMaterialNormalLambert,
    /// An indexed camera/material mesh with separate f32x3 position and
    /// normalized RGBA8 vertex-color streams.
    IndexedPositionFloat32x3CameraMaterialVertexColor,
}

/// The non-configurable vertex layout selected by a fixed raster artifact.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterVertexLayout {
    /// The vertex shader derives its three fixed positions from `vertex_index`.
    None,
    /// One vertex is `float32x2 position` at byte zero followed by
    /// `unorm8x4 color` at byte eight, for a 12-byte stride.
    PositionFloat32x2ColorUnorm8x4,
    /// One vertex is `float32x3 position` at byte zero, for a 12-byte stride.
    PositionFloat32x3,
    /// Slot zero is tightly packed f32x3 position and slot one is tightly
    /// packed f32x2 texture coordinates.
    PositionFloat32x3AndTextureCoordinateFloat32x2,
    /// Slot zero is tightly packed f32x3 position and slot one is tightly
    /// packed f32x3 unit normal.
    PositionFloat32x3AndNormalFloat32x3,
    /// Slot zero is tightly packed f32x3 position and slot one is tightly
    /// packed normalized RGBA8 vertex color.
    PositionFloat32x3AndColorUnorm8x4,
}

/// Interpolation rule for the fixed normal stream.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterNormalInterpolation {
    /// Perspective-correct interpolation evaluated at fragment center.
    PerspectiveCenter,
}

/// Normalization rule for the interpolated normal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterNormalNormalization {
    /// Normalize only when length squared is strictly positive; otherwise the
    /// Lambert contribution is exactly zero.
    ExactPositiveUnitAfterInterpolationZeroLambert,
}

/// Coordinate space used by the fixed lighting recipe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterLightingSpace {
    /// Positions and normals are consumed in object space.
    Object,
}

/// The non-configurable light direction used by the fixed recipe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterFixedLightDirection {
    /// Unit positive Z direction.
    PositiveZ,
}

/// Color operation performed by the fixed lighting recipe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterLightingModel {
    /// RGB is modulated by Lambert while material alpha is preserved.
    LambertRgbAlphaPassthrough,
}

/// Shader-stage visibility recorded by a closed raster binding identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterBindingVisibility {
    /// The binding is visible to both vertex and fragment stages.
    VertexFragment,
    /// The binding is visible only to the fragment stage.
    Fragment,
}

/// Texture sample type recorded by a closed raster binding identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterTextureSampleType {
    /// A normalized floating-point sampled texture read as `f32`.
    Float,
}

/// Sampler type recorded by a closed raster artifact identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerBindingType {
    /// A filtering sampler used with a float sampled texture.
    Filtering,
}

/// Fixed filter choice recorded by a closed raster sampler identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerFilter {
    /// Linear interpolation between neighboring texels.
    Linear,
    /// Nearest mip selection.
    Nearest,
}

/// Fixed coordinate addressing recorded by a closed raster sampler identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerAddressMode {
    /// Clamp each coordinate to the texture edge.
    ClampToEdge,
}

/// Portable identity of one fixed raster artifact.
///
/// It describes the source-level recipe shared by DX12 and Vulkan, rather
/// than their intentionally different native binaries.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct RasterArtifactIdentity {
    /// Hash of the complete embedded WGSL module.
    pub module_source_hash: u64,
    /// Selected vertex entry point.
    pub vertex_entry_point: &'static str,
    /// Selected fragment entry point.
    pub fragment_entry_point: &'static str,
    /// Target format required by this artifact.
    pub target_format: TextureFormat,
    /// Complete non-configurable vertex layout of this artifact.
    pub vertex_layout: RasterVertexLayout,
    /// Bytes between vertices; zero means the fixed shader uses vertex index.
    pub vertex_stride: u32,
    /// The required index element format, or `None` for the non-indexed recipe.
    pub index_format: Option<IndexFormat>,
    /// Number of fixed bind-group entries; zero means no raster bindings.
    pub binding_count: u32,
    /// Minimum byte size of the sole fixed uniform binding, or zero when none.
    pub uniform_binding_size: u64,
    /// Uniform binding number for a fixed uniform recipe.
    pub uniform_binding: Option<u32>,
    /// Shader-stage visibility of the fixed uniform binding.
    pub uniform_visibility: Option<RasterBindingVisibility>,
    /// Version of the fixed binding layout; zero means no binding layout.
    pub binding_recipe_version: u32,
    /// Texture ABI schema version; zero means no texture binding.
    pub texture_binding_recipe_version: u32,
    /// Texture binding number for the closed textured recipe.
    pub texture_binding: Option<u32>,
    /// Texture dimension for the closed textured recipe.
    pub texture_dimension: Option<TextureDimension>,
    /// Texture format for the closed textured recipe.
    pub texture_format: Option<TextureFormat>,
    /// Shader-stage visibility of the closed texture binding.
    pub texture_visibility: Option<RasterBindingVisibility>,
    /// Sample type exposed by the closed texture binding.
    pub texture_sample_type: Option<RasterTextureSampleType>,
    /// The only mip level read by the closed texture recipe.
    pub texture_mip_level: Option<u32>,
    /// Version of the fixed texture-coordinate mapping recipe.
    pub texture_mapping_recipe_version: u32,
    /// Slot-one layout, only present for the explicit-UV recipe.
    pub texture_coordinate_vertex_layout: Option<RasterVertexLayout>,
    /// Slot-one byte stride, only present for the explicit-UV recipe.
    pub texture_coordinate_vertex_stride: Option<u32>,
    /// Shader location consumed from slot one, only present for the explicit-UV recipe.
    pub texture_coordinate_shader_location: Option<u32>,
    /// Sampler binding number for the fixed linear-clamp recipe.
    pub sampler_binding: Option<u32>,
    /// Sampler ABI schema version; zero means this artifact has no sampler.
    pub sampler_binding_recipe_version: u32,
    /// Sampler type for the fixed linear-clamp recipe.
    pub sampler_binding_type: Option<RasterSamplerBindingType>,
    /// Minification filter for the fixed linear-clamp recipe.
    pub sampler_min_filter: Option<RasterSamplerFilter>,
    /// Magnification filter for the fixed linear-clamp recipe.
    pub sampler_mag_filter: Option<RasterSamplerFilter>,
    /// Mipmap filter for the fixed linear-clamp recipe.
    pub sampler_mipmap_filter: Option<RasterSamplerFilter>,
    /// U coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_u: Option<RasterSamplerAddressMode>,
    /// V coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_v: Option<RasterSamplerAddressMode>,
    /// W coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_w: Option<RasterSamplerAddressMode>,
    /// Explicit mip level sampled by the fixed linear-clamp recipe.
    pub sampler_explicit_mip_level: Option<u32>,
    /// Version of the fixed normal/lighting ABI; zero means no lighting recipe.
    pub lighting_recipe_version: u32,
    /// Interpolation selected for the normal stream.
    pub normal_interpolation: Option<RasterNormalInterpolation>,
    /// Safe normalization and zero-vector fallback selected by the shader.
    pub normal_normalization: Option<RasterNormalNormalization>,
    /// Space in which the normal and fixed light are compared.
    pub lighting_space: Option<RasterLightingSpace>,
    /// Direction of the fixed light.
    pub fixed_light_direction: Option<RasterFixedLightDirection>,
    /// Fixed RGB/alpha lighting operation.
    pub lighting_model: Option<RasterLightingModel>,
    /// Position stream slot for the normal-Lambert recipe.
    pub position_vertex_slot: Option<u32>,
    /// Position shader location for the normal-Lambert recipe.
    pub position_shader_location: Option<u32>,
    /// Normal stream slot for the normal-Lambert recipe.
    pub normal_vertex_slot: Option<u32>,
    /// Normal stream byte stride for the normal-Lambert recipe.
    pub normal_vertex_stride: Option<u32>,
    /// Normal shader location for the normal-Lambert recipe.
    pub normal_shader_location: Option<u32>,
    /// Version of the fixed vertex-color stream ABI; zero means no color stream.
    pub vertex_color_recipe_version: u32,
    /// Position stream slot for the vertex-color recipe.
    pub vertex_color_position_slot: Option<u32>,
    /// Position shader location for the vertex-color recipe.
    pub vertex_color_position_shader_location: Option<u32>,
    /// Color stream slot for the vertex-color recipe.
    pub vertex_color_slot: Option<u32>,
    /// Color stream byte stride for the vertex-color recipe.
    pub vertex_color_stride: Option<u32>,
    /// Color shader location for the vertex-color recipe.
    pub vertex_color_shader_location: Option<u32>,
    /// Version of the fixed vertex/index/target recipe.
    pub recipe_version: u32,
}

impl fmt::Debug for RasterArtifactIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut identity = formatter.debug_struct("RasterArtifactIdentity");
        identity
            .field("module_source_hash", &self.module_source_hash)
            .field("vertex_entry_point", &self.vertex_entry_point)
            .field("fragment_entry_point", &self.fragment_entry_point)
            .field("target_format", &self.target_format)
            .field("vertex_layout", &self.vertex_layout)
            .field("vertex_stride", &self.vertex_stride)
            .field("index_format", &self.index_format);
        // Preserve the exact 0.2.1 U02 artifact representation. Binding facts
        // are an additive 0.2.2 identity component only for artifacts that
        // actually declare a binding recipe.
        if self.binding_count != 0
            || self.uniform_binding_size != 0
            || self.binding_recipe_version != 0
        {
            identity
                .field("binding_count", &self.binding_count)
                .field("uniform_binding_size", &self.uniform_binding_size)
                .field("binding_recipe_version", &self.binding_recipe_version);
        }
        if self.texture_binding_recipe_version != 0 {
            identity
                .field(
                    "texture_binding_recipe_version",
                    &self.texture_binding_recipe_version,
                )
                .field("texture_binding", &self.texture_binding)
                .field("texture_dimension", &self.texture_dimension)
                .field("texture_format", &self.texture_format)
                .field("uniform_binding", &self.uniform_binding)
                .field("uniform_visibility", &self.uniform_visibility)
                .field("texture_visibility", &self.texture_visibility)
                .field("texture_sample_type", &self.texture_sample_type)
                .field("texture_mip_level", &self.texture_mip_level)
                .field(
                    "texture_mapping_recipe_version",
                    &self.texture_mapping_recipe_version,
                );
        }
        if self.texture_coordinate_vertex_layout.is_some() {
            identity
                .field(
                    "texture_coordinate_vertex_layout",
                    &self.texture_coordinate_vertex_layout,
                )
                .field(
                    "texture_coordinate_vertex_stride",
                    &self.texture_coordinate_vertex_stride,
                )
                .field(
                    "texture_coordinate_shader_location",
                    &self.texture_coordinate_shader_location,
                );
        }
        if self.sampler_binding.is_some() {
            identity
                .field(
                    "sampler_binding_recipe_version",
                    &self.sampler_binding_recipe_version,
                )
                .field("sampler_binding", &self.sampler_binding)
                .field("sampler_binding_type", &self.sampler_binding_type)
                .field("sampler_min_filter", &self.sampler_min_filter)
                .field("sampler_mag_filter", &self.sampler_mag_filter)
                .field("sampler_mipmap_filter", &self.sampler_mipmap_filter)
                .field("sampler_address_mode_u", &self.sampler_address_mode_u)
                .field("sampler_address_mode_v", &self.sampler_address_mode_v)
                .field("sampler_address_mode_w", &self.sampler_address_mode_w)
                .field(
                    "sampler_explicit_mip_level",
                    &self.sampler_explicit_mip_level,
                );
        }
        // Lighting is additive identity data. Keeping it conditional preserves
        // the exact Debug representation recorded by pre-0.2.7 artifacts.
        if self.lighting_recipe_version != 0 {
            identity
                .field("lighting_recipe_version", &self.lighting_recipe_version)
                .field("normal_interpolation", &self.normal_interpolation)
                .field("normal_normalization", &self.normal_normalization)
                .field("lighting_space", &self.lighting_space)
                .field("fixed_light_direction", &self.fixed_light_direction)
                .field("lighting_model", &self.lighting_model)
                .field("position_vertex_slot", &self.position_vertex_slot)
                .field("position_shader_location", &self.position_shader_location)
                .field("normal_vertex_slot", &self.normal_vertex_slot)
                .field("normal_vertex_stride", &self.normal_vertex_stride)
                .field("normal_shader_location", &self.normal_shader_location);
        }
        if self.vertex_color_recipe_version != 0 {
            identity
                .field(
                    "vertex_color_recipe_version",
                    &self.vertex_color_recipe_version,
                )
                .field(
                    "vertex_color_position_slot",
                    &self.vertex_color_position_slot,
                )
                .field(
                    "vertex_color_position_shader_location",
                    &self.vertex_color_position_shader_location,
                )
                .field("vertex_color_slot", &self.vertex_color_slot)
                .field("vertex_color_stride", &self.vertex_color_stride)
                .field(
                    "vertex_color_shader_location",
                    &self.vertex_color_shader_location,
                );
        }
        identity
            .field("recipe_version", &self.recipe_version)
            .finish()
    }
}
