//! Shared authoring values and callable descriptions.

use fluxel_shader::{ScalarType, ValueType};

/// Types exposed by the minimal material function boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialType {
    /// A scalar `f32` value.
    F32,
    /// A `vec2<f32>` value.
    Vec2,
    /// A `vec3<f32>` value.
    Vec3,
    /// A `vec4<f32>` value.
    Vec4,
}

impl MaterialType {
    /// Returns the WGSL spelling for this type.
    #[must_use]
    pub const fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
        }
    }

    /// Returns the exact shader value type used by callable and uniform reflection.
    #[must_use]
    pub const fn shader_type(self) -> ValueType {
        match self {
            Self::F32 => ValueType::Scalar(ScalarType::F32),
            Self::Vec2 => ValueType::Vec2(ScalarType::F32),
            Self::Vec3 => ValueType::Vec3(ScalarType::F32),
            Self::Vec4 => ValueType::Vec4(ScalarType::F32),
        }
    }
}

/// A typed material parameter supplied per instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialParameter {
    /// Stable parameter name used by authoring and WGSL fragments.
    pub name: String,
    /// Parameter value type.
    pub ty: MaterialType,
    /// Logical resource that receives this parameter's per-instance value.
    pub resource: String,
}

/// A runtime value for a dynamic material parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicValue {
    /// Scalar parameter value.
    F32(f32),
    /// Two-component parameter value.
    Vec2([f32; 2]),
    /// Three-component parameter value.
    Vec3([f32; 3]),
    /// Four-component parameter value.
    Vec4([f32; 4]),
}

impl DynamicValue {
    /// Returns this value's declared material type.
    #[must_use]
    pub const fn ty(&self) -> MaterialType {
        match self {
            Self::F32(_) => MaterialType::F32,
            Self::Vec2(_) => MaterialType::Vec2,
            Self::Vec3(_) => MaterialType::Vec3,
            Self::Vec4(_) => MaterialType::Vec4,
        }
    }
}

/// A bind-group location declared by a material resource.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BindingLocation {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the bind group.
    pub binding: u32,
}

/// Logical resource kinds supported by the material core.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialResourceKind {
    /// A uniform `f32` binding.
    UniformF32,
    /// A uniform `vec2<f32>` binding.
    UniformVec2,
    /// A uniform `vec3<f32>` binding.
    UniformVec3,
    /// A uniform `vec4<f32>` binding.
    UniformVec4,
    /// A read-only `texture_2d<f32>` binding.
    Texture2dF32,
    /// A sampler binding with its WGSL comparison mode.
    Sampler {
        /// Whether the shader declares `sampler_comparison` rather than `sampler`.
        comparison: bool,
    },
}

/// Transfer encoding required by an explicitly typed material texture binding.
///
/// This is optional metadata. Generic material textures retain no prescribed
/// encoding until a consumer supplies an explicit requirement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialTextureEncoding {
    /// Texture RGB values are already linear.
    Linear,
    /// Texture RGB values use the sRGB transfer function.
    Srgb,
}

/// One optional transfer-encoding requirement for a declared texture resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialTextureEncodingRequirement {
    /// Declared `Texture2dF32` resource name.
    pub resource: String,
    /// Required transfer encoding for an asset bound to this resource.
    pub encoding: MaterialTextureEncoding,
}

impl MaterialResourceKind {
    /// Returns the WGSL type spelling for this resource.
    #[must_use]
    pub const fn wgsl(self) -> &'static str {
        match self {
            Self::UniformF32 => "f32",
            Self::UniformVec2 => "vec2<f32>",
            Self::UniformVec3 => "vec3<f32>",
            Self::UniformVec4 => "vec4<f32>",
            Self::Texture2dF32 => "texture_2d<f32>",
            Self::Sampler { comparison: false } => "sampler",
            Self::Sampler { comparison: true } => "sampler_comparison",
        }
    }

    /// Returns the dynamic material value type for a uniform resource.
    #[must_use]
    pub const fn uniform_material_type(self) -> Option<MaterialType> {
        match self {
            Self::UniformF32 => Some(MaterialType::F32),
            Self::UniformVec2 => Some(MaterialType::Vec2),
            Self::UniformVec3 => Some(MaterialType::Vec3),
            Self::UniformVec4 => Some(MaterialType::Vec4),
            Self::Texture2dF32 | Self::Sampler { .. } => None,
        }
    }

    /// Returns a portable uniform binding allocation in bytes.
    ///
    /// `vec3<f32>` has 16-byte uniform layout alignment and therefore occupies
    /// a 16-byte dynamic allocation even though it contains three components.
    #[must_use]
    pub const fn uniform_min_binding_size(self) -> Option<u64> {
        match self {
            Self::UniformF32 => Some(4),
            Self::UniformVec2 => Some(8),
            Self::UniformVec3 | Self::UniformVec4 => Some(16),
            Self::Texture2dF32 | Self::Sampler { .. } => None,
        }
    }
}

/// A named material resource and its shader-visible binding location.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialResource {
    /// Stable resource name used by fragments.
    pub name: String,
    /// Generic resource kind.
    pub kind: MaterialResourceKind,
    /// Shader-visible bind-group location.
    pub location: BindingLocation,
}

/// A statically selected material option.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticValue {
    /// Boolean static value.
    Bool(bool),
    /// Unsigned integer static value.
    U32(u32),
}

impl StaticValue {
    pub(crate) fn canonical(&self) -> String {
        match self {
            Self::Bool(value) => format!("bool:{value}"),
            Self::U32(value) => format!("u32:{value}"),
        }
    }
}

/// A typed input expected by a material fragment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FragmentInput {
    /// Input name for diagnostics.
    pub name: String,
    /// Input type.
    pub ty: MaterialType,
}

/// An explicit call made by a fragment to another fragment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FragmentCall {
    /// Called fragment name.
    pub fragment: String,
    /// Argument types in call order.
    pub arguments: Vec<MaterialType>,
}

/// Rust-defined material function metadata plus its WGSL implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialFragment {
    /// Stable fragment name.
    pub name: String,
    /// Input metadata used to validate explicit material calls.
    pub inputs: Vec<FragmentInput>,
    /// Fragment result type.
    pub output: MaterialType,
    /// Explicit outgoing calls used for deterministic graph validation.
    pub calls: Vec<FragmentCall>,
    /// WGSL declarations and function implementation for this fragment.
    pub wgsl: String,
}
