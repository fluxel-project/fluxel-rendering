//! Capability requirements inferred from validated Naga modules.
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
/// A stable name for one Naga 27 capability required by composed WGSL.
///
/// Consumers must admit every value returned by an artifact before attempting
/// native shader creation. A value does not imply a particular backend API or
/// extension: the selected RHI maps it to its queried capabilities.
pub enum NagaCapability {
    /// `PUSH_CONSTANT`.
    PushConstant,
    /// `FLOAT64`.
    Float64,
    /// `PRIMITIVE_INDEX`.
    PrimitiveIndex,
    /// `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`.
    SampledTextureAndStorageBufferArrayNonUniformIndexing,
    /// `STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING`.
    StorageTextureArrayNonUniformIndexing,
    /// `UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING`.
    UniformBufferArrayNonUniformIndexing,
    /// `SAMPLER_NON_UNIFORM_INDEXING`.
    SamplerNonUniformIndexing,
    /// `CLIP_DISTANCE`.
    ClipDistance,
    /// `CULL_DISTANCE`.
    CullDistance,
    /// `STORAGE_TEXTURE_16BIT_NORM_FORMATS`.
    StorageTexture16BitNormFormats,
    /// `MULTIVIEW`.
    Multiview,
    /// `EARLY_DEPTH_TEST`.
    EarlyDepthTest,
    /// `MULTISAMPLED_SHADING`.
    MultisampledShading,
    /// `RAY_QUERY`.
    RayQuery,
    /// `DUAL_SOURCE_BLENDING`.
    DualSourceBlending,
    /// `CUBE_ARRAY_TEXTURES`.
    CubeArrayTextures,
    /// `SHADER_INT64`.
    ShaderInt64,
    /// `SUBGROUP`.
    Subgroup,
    /// `SUBGROUP_BARRIER`.
    SubgroupBarrier,
    /// `SUBGROUP_VERTEX_STAGE`.
    SubgroupVertexStage,
    /// `SHADER_INT64_ATOMIC_MIN_MAX`.
    ShaderInt64AtomicMinMax,
    /// `SHADER_INT64_ATOMIC_ALL_OPS`.
    ShaderInt64AtomicAllOps,
    /// `SHADER_FLOAT32_ATOMIC`.
    ShaderFloat32Atomic,
    /// `TEXTURE_ATOMIC`.
    TextureAtomic,
    /// `TEXTURE_INT64_ATOMIC`.
    TextureInt64Atomic,
    /// `RAY_HIT_VERTEX_POSITION`.
    RayHitVertexPosition,
    /// `SHADER_FLOAT16`.
    ShaderFloat16,
    /// `TEXTURE_EXTERNAL`.
    TextureExternal,
    /// `SHADER_FLOAT16_IN_FLOAT32`.
    ShaderFloat16InFloat32,
}
impl NagaCapability {
    /// Returns the stable, machine-readable capability name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::PushConstant => "push-constant",
            Self::Float64 => "float64",
            Self::PrimitiveIndex => "primitive-index",
            Self::SampledTextureAndStorageBufferArrayNonUniformIndexing => {
                "sampled-texture-and-storage-buffer-array-non-uniform-indexing"
            }
            Self::StorageTextureArrayNonUniformIndexing => {
                "storage-texture-array-non-uniform-indexing"
            }
            Self::UniformBufferArrayNonUniformIndexing => {
                "uniform-buffer-array-non-uniform-indexing"
            }
            Self::SamplerNonUniformIndexing => "sampler-non-uniform-indexing",
            Self::ClipDistance => "clip-distance",
            Self::CullDistance => "cull-distance",
            Self::StorageTexture16BitNormFormats => "storage-texture-16bit-norm-formats",
            Self::Multiview => "multiview",
            Self::EarlyDepthTest => "early-depth-test",
            Self::MultisampledShading => "multisampled-shading",
            Self::RayQuery => "ray-query",
            Self::DualSourceBlending => "dual-source-blending",
            Self::CubeArrayTextures => "cube-array-textures",
            Self::ShaderInt64 => "shader-int64",
            Self::Subgroup => "subgroup",
            Self::SubgroupBarrier => "subgroup-barrier",
            Self::SubgroupVertexStage => "subgroup-vertex-stage",
            Self::ShaderInt64AtomicMinMax => "shader-int64-atomic-min-max",
            Self::ShaderInt64AtomicAllOps => "shader-int64-atomic-all-ops",
            Self::ShaderFloat32Atomic => "shader-float32-atomic",
            Self::TextureAtomic => "texture-atomic",
            Self::TextureInt64Atomic => "texture-int64-atomic",
            Self::RayHitVertexPosition => "ray-hit-vertex-position",
            Self::ShaderFloat16 => "shader-float16",
            Self::TextureExternal => "texture-external",
            Self::ShaderFloat16InFloat32 => "shader-float16-in-float32",
        }
    }
}

fn naga_capability_bits() -> [(NagaCapability, naga::valid::Capabilities); 29] {
    use naga::valid::Capabilities;
    [
        (NagaCapability::PushConstant, Capabilities::PUSH_CONSTANT),
        (NagaCapability::Float64, Capabilities::FLOAT64),
        (
            NagaCapability::PrimitiveIndex,
            Capabilities::PRIMITIVE_INDEX,
        ),
        (
            NagaCapability::SampledTextureAndStorageBufferArrayNonUniformIndexing,
            Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        ),
        (
            NagaCapability::StorageTextureArrayNonUniformIndexing,
            Capabilities::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
        ),
        (
            NagaCapability::UniformBufferArrayNonUniformIndexing,
            Capabilities::UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        ),
        (
            NagaCapability::SamplerNonUniformIndexing,
            Capabilities::SAMPLER_NON_UNIFORM_INDEXING,
        ),
        (NagaCapability::ClipDistance, Capabilities::CLIP_DISTANCE),
        (NagaCapability::CullDistance, Capabilities::CULL_DISTANCE),
        (
            NagaCapability::StorageTexture16BitNormFormats,
            Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS,
        ),
        (NagaCapability::Multiview, Capabilities::MULTIVIEW),
        (
            NagaCapability::EarlyDepthTest,
            Capabilities::EARLY_DEPTH_TEST,
        ),
        (
            NagaCapability::MultisampledShading,
            Capabilities::MULTISAMPLED_SHADING,
        ),
        (NagaCapability::RayQuery, Capabilities::RAY_QUERY),
        (
            NagaCapability::DualSourceBlending,
            Capabilities::DUAL_SOURCE_BLENDING,
        ),
        (
            NagaCapability::CubeArrayTextures,
            Capabilities::CUBE_ARRAY_TEXTURES,
        ),
        (NagaCapability::ShaderInt64, Capabilities::SHADER_INT64),
        (NagaCapability::Subgroup, Capabilities::SUBGROUP),
        (
            NagaCapability::SubgroupBarrier,
            Capabilities::SUBGROUP_BARRIER,
        ),
        (
            NagaCapability::SubgroupVertexStage,
            Capabilities::SUBGROUP_VERTEX_STAGE,
        ),
        // `ALL_OPS` is a strict superset of `MIN_MAX` for Naga validation.
        // Remove the stronger alternative first so a storage `atomicMin` or
        // `atomicMax` whose result is unused converges on the narrower demand.
        // Reversing these two probes would retain `ALL_OPS`: it can validate
        // the narrow operation, but `MIN_MAX` cannot then remove `ALL_OPS`.
        (
            NagaCapability::ShaderInt64AtomicAllOps,
            Capabilities::SHADER_INT64_ATOMIC_ALL_OPS,
        ),
        (
            NagaCapability::ShaderInt64AtomicMinMax,
            Capabilities::SHADER_INT64_ATOMIC_MIN_MAX,
        ),
        (
            NagaCapability::ShaderFloat32Atomic,
            Capabilities::SHADER_FLOAT32_ATOMIC,
        ),
        (NagaCapability::TextureAtomic, Capabilities::TEXTURE_ATOMIC),
        (
            NagaCapability::TextureInt64Atomic,
            Capabilities::TEXTURE_INT64_ATOMIC,
        ),
        (
            NagaCapability::RayHitVertexPosition,
            Capabilities::RAY_HIT_VERTEX_POSITION,
        ),
        (NagaCapability::ShaderFloat16, Capabilities::SHADER_FLOAT16),
        (
            NagaCapability::TextureExternal,
            Capabilities::TEXTURE_EXTERNAL,
        ),
        (
            NagaCapability::ShaderFloat16InFloat32,
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        ),
    ]
}
pub(crate) fn required_naga_capabilities(
    module: &naga::Module,
) -> Result<BTreeSet<NagaCapability>, Box<naga::WithSpan<naga::valid::ValidationError>>> {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let mut sufficient = Capabilities::all();
    for (_, capability) in naga_capability_bits() {
        let candidate = sufficient - capability;
        if Validator::new(ValidationFlags::all(), candidate)
            .validate(module)
            .is_ok()
        {
            sufficient = candidate;
        }
    }
    // The individual removals must not turn alternatives into an empty or
    // otherwise insufficient set. Keep this final validation as the invariant
    // that makes the returned demand set safe for consumer admission.
    Validator::new(ValidationFlags::all(), sufficient)
        .validate(module)
        .map_err(Box::new)?;
    Ok(naga_capability_bits()
        .into_iter()
        .filter_map(|(capability, bit)| sufficient.contains(bit).then_some(capability))
        .collect())
}
