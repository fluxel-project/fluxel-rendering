//! Private, backend-neutral shader artifact vocabulary.
//!
//! This is deliberately not a public RHI surface yet.  It records the input
//! language separately from its reflection metadata, so neither RenderGraph
//! nor a backend contract is accidentally made WGSL-only. Source routing is
//! native-passthrough first; Naga is a translation option, never mandatory IR.

/// A pipeline stage in Fluxel's private shader artifact contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// A stable content identity supplied by the shader preparation path.
///
/// The digest algorithm is intentionally outside this transport contract. The
/// producer and artifact cache must use the same algorithm and include the
/// language, entry point, and relevant lowering options in their cache key.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ShaderSourceHash(pub(crate) [u8; 32]);

/// The GLSL dialect emitted by an authoring or backend-lowering path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum GlslDialect {
    Desktop { version: u16 },
    Embedded { version: u16 },
}

/// The backend family selecting a shader route.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) enum ShaderBackendFamily {
    Wgpu,
    Dx12,
    Vulkan,
    /// The fixed WebGL2 shader target: ESSL 300 only.
    WebGl2,
    /// A native GL target with the exact dialect selected from its profile.
    Gl {
        dialect: GlslDialect,
    },
}

/// The source route selected before backend creation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) enum ShaderRoute {
    /// The backend accepts this source representation directly.
    Native,
    /// The source requires a selected Naga-supported translation route.
    Translatable,
    /// Neither native passthrough nor the selected Naga route can accept it.
    Unsupported,
}

/// A source language accepted by the private preparation boundary.
///
/// `Hlsl` and `Dxil` have a DX12-native passthrough route. They are not
/// silently forced through Naga for other backends.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) enum ShaderModuleSource {
    Wgsl {
        source: String,
        entry_point: String,
    },
    Glsl {
        stage: ShaderStage,
        dialect: GlslDialect,
        source: String,
        entry_point: String,
    },
    SpirV {
        stage: ShaderStage,
        words: Vec<u32>,
        entry_point: String,
    },
    Hlsl {
        stage: ShaderStage,
        source: String,
        entry_point: String,
    },
    Dxil {
        stage: ShaderStage,
        bytes: Vec<u8>,
        entry_point: String,
    },
}

impl ShaderModuleSource {
    /// Chooses a source route without inventing an intermediate representation.
    ///
    /// GL receives `Native` only for its exact selected dialect; a desktop
    /// GLSL string is not legal WebGL2 source merely because it is labelled
    /// GLSL. Every `Translatable` answer names a combination that Naga can
    /// actually lower (frontend -> backend); combinations with no such pair —
    /// notably anything non-native on DX12, because Naga has no HLSL output —
    /// are `Unsupported` instead of a silent translation promise.
    #[allow(
        dead_code,
        reason = "The private preparation seam is specified before its first compat consumer exists."
    )]
    pub(crate) fn route_for(&self, target: ShaderBackendFamily) -> ShaderRoute {
        match (target, self) {
            (ShaderBackendFamily::Wgpu, Self::Wgsl { .. })
            | (ShaderBackendFamily::Dx12, Self::Hlsl { .. })
            | (ShaderBackendFamily::Dx12, Self::Dxil { .. })
            | (ShaderBackendFamily::Vulkan, Self::SpirV { .. }) => ShaderRoute::Native,
            (
                ShaderBackendFamily::WebGl2,
                Self::Glsl {
                    dialect: GlslDialect::Embedded { version: 300 },
                    ..
                },
            ) => ShaderRoute::Native,
            (
                ShaderBackendFamily::Gl { dialect: target },
                Self::Glsl {
                    dialect: source, ..
                },
            ) if *source == target => ShaderRoute::Native,
            // HLSL/DXIL payloads have no Naga frontend, so their only route is
            // the DX12 native passthrough above; DX12 in turn accepts no other
            // source representation at all.
            (ShaderBackendFamily::Dx12, _) | (_, Self::Hlsl { .. } | Self::Dxil { .. }) => {
                ShaderRoute::Unsupported
            }
            // WGSL/GLSL/SPIR-V crosses, each backed by a real Naga
            // frontend/backend pair (e.g. wgsl-in + glsl-out for
            // WebGl2 + Wgsl). Executing a `Translatable` route additionally
            // requires the matching naga features to be enabled when the
            // translation bridge is implemented.
            _ => ShaderRoute::Translatable,
        }
    }
}

/// Backend-neutral reflection emitted by validation/lowering.
///
/// Concrete binding mappings remain backend artifacts because an RHI binding
/// is not a GL texture unit or uniform-block index.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) struct ShaderBindingMapping<Binding> {
    /// Binding identity in the common artifact recipe.
    pub(crate) logical: Binding,
    /// Binding identity in the validated module's reflection.
    pub(crate) module_binding: u32,
}

/// Backend-neutral reflection emitted by validation/lowering.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) struct ValidatedShaderReflection<Binding> {
    pub(crate) bindings: Vec<ShaderBindingMapping<Binding>>,
}

/// A source artifact with validated reflection and logical binding metadata.
///
/// It intentionally contains no universal Naga IR. Native passthrough retains
/// its original source payload; translation stores the output selected by that
/// translation path in backend-private data.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(
    dead_code,
    reason = "The private preparation seam is specified before its first compat consumer exists."
)]
pub(crate) struct ValidatedShaderModule<Binding> {
    pub(crate) stage: ShaderStage,
    pub(crate) entry_point: String,
    pub(crate) source_hash: ShaderSourceHash,
    pub(crate) reflection: ValidatedShaderReflection<Binding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> ShaderModuleSource {
        ShaderModuleSource::Wgsl {
            source: "@vertex fn main() {}".into(),
            entry_point: "main".into(),
        }
    }

    #[test]
    fn chooses_native_routes_before_translation() {
        assert_eq!(
            source().route_for(ShaderBackendFamily::Wgpu),
            ShaderRoute::Native
        );
        let dxil = ShaderModuleSource::Dxil {
            stage: ShaderStage::Vertex,
            bytes: vec![1],
            entry_point: "main".into(),
        };
        assert_eq!(
            dxil.route_for(ShaderBackendFamily::Dx12),
            ShaderRoute::Native
        );
        assert_eq!(
            dxil.route_for(ShaderBackendFamily::Vulkan),
            ShaderRoute::Unsupported
        );
    }

    #[test]
    fn gl_native_route_requires_an_exact_dialect() {
        let source = ShaderModuleSource::Glsl {
            stage: ShaderStage::Vertex,
            dialect: GlslDialect::Embedded { version: 300 },
            source: "void main() {}".into(),
            entry_point: "main".into(),
        };
        assert_eq!(
            source.route_for(ShaderBackendFamily::WebGl2),
            ShaderRoute::Native
        );
        assert_eq!(
            source.route_for(ShaderBackendFamily::Gl {
                dialect: GlslDialect::Desktop { version: 460 },
            }),
            ShaderRoute::Translatable
        );
    }

    #[test]
    fn webgl2_hlsl_is_unsupported_not_a_glsl_fallback() {
        let source = ShaderModuleSource::Hlsl {
            stage: ShaderStage::Vertex,
            source: "float4 main() : SV_Position { return 0; }".into(),
            entry_point: "main".into(),
        };
        assert_eq!(
            source.route_for(ShaderBackendFamily::WebGl2),
            ShaderRoute::Unsupported
        );
    }

    #[test]
    fn dx12_rejects_sources_without_an_hlsl_output_path() {
        // Naga has no HLSL backend, so DX12 cannot lower anything except its
        // native HLSL/DXIL passthrough routes.
        let spirv = ShaderModuleSource::SpirV {
            stage: ShaderStage::Vertex,
            words: vec![1],
            entry_point: "main".into(),
        };
        assert_eq!(
            spirv.route_for(ShaderBackendFamily::Dx12),
            ShaderRoute::Unsupported
        );
        assert_eq!(
            source().route_for(ShaderBackendFamily::Dx12),
            ShaderRoute::Unsupported
        );
    }

    #[test]
    fn translatable_answers_name_real_naga_routes() {
        // spirv-in + wgsl-out, wgsl-in + spirv-out, spirv-in + glsl-out,
        // and glsl-in + wgsl-out are all existing Naga frontend/backend pairs.
        let spirv = ShaderModuleSource::SpirV {
            stage: ShaderStage::Vertex,
            words: vec![1],
            entry_point: "main".into(),
        };
        assert_eq!(
            spirv.route_for(ShaderBackendFamily::Wgpu),
            ShaderRoute::Translatable
        );
        assert_eq!(
            source().route_for(ShaderBackendFamily::Vulkan),
            ShaderRoute::Translatable
        );
        assert_eq!(
            spirv.route_for(ShaderBackendFamily::WebGl2),
            ShaderRoute::Translatable
        );
        let glsl = ShaderModuleSource::Glsl {
            stage: ShaderStage::Vertex,
            dialect: GlslDialect::Desktop { version: 460 },
            source: "void main() {}".into(),
            entry_point: "main".into(),
        };
        assert_eq!(
            glsl.route_for(ShaderBackendFamily::Wgpu),
            ShaderRoute::Translatable
        );
    }
}
