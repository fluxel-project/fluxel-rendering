//! Typed extension evidence and fail-closed core-or-extension resolution.

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use super::profile::{GlFamilyProfile, GlVersion};

/// Known extension names with a semantic role in the first GL-family slices.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GlKnownExtension {
    /// Desktop compute-shader commands.
    ArbComputeShader,
    /// Desktop storage-buffer commands.
    ArbShaderStorageBufferObject,
    /// Desktop image load/store commands.
    ArbShaderImageLoadStore,
    /// WebGL2 timer queries.
    ExtDisjointTimerQueryWebgl2,
    /// WebGL float render targets.
    ExtColorBufferFloat,
    /// WebGL float blending.
    ExtFloatBlend,
    /// Float texture filtering.
    OesTextureFloatLinear,
    /// Anisotropic texture filtering (including native `GL_EXT` spelling).
    ExtTextureFilterAnisotropic,
    /// WebGL multi-draw commands.
    WebglMultiDraw,
    /// WebGL multiview.
    OvrMultiview2,
    /// WebGL parallel-compilation polling.
    KhrParallelShaderCompile,
    /// Robustness/reset-status support.
    KhrRobustness,
    /// Debug diagnostics.
    KhrDebug,
    /// S3TC / BC1-3 compressed textures.
    CompressedTextureS3tc,
    /// sRGB S3TC / BC1-3 compressed textures.
    CompressedTextureS3tcSrgb,
    /// BPTC / BC6H-BC7 compressed textures.
    CompressedTextureBptc,
    /// RGTC / BC4-BC5 compressed textures.
    CompressedTextureRgtc,
    /// ASTC LDR compressed textures.
    CompressedTextureAstc,
    /// ETC compressed textures, including non-core native availability.
    CompressedTextureEtc,
}

impl GlKnownExtension {
    /// Returns the registry spelling retained as raw runtime evidence.
    pub const fn raw_name(self) -> &'static str {
        match self {
            Self::ArbComputeShader => "GL_ARB_compute_shader",
            Self::ArbShaderStorageBufferObject => "GL_ARB_shader_storage_buffer_object",
            Self::ArbShaderImageLoadStore => "GL_ARB_shader_image_load_store",
            Self::ExtDisjointTimerQueryWebgl2 => "EXT_disjoint_timer_query_webgl2",
            Self::ExtColorBufferFloat => "EXT_color_buffer_float",
            Self::ExtFloatBlend => "EXT_float_blend",
            Self::OesTextureFloatLinear => "OES_texture_float_linear",
            Self::ExtTextureFilterAnisotropic => "EXT_texture_filter_anisotropic",
            Self::WebglMultiDraw => "WEBGL_multi_draw",
            Self::OvrMultiview2 => "OVR_multiview2",
            Self::KhrParallelShaderCompile => "KHR_parallel_shader_compile",
            Self::KhrRobustness => "KHR_robustness",
            Self::KhrDebug => "KHR_debug",
            Self::CompressedTextureS3tc => "WEBGL_compressed_texture_s3tc",
            Self::CompressedTextureS3tcSrgb => "WEBGL_compressed_texture_s3tc_srgb",
            Self::CompressedTextureBptc => "WEBGL_compressed_texture_bptc",
            Self::CompressedTextureRgtc => "WEBGL_compressed_texture_rgtc",
            Self::CompressedTextureAstc => "WEBGL_compressed_texture_astc",
            Self::CompressedTextureEtc => "WEBGL_compressed_texture_etc",
        }
    }

    /// Converts an exact registry spelling into its typed counterpart.
    pub fn from_raw_name(name: &str) -> Option<Self> {
        [
            Self::ArbComputeShader,
            Self::ArbShaderStorageBufferObject,
            Self::ArbShaderImageLoadStore,
            Self::ExtDisjointTimerQueryWebgl2,
            Self::ExtColorBufferFloat,
            Self::ExtFloatBlend,
            Self::OesTextureFloatLinear,
            Self::ExtTextureFilterAnisotropic,
            Self::WebglMultiDraw,
            Self::OvrMultiview2,
            Self::KhrParallelShaderCompile,
            Self::KhrRobustness,
            Self::KhrDebug,
            Self::CompressedTextureS3tc,
            Self::CompressedTextureS3tcSrgb,
            Self::CompressedTextureBptc,
            Self::CompressedTextureRgtc,
            Self::CompressedTextureAstc,
            Self::CompressedTextureEtc,
        ]
        .into_iter()
        .find(|known| {
            known.raw_name() == name
                || (*known == Self::ExtTextureFilterAnisotropic
                    && matches!(
                        name,
                        "GL_EXT_texture_filter_anisotropic"
                            | "WEBKIT_EXT_texture_filter_anisotropic"
                            | "MOZ_EXT_texture_filter_anisotropic"
                    ))
                || matches!(
                    (*known, name),
                    (
                        Self::CompressedTextureS3tc,
                        "GL_EXT_texture_compression_s3tc"
                            | "GL_S3_s3tc"
                            | "GL_EXT_texture_compression_dxt1"
                    ) | (
                        Self::CompressedTextureS3tcSrgb,
                        "GL_EXT_texture_compression_s3tc_srgb"
                    ) | (
                        Self::CompressedTextureBptc,
                        "GL_ARB_texture_compression_bptc" | "GL_EXT_texture_compression_bptc"
                    ) | (
                        Self::CompressedTextureRgtc,
                        "GL_ARB_texture_compression_rgtc" | "GL_EXT_texture_compression_rgtc"
                    ) | (
                        Self::CompressedTextureAstc,
                        "GL_KHR_texture_compression_astc_ldr" | "GL_OES_texture_compression_astc"
                    ) | (
                        Self::CompressedTextureEtc,
                        "GL_OES_compressed_ETC2_RGB8_texture"
                            | "GL_OES_compressed_ETC2_sRGB8_texture"
                            | "GL_OES_compressed_ETC2_EAC"
                    )
                )
        })
    }

    /// Returns whether this registry extension is meaningful for the profile family.
    pub const fn is_legal_for(self, profile: GlFamilyProfile) -> bool {
        match self {
            Self::ArbComputeShader
            | Self::ArbShaderStorageBufferObject
            | Self::ArbShaderImageLoadStore => matches!(profile, GlFamilyProfile::Desktop { .. }),
            Self::ExtDisjointTimerQueryWebgl2
            | Self::ExtColorBufferFloat
            | Self::ExtFloatBlend
            | Self::WebglMultiDraw
            | Self::OvrMultiview2
            | Self::KhrParallelShaderCompile => matches!(profile, GlFamilyProfile::WebGl2),
            Self::OesTextureFloatLinear => matches!(
                profile,
                GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2
            ),
            Self::ExtTextureFilterAnisotropic => true,
            Self::KhrRobustness | Self::KhrDebug => !matches!(profile, GlFamilyProfile::WebGl2),
            Self::CompressedTextureS3tc
            | Self::CompressedTextureS3tcSrgb
            | Self::CompressedTextureBptc
            | Self::CompressedTextureRgtc
            | Self::CompressedTextureAstc
            | Self::CompressedTextureEtc => true,
        }
    }
}

/// Evidence reached while making a reported extension usable.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExtensionProvenance {
    /// The runtime listed a raw name, with no acquisition attempt yet.
    Reported,
    /// Required entry points or browser extension object were acquired.
    Acquired,
    /// A required operation probe succeeded.
    Probed,
    /// Acquisition or operation probing failed.
    Failed,
}

/// A complete extension evidence ledger preserving all runtime names.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GlExtensionSet {
    raw_reported_names: BTreeSet<String>,
    known: BTreeMap<GlKnownExtension, ExtensionProvenance>,
}

impl GlExtensionSet {
    /// Records a runtime extension string, even when Fluxel has no typed use for it.
    pub fn report_raw(&mut self, name: impl Into<String>) {
        let name = name.into();
        if let Some(known) = GlKnownExtension::from_raw_name(&name) {
            self.known
                .entry(known)
                .or_insert(ExtensionProvenance::Reported);
        }
        self.raw_reported_names.insert(name);
    }

    /// Records successful table/object acquisition for a previously reported extension.
    pub fn acquire(&mut self, extension: GlKnownExtension) -> bool {
        self.advance(extension, ExtensionProvenance::Acquired)
    }

    /// Records successful operation probing after acquisition.
    pub fn probe(&mut self, extension: GlKnownExtension) -> bool {
        self.advance(extension, ExtensionProvenance::Probed)
    }

    /// Records a failed acquisition or operation probe.
    pub fn fail(&mut self, extension: GlKnownExtension) -> bool {
        match self.known.entry(extension) {
            Entry::Occupied(mut entry) => {
                entry.insert(ExtensionProvenance::Failed);
                true
            }
            Entry::Vacant(_) => false,
        }
    }

    /// Returns raw runtime names, including names Fluxel does not yet model.
    pub fn raw_reported_names(&self) -> impl Iterator<Item = &str> {
        self.raw_reported_names.iter().map(String::as_str)
    }

    /// Returns the current evidence state for one typed extension.
    pub fn provenance(&self, extension: GlKnownExtension) -> Option<ExtensionProvenance> {
        self.known.get(&extension).copied()
    }

    /// Returns whether the extension has an acquired callable interface.
    pub fn is_acquired(&self, extension: GlKnownExtension) -> bool {
        matches!(
            self.provenance(extension),
            Some(ExtensionProvenance::Acquired | ExtensionProvenance::Probed)
        )
    }

    /// Returns whether an operation probe succeeded.
    pub fn is_probed(&self, extension: GlKnownExtension) -> bool {
        matches!(
            self.provenance(extension),
            Some(ExtensionProvenance::Probed)
        )
    }

    fn advance(&mut self, extension: GlKnownExtension, target: ExtensionProvenance) -> bool {
        match self.provenance(extension) {
            Some(ExtensionProvenance::Reported | ExtensionProvenance::Acquired) => {
                self.known.insert(extension, target);
                true
            }
            Some(ExtensionProvenance::Probed) if target == ExtensionProvenance::Probed => true,
            _ => false,
        }
    }
}

/// A semantic feature enabled by a core version or one exact extension.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CoreOrExtension {
    /// Minimum desktop core version, if core can supply the feature.
    pub desktop_core: Option<GlVersion>,
    /// Minimum embedded core version, if core can supply the feature.
    pub embedded_core: Option<GlVersion>,
    /// Exact extension alternative, if one is meaningful.
    pub extension: Option<GlKnownExtension>,
    /// Whether enabling the extension route requires a successful operation probe.
    pub extension_requires_probe: bool,
}

/// Provenance of an enabled semantic feature.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CapabilityEvidence {
    /// The current profile's core version supplied the feature.
    Core(GlFamilyProfile),
    /// An acquired extension supplied the feature.
    Extension(GlKnownExtension),
}

impl CoreOrExtension {
    /// Resolves core and extension alternatives without treating raw reporting as enablement.
    pub fn resolve(
        self,
        profile: GlFamilyProfile,
        extensions: &GlExtensionSet,
    ) -> Option<CapabilityEvidence> {
        if profile.meets(self.desktop_core, self.embedded_core) {
            return Some(CapabilityEvidence::Core(profile));
        }
        let extension = self.extension?;
        if !extension.is_legal_for(profile) {
            return None;
        }
        let usable = if self.extension_requires_probe {
            extensions.is_probed(extension)
        } else {
            extensions.is_acquired(extension)
        };
        usable.then_some(CapabilityEvidence::Extension(extension))
    }
}

#[cfg(test)]
mod tests {
    use super::{GlExtensionSet, GlKnownExtension};

    #[test]
    fn compressed_extension_aliases_share_one_typed_ledger_entry() {
        let cases = [
            (
                "WEBGL_compressed_texture_s3tc",
                GlKnownExtension::CompressedTextureS3tc,
            ),
            (
                "GL_EXT_texture_compression_s3tc",
                GlKnownExtension::CompressedTextureS3tc,
            ),
            (
                "GL_ARB_texture_compression_bptc",
                GlKnownExtension::CompressedTextureBptc,
            ),
            (
                "GL_EXT_texture_compression_rgtc",
                GlKnownExtension::CompressedTextureRgtc,
            ),
            (
                "GL_KHR_texture_compression_astc_ldr",
                GlKnownExtension::CompressedTextureAstc,
            ),
            (
                "GL_OES_compressed_ETC2_EAC",
                GlKnownExtension::CompressedTextureEtc,
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(
                GlKnownExtension::from_raw_name(raw),
                Some(expected),
                "{raw}"
            );
        }
        let mut ledger = GlExtensionSet::default();
        ledger.report_raw("WEBGL_compressed_texture_s3tc");
        ledger.report_raw("GL_EXT_texture_compression_s3tc");
        assert!(ledger.acquire(GlKnownExtension::CompressedTextureS3tc));
        assert!(ledger.is_acquired(GlKnownExtension::CompressedTextureS3tc));
    }
}
