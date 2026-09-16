//! Typed extension evidence and fail-closed core-or-extension resolution.
//!
//! # Inventory decisions
//!
//! Every registry name the accepted profiles can report is either typed with a
//! real command path or recorded from its exact spelling with a written reason
//! for having none. An entry without a command path still keeps its provenance
//! and stays visible in `raw_reported_names`, but no `CoreOrExtension` row
//! resolves against it, so it acquires nothing and enables nothing:
//!
//! - `WEBGL_multi_draw_instanced_base_vertex_base_instance` is a working draft
//!   with no ratified entry-point set and no browser shipping it as a stable
//!   API, and its commands are a strict refinement (per-draw base vertex/base
//!   instance) of the normalized advanced-draw domain, whose browser path does
//!   not exist yet. A typed wrapper would own a call path nothing can prove, so
//!   the name is recorded raw-only until a browser advanced-draw domain exists
//!   to consume it.
//! - `WEBGL_shader_pixel_local_storage` is an isolated fragment-local semantic
//!   with no matching RHI semantic in this series; the plan forbids exposing it
//!   as compute, storage-buffer, or general storage-image capability. It is
//!   recorded raw-only rather than mapped to a domain whose semantics differ.
//! - The first multiview revision (`GL_OVR_multiview`) is not aliased onto the
//!   typed multiview name: that name carries the second revision's semantics,
//!   and aliasing revisions would claim a shader contract this contract does
//!   not implement. It remains an untyped raw name.
//!
//! Provenance is not capability: recording a name here never enables a
//! capability on its own.

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
    /// WebGL draft per-draw base-vertex/base-instance multi-draw commands.
    ///
    /// Recorded raw-only: see the inventory decisions above this enum.
    WebglMultiDrawInstancedBaseVertexBaseInstance,
    /// WebGL multiview.
    OvrMultiview2,
    /// WebGL fragment pixel-local storage.
    ///
    /// Recorded raw-only: see the inventory decisions above this enum.
    WebglShaderPixelLocalStorage,
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
            Self::WebglMultiDrawInstancedBaseVertexBaseInstance => {
                "WEBGL_multi_draw_instanced_base_vertex_base_instance"
            }
            Self::OvrMultiview2 => "OVR_multiview2",
            Self::WebglShaderPixelLocalStorage => "WEBGL_shader_pixel_local_storage",
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
            Self::WebglMultiDrawInstancedBaseVertexBaseInstance,
            Self::OvrMultiview2,
            Self::WebglShaderPixelLocalStorage,
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
                || (*known == Self::OvrMultiview2 && name == "GL_OVR_multiview2")
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
            | Self::WebglMultiDrawInstancedBaseVertexBaseInstance
            | Self::WebglShaderPixelLocalStorage
            | Self::KhrParallelShaderCompile => matches!(profile, GlFamilyProfile::WebGl2),
            // Multiview exists for the embedded family and for WebGL2. The
            // desktop core profile offers no multiview route in this contract,
            // so a desktop context resolves no evidence and stays fail-closed.
            Self::OvrMultiview2 => matches!(
                profile,
                GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2
            ),
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
    use super::{
        CapabilityEvidence, CoreOrExtension, ExtensionProvenance, GlExtensionSet, GlFamilyProfile,
        GlKnownExtension,
    };

    /// Every typed name in the inventory, so a new variant cannot be added
    /// without a registry spelling that maps back to it.
    const INVENTORY: [GlKnownExtension; 21] = [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtDisjointTimerQueryWebgl2,
        GlKnownExtension::ExtColorBufferFloat,
        GlKnownExtension::ExtFloatBlend,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::WebglMultiDraw,
        GlKnownExtension::WebglMultiDrawInstancedBaseVertexBaseInstance,
        GlKnownExtension::OvrMultiview2,
        GlKnownExtension::WebglShaderPixelLocalStorage,
        GlKnownExtension::KhrParallelShaderCompile,
        GlKnownExtension::KhrRobustness,
        GlKnownExtension::KhrDebug,
        GlKnownExtension::CompressedTextureS3tc,
        GlKnownExtension::CompressedTextureS3tcSrgb,
        GlKnownExtension::CompressedTextureBptc,
        GlKnownExtension::CompressedTextureRgtc,
        GlKnownExtension::CompressedTextureAstc,
        GlKnownExtension::CompressedTextureEtc,
    ];

    #[test]
    fn every_typed_name_maps_back_from_its_own_registry_spelling() {
        for known in INVENTORY {
            assert_eq!(
                GlKnownExtension::from_raw_name(known.raw_name()),
                Some(known),
                "{known:?}"
            );
        }
        // A name that is not in the inventory stays untyped rather than being
        // guessed at by prefix.
        assert_eq!(GlKnownExtension::from_raw_name("WEBGL_unknown_thing"), None);
        assert_eq!(GlKnownExtension::from_raw_name("webgl_multi_draw"), None);
    }

    /// The two inventory entries kept without a command path are reachable by
    /// their exact spelling, so a context that reports one is recorded rather
    /// than dropped, while nothing in the ledger can make them usable.
    #[test]
    fn draft_names_are_recorded_in_the_ledger_without_a_typed_route() {
        let drafts = [
            (
                "WEBGL_multi_draw_instanced_base_vertex_base_instance",
                GlKnownExtension::WebglMultiDrawInstancedBaseVertexBaseInstance,
            ),
            (
                "WEBGL_shader_pixel_local_storage",
                GlKnownExtension::WebglShaderPixelLocalStorage,
            ),
        ];
        for (name, known) in drafts {
            assert_eq!(GlKnownExtension::from_raw_name(name), Some(known), "{name}");
            let mut ledger = GlExtensionSet::default();
            ledger.report_raw(name);
            assert_eq!(ledger.raw_reported_names().collect::<Vec<_>>(), [name]);
            // Reporting is the whole of it: no row resolves against these, and
            // the draft spellings are not accepted under any other casing.
            assert_eq!(
                ledger.provenance(known),
                Some(ExtensionProvenance::Reported)
            );
            assert!(!ledger.is_acquired(known));
            assert_eq!(GlKnownExtension::from_raw_name(&name.to_lowercase()), None);
        }
    }

    /// Only the second multiview revision is aliased onto the typed name: the
    /// first carries a shader contract this series does not implement, so it
    /// stays an untyped raw name that enables nothing.
    #[test]
    fn only_the_second_multiview_revision_reaches_the_typed_name() {
        assert_eq!(
            GlKnownExtension::from_raw_name("GL_OVR_multiview2"),
            Some(GlKnownExtension::OvrMultiview2)
        );
        assert_eq!(GlKnownExtension::from_raw_name("GL_OVR_multiview"), None);
        let mut ledger = GlExtensionSet::default();
        ledger.report_raw("GL_OVR_multiview");
        assert_eq!(ledger.provenance(GlKnownExtension::OvrMultiview2), None);
        assert_eq!(
            ledger.raw_reported_names().collect::<Vec<_>>(),
            ["GL_OVR_multiview"]
        );
    }

    /// A route is legal only where its family can report the name, which is
    /// what keeps a desktop context from ever resolving multiview evidence.
    #[test]
    fn a_typed_route_is_legal_only_inside_its_own_family() {
        let webgl = GlFamilyProfile::WebGl2;
        let embedded = GlFamilyProfile::Embedded { major: 3, minor: 1 };
        let desktop = GlFamilyProfile::Desktop { major: 4, minor: 3 };
        assert!(GlKnownExtension::WebglMultiDraw.is_legal_for(webgl));
        assert!(!GlKnownExtension::WebglMultiDraw.is_legal_for(embedded));
        assert!(!GlKnownExtension::WebglMultiDraw.is_legal_for(desktop));
        assert!(GlKnownExtension::OvrMultiview2.is_legal_for(webgl));
        assert!(GlKnownExtension::OvrMultiview2.is_legal_for(embedded));
        assert!(!GlKnownExtension::OvrMultiview2.is_legal_for(desktop));
    }

    /// The three evidence states a route can pass through, in order: reporting
    /// enables nothing, acquiring enables a route with no probe, and a probed
    /// route enables one that requires it.
    #[test]
    fn resolve_requires_the_exact_evidence_its_route_asks_for() {
        let profile = GlFamilyProfile::WebGl2;
        let probed_route = CoreOrExtension {
            desktop_core: None,
            embedded_core: None,
            extension: Some(GlKnownExtension::OvrMultiview2),
            extension_requires_probe: true,
        };
        let direct_route = CoreOrExtension {
            extension_requires_probe: false,
            ..probed_route
        };
        let mut ledger = GlExtensionSet::default();
        ledger.report_raw("OVR_multiview2");
        assert_eq!(probed_route.resolve(profile, &ledger), None);
        assert_eq!(direct_route.resolve(profile, &ledger), None);
        assert!(ledger.acquire(GlKnownExtension::OvrMultiview2));
        assert_eq!(
            direct_route.resolve(profile, &ledger),
            Some(CapabilityEvidence::Extension(
                GlKnownExtension::OvrMultiview2
            ))
        );
        assert_eq!(probed_route.resolve(profile, &ledger), None);
        assert!(ledger.probe(GlKnownExtension::OvrMultiview2));
        assert_eq!(
            probed_route.resolve(profile, &ledger),
            Some(CapabilityEvidence::Extension(
                GlKnownExtension::OvrMultiview2
            ))
        );
        // A failed acquisition is terminal: no later report can revive it.
        let mut failed = GlExtensionSet::default();
        failed.report_raw("OVR_multiview2");
        assert!(failed.fail(GlKnownExtension::OvrMultiview2));
        assert!(!failed.acquire(GlKnownExtension::OvrMultiview2));
        assert_eq!(direct_route.resolve(profile, &failed), None);
    }

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
