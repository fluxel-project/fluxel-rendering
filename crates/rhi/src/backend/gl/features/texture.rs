use super::{
    GL30, GL42, GL43, GLES30, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::SamplerAnisotropy => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::ExtTextureFilterAnisotropic],
            functions: &[Fn::TexParameterAnisotropy],
        },
        F::CompressionBcS3tc => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::ExtTextureCompressionS3tc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionBcRgtc => Requirement {
            desktop_core: GL30,
            gles_core: NEVER,
            extensions: &[E::ExtTextureCompressionRgtc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionBcBptc => Requirement {
            desktop_core: GL42,
            gles_core: NEVER,
            extensions: &[E::ArbTextureCompressionBptc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionEtc2Eac => Requirement {
            desktop_core: GL43,
            gles_core: GLES30,
            extensions: &[E::ArbEs3Compatibility, E::OesCompressedEtc2Rgb8Texture],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionAstcLdr => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::KhrTextureCompressionAstcLdr],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionAstcHdr => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::KhrTextureCompressionAstcHdr],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::Multiview => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::OvrMultiview2],
            functions: &[Fn::FramebufferTextureMultiview],
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::{features, profile::GlProfile};
    use std::collections::BTreeSet;

    fn compressed_entries() -> BTreeSet<Fn> {
        BTreeSet::from([Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d])
    }

    #[test]
    fn etc2_is_gles30_and_gl43_core_but_not_webgl2_by_inference() {
        let functions = compressed_entries();
        let none = BTreeSet::new();
        assert!(features::supports(
            F::CompressionEtc2Eac,
            GlProfile::Embedded { major: 3, minor: 0 },
            &none,
            &functions
        ));
        assert!(features::supports(
            F::CompressionEtc2Eac,
            GlProfile::Desktop { major: 4, minor: 3 },
            &none,
            &functions
        ));
        assert!(!features::supports(
            F::CompressionEtc2Eac,
            GlProfile::WebGl2,
            &none,
            &functions
        ));
    }

    #[test]
    fn bc_aggregate_requires_every_bc_family_and_both_upload_tokens() {
        // RGTC is desktop core since GL 3.0 and BPTC since GL 4.2.  Test the
        // extension fallback below 4.2 rather than pretending those core
        // families still need their extension tokens on a GL 4.2 context.
        let profile = GlProfile::Desktop { major: 4, minor: 1 };
        let mut extensions = BTreeSet::from([E::ExtTextureCompressionS3tc]);
        assert!(!features::supports(
            F::CompressionBc,
            profile,
            &extensions,
            &compressed_entries()
        ));
        extensions.insert(E::ArbTextureCompressionBptc);
        assert!(features::supports(
            F::CompressionBc,
            profile,
            &extensions,
            &compressed_entries()
        ));
        assert!(!features::supports(
            F::CompressionBc,
            profile,
            &extensions,
            &BTreeSet::from([Fn::CompressedTexImage2d])
        ));

        // On GL 4.2, S3TC is the only BC family that still needs an
        // extension route: RGTC and BPTC are both core.
        assert!(features::supports(
            F::CompressionBc,
            GlProfile::Desktop { major: 4, minor: 2 },
            &BTreeSet::from([E::ExtTextureCompressionS3tc]),
            &compressed_entries()
        ));
    }
}
