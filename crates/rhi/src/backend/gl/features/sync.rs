//! Synchronisation routes. `ARB_texture_barrier` and `NV_texture_barrier`
//! remain distinct evidence because the latter uses a different entry token.
use super::{
    GL32, GL42, GL45, GLES30, GLES31, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::Sync => Requirement {
            desktop_core: GL32,
            gles_core: GLES30,
            extensions: &[E::ArbSync],
            functions: &[Fn::FenceSync, Fn::ClientWaitSync, Fn::DeleteSync],
        },
        F::MemoryBarrier => Requirement {
            desktop_core: GL42,
            gles_core: GLES31,
            extensions: &[E::ArbShaderImageLoadStore],
            functions: &[Fn::MemoryBarrier],
        },
        F::MemoryBarrierByRegion => Requirement {
            desktop_core: GL45,
            gles_core: GLES31,
            extensions: &[],
            functions: &[Fn::MemoryBarrierByRegion],
        },
        // ARB/core use `glTextureBarrier`; NV uses `glTextureBarrierNV`.
        // `GlFeatureProbe` chooses the route and checks its exact token.
        F::TextureBarrier => Requirement {
            desktop_core: GL45,
            gles_core: NEVER,
            extensions: &[E::ArbTextureBarrier, E::NvTextureBarrier],
            functions: &[],
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::backend::gl::{
        features::{self, GlFunction},
        profile::GlProfile,
    };

    fn set(items: &[GlFunction]) -> BTreeSet<GlFunction> {
        items.iter().copied().collect()
    }

    #[test]
    fn sync_accepts_core_and_arb_but_needs_every_token() {
        let profile = GlProfile::Desktop { major: 4, minor: 0 };
        let mut extensions = BTreeSet::new();
        extensions.insert(E::ArbSync);
        let required = [Fn::FenceSync, Fn::ClientWaitSync, Fn::DeleteSync];
        assert!(!features::supports(
            F::Sync,
            profile,
            &extensions,
            &set(&required[..2])
        ));
        assert_eq!(
            features::missing_function(F::Sync, profile, &extensions, &set(&required[..2])),
            Some(Fn::DeleteSync)
        );
        assert!(features::supports(
            F::Sync,
            profile,
            &extensions,
            &set(&required)
        ));
    }

    #[test]
    fn barriers_are_profile_and_token_precise() {
        let none = BTreeSet::new();
        assert!(!features::supports(
            F::MemoryBarrier,
            GlProfile::WebGl2,
            &none,
            &set(&[Fn::MemoryBarrier])
        ));
        let gles = GlProfile::Embedded { major: 3, minor: 1 };
        assert!(features::supports(
            F::MemoryBarrier,
            gles,
            &none,
            &set(&[Fn::MemoryBarrier])
        ));
        assert!(features::supports(
            F::MemoryBarrierByRegion,
            gles,
            &none,
            &set(&[Fn::MemoryBarrierByRegion])
        ));
        assert!(!features::supports(
            F::MemoryBarrierByRegion,
            GlProfile::Desktop { major: 4, minor: 0 },
            &none,
            &set(&[Fn::MemoryBarrierByRegion])
        ));
    }

    #[test]
    fn texture_barrier_retains_core_arb_and_nv_tokens() {
        let empty = BTreeSet::new();
        assert!(features::supports(
            F::TextureBarrier,
            GlProfile::Desktop { major: 4, minor: 5 },
            &empty,
            &set(&[Fn::TextureBarrier])
        ));
        let mut arb = BTreeSet::new();
        arb.insert(E::ArbTextureBarrier);
        assert!(features::supports(
            F::TextureBarrier,
            GlProfile::Desktop { major: 4, minor: 0 },
            &arb,
            &set(&[Fn::TextureBarrier])
        ));
        let mut nv = BTreeSet::new();
        nv.insert(E::NvTextureBarrier);
        assert!(!features::supports(
            F::TextureBarrier,
            GlProfile::Desktop { major: 4, minor: 0 },
            &nv,
            &set(&[Fn::TextureBarrier])
        ));
        assert!(features::supports(
            F::TextureBarrier,
            GlProfile::Desktop { major: 4, minor: 0 },
            &nv,
            &set(&[Fn::TextureBarrierNv])
        ));
    }
}
