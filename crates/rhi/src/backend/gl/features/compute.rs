use super::{GL42, GL43, GLES31, GlExtension as E, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::Compute => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbComputeShader],
            functions: &[Fn::DispatchCompute],
        },
        F::IndirectDispatch => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbComputeShader],
            functions: &[Fn::DispatchComputeIndirect],
        },
        F::ShaderStorageBuffers => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbShaderStorageBufferObject],
            functions: &[Fn::ShaderStorageBlockBinding],
        },
        F::ImageLoadStore => Requirement {
            desktop_core: GL42,
            gles_core: GLES31,
            extensions: &[E::ArbShaderImageLoadStore],
            functions: &[Fn::BindImageTexture, Fn::MemoryBarrier],
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gl::{features, profile::GlProfile};
    use std::collections::BTreeSet;

    #[test]
    fn compute_is_core_in_43_and_gles31_but_not_webgl2() {
        let functions = BTreeSet::from([Fn::DispatchCompute]);
        let none = BTreeSet::new();
        assert!(features::supports(
            F::Compute,
            GlProfile::Desktop { major: 4, minor: 3 },
            &none,
            &functions
        ));
        assert!(features::supports(
            F::Compute,
            GlProfile::Embedded { major: 3, minor: 1 },
            &none,
            &functions
        ));
        assert!(!features::supports(
            F::Compute,
            GlProfile::WebGl2,
            &none,
            &functions
        ));
    }

    #[test]
    fn compute_extension_still_requires_its_entrypoint() {
        let profile = GlProfile::Desktop { major: 4, minor: 0 };
        let extensions = BTreeSet::from([E::ArbComputeShader]);
        assert!(!features::supports(
            F::Compute,
            profile,
            &extensions,
            &BTreeSet::new()
        ));
        assert_eq!(
            features::missing_function(F::Compute, profile, &extensions, &BTreeSet::new()),
            Some(Fn::DispatchCompute)
        );
    }
}
