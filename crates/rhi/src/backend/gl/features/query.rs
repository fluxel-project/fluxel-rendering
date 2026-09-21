use super::{GL15, GLES30, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::OcclusionQuery => Requirement {
            desktop_core: GL15,
            gles_core: GLES30,
            extensions: &[],
            functions: &[Fn::BeginQuery, Fn::EndQuery, Fn::GetQueryObject],
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
    fn occlusion_query_needs_every_entrypoint_even_at_core_floor() {
        let profile = GlProfile::Embedded { major: 3, minor: 0 };
        let partial = BTreeSet::from([Fn::BeginQuery, Fn::EndQuery]);
        assert!(!features::supports(
            F::OcclusionQuery,
            profile,
            &BTreeSet::new(),
            &partial
        ));
        assert_eq!(
            features::missing_function(F::OcclusionQuery, profile, &BTreeSet::new(), &partial),
            Some(Fn::GetQueryObject)
        );
    }

    #[test]
    fn webgl2_does_not_inherit_gles_query_core_route() {
        let functions = BTreeSet::from([Fn::BeginQuery, Fn::EndQuery, Fn::GetQueryObject]);
        assert!(!features::supports(
            F::OcclusionQuery,
            GlProfile::WebGl2,
            &BTreeSet::new(),
            &functions
        ));
    }
}
