use super::{GL43, GLES32, GlExtension as E, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::DebugMarkers => Requirement {
            desktop_core: GL43,
            gles_core: GLES32,
            extensions: &[E::KhrDebug],
            functions: &[Fn::PushDebugGroup, Fn::PopDebugGroup, Fn::ObjectLabel],
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
    fn debug_route_needs_push_pop_and_label() {
        let profile = GlProfile::Desktop { major: 4, minor: 3 };
        let partial = BTreeSet::from([Fn::PushDebugGroup, Fn::PopDebugGroup]);
        assert!(!features::supports(
            F::DebugMarkers,
            profile,
            &BTreeSet::new(),
            &partial
        ));
        assert_eq!(
            features::missing_function(F::DebugMarkers, profile, &BTreeSet::new(), &partial),
            Some(Fn::ObjectLabel)
        );
    }

    #[test]
    fn webgl2_does_not_infer_khr_debug_from_browser_profile() {
        let functions = BTreeSet::from([Fn::PushDebugGroup, Fn::PopDebugGroup, Fn::ObjectLabel]);
        assert!(!features::supports(
            F::DebugMarkers,
            GlProfile::WebGl2,
            &BTreeSet::new(),
            &functions
        ));
    }
}
