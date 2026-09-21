use super::{
    GL40, GL43, GL46, GLES31, GLES32, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::IndirectDraw => Requirement {
            desktop_core: GL40,
            gles_core: GLES31,
            extensions: &[E::ArbDrawIndirect],
            functions: &[Fn::DrawArraysIndirect, Fn::DrawElementsIndirect],
        },
        F::MultiDrawIndirect => Requirement {
            desktop_core: GL43,
            gles_core: GLES32,
            extensions: &[E::ArbMultiDrawIndirect],
            functions: &[Fn::MultiDrawArraysIndirect, Fn::MultiDrawElementsIndirect],
        },
        F::IndirectCount => Requirement {
            desktop_core: GL46,
            gles_core: NEVER,
            extensions: &[E::ArbIndirectParameters],
            functions: &[
                Fn::MultiDrawArraysIndirectCount,
                Fn::MultiDrawElementsIndirectCount,
            ],
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
    fn indirect_draw_obeys_profile_floor_and_both_tokens() {
        let one = BTreeSet::from([Fn::DrawArraysIndirect]);
        let both = BTreeSet::from([Fn::DrawArraysIndirect, Fn::DrawElementsIndirect]);
        let none = BTreeSet::new();
        assert!(!features::supports(
            F::IndirectDraw,
            GlProfile::Desktop { major: 3, minor: 3 },
            &none,
            &both
        ));
        assert!(!features::supports(
            F::IndirectDraw,
            GlProfile::Desktop { major: 4, minor: 0 },
            &none,
            &one
        ));
        assert!(features::supports(
            F::IndirectDraw,
            GlProfile::Desktop { major: 4, minor: 0 },
            &none,
            &both
        ));
    }

    #[test]
    fn indirect_count_is_never_in_gles_core() {
        let functions = BTreeSet::from([
            Fn::MultiDrawArraysIndirectCount,
            Fn::MultiDrawElementsIndirectCount,
        ]);
        assert!(!features::supports(
            F::IndirectCount,
            GlProfile::Embedded { major: 3, minor: 2 },
            &BTreeSet::new(),
            &functions
        ));
    }
}
