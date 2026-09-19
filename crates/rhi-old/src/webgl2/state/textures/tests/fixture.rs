//! The recorder fixture the texture-domain trace tests run against: a domain
//! over a fresh [`MockGlFamilyApi`], the objects a test binds, and the trace
//! readers every assertion is phrased in terms of.
//!
//! Responsibility: build that stage and answer questions about the trace.
//!
//! Not owned here: any assertion.  What a trace *means* is the test's judgement,
//! so this file holds no expectation about which calls should appear; a fixture
//! that encoded one would have to be re-read to see whether a test still tests
//! what its name says.

use super::super::*;
use crate::webgl2::api::tests::{snapshot, texture_desc};
use crate::webgl2::api::{
    GlAddressMode, GlFamilyProfile, GlFilterMode, GlMipmapFilterMode, GlResourceApi, GlSamplerApi,
    GlSamplerDesc, MockCall, MockGlFamilyApi,
};
use crate::webgl2::state::counters::DomainCounters;

/// A sampler description that passes Layer 1's validation.
pub(super) fn sampler_desc() -> GlSamplerDesc {
    GlSamplerDesc {
        address_mode_u: GlAddressMode::ClampToEdge,
        address_mode_v: GlAddressMode::ClampToEdge,
        address_mode_w: GlAddressMode::ClampToEdge,
        mag_filter: GlFilterMode::Linear,
        min_filter: GlFilterMode::Linear,
        mipmap_filter: GlMipmapFilterMode::Linear,
        lod_min_bits: 0.0f32.to_bits(),
        lod_max_bits: 1.0f32.to_bits(),
        compare: None,
        max_anisotropy_bits: None,
    }
}

/// A domain over a fresh recorder, plus the objects and counters a test binds.
///
/// Two textures, not one, because the interesting cases are about *which* object a
/// slot names, and a single texture cannot tell "the mirror forgot the slot" apart
/// from "the mirror still names the same object".
pub(super) struct Fixture {
    pub(super) domain: TexturesState,
    pub(super) backend: MockGlFamilyApi,
    pub(super) first: TextureId,
    pub(super) second: TextureId,
    pub(super) sampler: SamplerId,
    pub(super) counters: StateCounters,
}

impl Fixture {
    pub(super) fn new(mode: ExecutionMode) -> Self {
        let mut backend = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
        let first = backend
            .create_texture_resource(texture_desc(1))
            .expect("texture");
        let second = backend
            .create_texture_resource(texture_desc(1))
            .expect("texture");
        let sampler = backend.create_sampler(sampler_desc()).expect("sampler");
        // Every assertion below is about what the domain emitted, so the fixture's
        // own object creation is cleared rather than subtracted at each call site.
        backend.clear_calls();
        Self {
            domain: TexturesState::new(mode),
            backend,
            first,
            second,
            sampler,
            counters: StateCounters::default(),
        }
    }

    /// Applies the recorded request, which is what a machine entry point does: the
    /// entry point records and the reconcile decides.
    pub(super) fn apply(&mut self) {
        self.reconcile().expect("the request applies");
    }

    pub(super) fn reconcile(&mut self) -> Result<(), StateError> {
        self.domain.reconcile(&mut self.backend, &mut self.counters)
    }

    pub(super) fn invalidate(&mut self, event: StateEvent) {
        self.domain
            .invalidate(&mut self.backend, &event, &mut self.counters);
    }

    pub(super) fn counts(&self) -> &DomainCounters {
        self.counters.domain_counts(StateDomain::Textures)
    }

    /// The number of calls in the recorder's trace.
    pub(super) fn trace_len(&self) -> usize {
        self.backend.calls().len()
    }

    /// The units the trace shows this domain selecting, in order.
    pub(super) fn selected_units(&self) -> Vec<u32> {
        self.backend
            .calls()
            .iter()
            .filter_map(|call| match call {
                MockCall::ActiveTexture(unit) => Some(*unit),
                _ => None,
            })
            .collect()
    }

    /// The texture binds the trace shows, in order.
    pub(super) fn bound_textures(&self) -> Vec<(u32, GlTextureTarget, Option<TextureId>)> {
        self.backend
            .calls()
            .iter()
            .filter_map(|call| match call {
                MockCall::BindTexture {
                    unit,
                    target,
                    texture,
                } => Some((*unit, *target, *texture)),
                _ => None,
            })
            .collect()
    }

    /// The sampler binds the trace shows, in order.
    pub(super) fn bound_samplers(&self) -> Vec<(u32, Option<SamplerId>)> {
        self.backend
            .calls()
            .iter()
            .filter_map(|call| match call {
                MockCall::BindSampler { unit, sampler } => Some((*unit, *sampler)),
                _ => None,
            })
            .collect()
    }
}
