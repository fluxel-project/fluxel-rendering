//! Local callback evidence for the S6 typed-feature integration workload.

use std::cell::Cell;

use external_renderer::{PreparedQuads, QuadFeature, QuadInput};
use fluxel_renderer::{
    FeatureContext, FeatureError, ForwardFeaturePass, ForwardResources, PipelineError,
    PreparedFrame, RenderBounds, RenderFeature, TypedFeatureSnapshot,
};

/// Per-attempt observations, deliberately local to the integration fixture.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CallbackCounts {
    pub(super) prepare: u32,
    pub(super) contribute: u32,
}

/// Delegates all typed feature behavior while observing actual current callbacks.
pub(super) struct ObservedQuadFeature {
    inner: QuadFeature,
    prepare: Cell<u32>,
    contribute: Cell<u32>,
}

impl ObservedQuadFeature {
    pub(super) fn new(inner: QuadFeature) -> Self {
        Self {
            inner,
            prepare: Cell::new(0),
            contribute: Cell::new(0),
        }
    }

    pub(super) fn callback_counts(&self) -> CallbackCounts {
        CallbackCounts {
            prepare: self.prepare.get(),
            contribute: self.contribute.get(),
        }
    }

    pub(super) fn set_enabled(&mut self, enabled: bool) {
        self.inner.set_enabled(enabled);
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }
}

impl RenderFeature for ObservedQuadFeature {
    type Input = QuadInput;
    type Prepared = PreparedQuads;

    fn handle(&self) -> fluxel_renderer::FeatureHandle {
        self.inner.handle()
    }

    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
        frame: &PreparedFrame<'_, ForwardResources>,
        visibility: RenderBounds,
    ) -> Result<Self::Prepared, FeatureError> {
        self.prepare.set(self.prepare.get() + 1);
        self.inner.prepare(context, input, frame, visibility)
    }

    fn draws<'a>(&self, prepared: &'a Self::Prepared) -> &'a [fluxel_renderer::ForwardDrawRequest] {
        self.inner.draws(prepared)
    }

    fn contribute(
        &self,
        prepared: &Self::Prepared,
        pass: &mut ForwardFeaturePass<'_, '_>,
    ) -> Result<(), PipelineError> {
        self.contribute.set(self.contribute.get() + 1);
        self.inner.contribute(prepared, pass)
    }
}
