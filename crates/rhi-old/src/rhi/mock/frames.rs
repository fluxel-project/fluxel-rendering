//! The mock's presentation targets, leases, and acquired frames.
//!
//! The mock answers presentation capability queries from one fixed snapshot and
//! hands out one windowless drawable at a time, which is the P0 rule: at most
//! one outstanding frame per configured presentation.
//!
//! Frame state lives behind interior mutability shared by the frame token and
//! its attachment, because an attachment handed out before a state change must
//! observe that change afterwards. That sharing is not a mock convenience: a
//! real backend's native drawable has the same property.
//!
//! # What it deliberately does not model
//!
//! [`ConfiguredPresentationBackend::reconfigure`] is refused. The trait returns
//! `&PresentationConfiguration` borrowed from `&self` while `reconfigure` also
//! takes `&self`, so an implementation cannot replace the configuration and go
//! on handing out a borrow of it without interior mutability that yields
//! borrowed data. The mock declines rather than keeping a second copy that
//! could silently disagree with the first.

use std::sync::{Arc, Mutex};

use crate::rhi::format::TextureFormat;
use crate::rhi::platform::{next_object_id, DeviceIdentity, ObjectId, RhiError, RhiResult};
use crate::rhi::presentation::{
    AcquireError, AcquireErrorKind, AcquiredFrame, AcquiredFrameBackend, AcquiredFrameId,
    AcquiredFrameState, ConfiguredPresentation, ConfiguredPresentationBackend, Extent2d,
    FrameAttachment, FrameAttachmentBackend, PresentationConfiguration, PresentationExtent,
    PresentationExtentControl, PresentationTarget, PresentationTargetCapabilities,
};
use crate::rhi::resource::Extent3d;

use super::{Injected, Mock, MockState};

/// The drawable extent the mock's windowless target reports.
const MOCK_DRAWABLE_EXTENT: Extent2d = Extent2d {
    width: 1280,
    height: 720,
};

/// The surface facts the mock reports for every target.
///
/// [`PresentMode::Automatic`] is added by the constructor, so it is absent here
/// on purpose: the mock does not list facts it does not choose.
pub(super) fn target_capabilities() -> PresentationTargetCapabilities {
    PresentationTargetCapabilities::new(
        vec![TextureFormat::Bgra8UnormSrgb],
        Vec::new(),
        PresentationExtentControl::HostManaged {
            current: Some(MOCK_DRAWABLE_EXTENT),
        },
    )
}

/// The host payload the mock's targets carry.
///
/// It holds no native surface, because the mock has none. It does hold the
/// identity of the device that produced it, which is what lets
/// `configure_presentation` refuse a target from another provider instead of
/// configuring a drawable this backend does not own. One `Mock` is one provider
/// in this fixture, so the device identity is the marker; a real provider with
/// several devices would carry its own provider identity instead.
pub(super) struct MockTargetPayload {
    /// The identity of the device that handed out this target.
    pub(super) producer: DeviceIdentity,
}

/// The mutable half of one configured presentation.
#[derive(Default)]
struct PresentationState {
    /// The serial of the frame that is outstanding, if any.
    outstanding: Option<u64>,
}

/// The backend half of the mock's [`ConfiguredPresentation`].
pub(super) struct MockConfiguredPresentationBackend {
    id: ObjectId,
    device: DeviceIdentity,
    target_id: ObjectId,
    configuration: PresentationConfiguration,
    state: Arc<Mutex<PresentationState>>,
    journal: Arc<MockState>,
}

impl MockConfiguredPresentationBackend {
    /// A lease over a fresh target, configured as requested.
    pub(super) fn new(
        device: DeviceIdentity,
        journal: Arc<MockState>,
        configuration: PresentationConfiguration,
    ) -> Self {
        Self {
            id: next_object_id(),
            device,
            target_id: next_object_id(),
            configuration,
            state: Arc::new(Mutex::new(PresentationState::default())),
            journal,
        }
    }

    /// The texel extent an acquired frame reports.
    ///
    /// The configuration cannot request an exact extent from a host-managed
    /// target, so the snapshot's current extent is the only answer.
    fn drawable_extent(&self) -> Extent3d {
        match self.configuration.extent() {
            PresentationExtent::Exact(extent) => extent.to_extent3d(),
            PresentationExtent::HostManaged => MOCK_DRAWABLE_EXTENT.to_extent3d(),
        }
    }

    /// Builds a frame token over `serial`.
    fn token(&self, serial: u64) -> AcquiredFrame {
        let frame = Arc::new(MockFrame {
            id: AcquiredFrameId::new(self.device, serial),
            device: self.device,
            format: self.configuration.format(),
            extent: self.drawable_extent(),
            state: Mutex::new(AcquiredFrameState::Acquired),
            presentation: Arc::clone(&self.state),
            journal: Arc::clone(&self.journal),
        });
        AcquiredFrame::new(Arc::new(MockAcquiredFrame { frame }) as Arc<dyn AcquiredFrameBackend>)
    }

    /// Hands out a second token for the frame that is already outstanding.
    ///
    /// This is fault injection, not supported behaviour. It models a backend
    /// that returns the same drawable twice, which is the one case the plan's
    /// "a frame is presented at most once" rule can be reached with: the
    /// ordinary second `present_after` is refused earlier, by the frame's own
    /// state, because a frame that is already `PlannedForPresent` cannot be
    /// planned again.
    pub(super) fn duplicate_outstanding_frame(&self) -> Option<AcquiredFrame> {
        let serial = MockState::lock(&self.state).outstanding?;
        Some(self.token(serial))
    }
}

impl ConfiguredPresentationBackend for MockConfiguredPresentationBackend {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn target_id(&self) -> ObjectId {
        self.target_id
    }

    fn configuration(&self) -> &PresentationConfiguration {
        &self.configuration
    }

    fn reconfigure(&self, _configuration: &PresentationConfiguration) -> RhiResult<()> {
        self.journal.note("presentation:reconfigure");
        Err(RhiError::unsupported(
            "the mock does not model presentation reconfiguration",
        ))
    }

    fn acquire(&self) -> Result<AcquiredFrame, AcquireError> {
        self.journal.note("presentation:acquire");
        if self.journal.consume(Injected::Acquire) {
            return Err(AcquireError::new(
                AcquireErrorKind::NotReady,
                "the mock backend was told to fail the next acquire",
            ));
        }
        let mut state = MockState::lock(&self.state);
        if state.outstanding.is_some() {
            return Err(AcquireError::new(
                AcquireErrorKind::FrameOutstanding,
                "the previous frame is still outstanding",
            ));
        }
        let serial = self.journal.next_frame_serial();
        state.outstanding = Some(serial);
        drop(state);
        Ok(self.token(serial))
    }
}

/// The state of one acquired frame, shared by its token and its attachment.
struct MockFrame {
    id: AcquiredFrameId,
    device: DeviceIdentity,
    format: TextureFormat,
    extent: Extent3d,
    state: Mutex<AcquiredFrameState>,
    presentation: Arc<Mutex<PresentationState>>,
    journal: Arc<MockState>,
}

impl MockFrame {
    /// The frame's current lifecycle state.
    fn state(&self) -> AcquiredFrameState {
        *MockState::lock(&self.state)
    }

    /// Moves the frame to `next`.
    fn set_state(&self, next: AcquiredFrameState) {
        *MockState::lock(&self.state) = next;
    }

    /// Releases the presentation's outstanding slot when this frame holds it.
    ///
    /// A frame that was never the outstanding one must not clear a slot that a
    /// later acquisition owns, so the release is keyed on the serial.
    fn release(&self) {
        let mut presentation = MockState::lock(&self.presentation);
        if presentation.outstanding == Some(self.id.serial()) {
            presentation.outstanding = None;
        }
    }
}

impl FrameAttachmentBackend for MockFrame {
    fn frame_id(&self) -> AcquiredFrameId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn format(&self) -> TextureFormat {
        self.format
    }

    fn extent(&self) -> Extent3d {
        self.extent
    }

    fn sample_count(&self) -> u32 {
        1
    }

    fn state(&self) -> AcquiredFrameState {
        MockFrame::state(self)
    }
}

/// The frame-token half of one acquired frame.
///
/// It is separate from [`MockFrame`] because `attachment` has to hand out a
/// second reference to the same state, and a trait method that takes `&self`
/// cannot produce an `Arc` of itself.
struct MockAcquiredFrame {
    frame: Arc<MockFrame>,
}

impl AcquiredFrameBackend for MockAcquiredFrame {
    fn frame_id(&self) -> AcquiredFrameId {
        self.frame.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.frame.device
    }

    fn state(&self) -> AcquiredFrameState {
        MockFrame::state(&self.frame)
    }

    fn attachment(&self) -> FrameAttachment {
        FrameAttachment::new(Arc::clone(&self.frame) as Arc<dyn FrameAttachmentBackend>)
    }

    fn abandon(&self) -> RhiResult<()> {
        self.frame.journal.note("frame:abandon");
        let state = self.frame.state();
        if state != AcquiredFrameState::Acquired {
            return Err(RhiError::invalid_usage(format!(
                "the frame is {state:?}, so it cannot be abandoned"
            )));
        }
        self.frame.set_state(AcquiredFrameState::Abandoned);
        self.frame.release();
        Ok(())
    }

    fn abandon_on_drop(&self) {
        self.frame.journal.note("frame:abandon_on_drop");
        // Drop cannot report an error, so a frame that is still promised to a
        // present is abandoned as well: the alternative is leaving the
        // presentation system holding the image forever.
        let state = self.frame.state();
        if state == AcquiredFrameState::Acquired || state == AcquiredFrameState::PlannedForPresent {
            self.frame.set_state(AcquiredFrameState::Abandoned);
        }
        self.frame.release();
    }

    fn plan_for_present(&self) -> RhiResult<()> {
        let state = self.frame.state();
        if state != AcquiredFrameState::Acquired {
            return Err(RhiError::invalid_usage(format!(
                "the frame is {state:?}, so it cannot be planned for present"
            )));
        }
        self.frame.journal.note("frame:plan_for_present");
        self.frame.set_state(AcquiredFrameState::PlannedForPresent);
        Ok(())
    }
}

impl Mock {
    /// A presentation target over a windowless payload.
    pub(crate) fn target(&self) -> PresentationTarget {
        PresentationTarget::new(Arc::new(MockTargetPayload {
            producer: self.identity(),
        }))
    }

    /// The mock's configured presentation.
    ///
    /// Every call returns a façade over the same lease, so "one outstanding
    /// frame" is a property of the mock rather than of one clone.
    pub(crate) fn presentation(&self) -> ConfiguredPresentation {
        ConfiguredPresentation::new(
            Arc::clone(self.lease()) as Arc<dyn ConfiguredPresentationBackend>
        )
    }

    /// The lease backend, created on first use.
    ///
    /// It is built by the same constructor `Device::configure_presentation`
    /// uses, and it is deliberately the *same* lease on every call: "one
    /// outstanding frame" is a property of a configured presentation, so a
    /// fixture that made a new lease per call could not observe it.
    fn lease(&self) -> &Arc<MockConfiguredPresentationBackend> {
        self.presentation_state.get_or_init(|| {
            Arc::new(MockConfiguredPresentationBackend::new(
                self.identity(),
                Arc::clone(&self.state),
                PresentationConfiguration::new(TextureFormat::Bgra8UnormSrgb),
            ))
        })
    }

    /// Acquires the mock's next frame.
    ///
    /// It goes through [`Mock::presentation`], the public façade, rather than
    /// calling the lease backend directly: the façade is what a caller holds,
    /// and a fixture that bypassed it would not exercise the path a real caller
    /// takes.
    pub(crate) fn try_frame(&self) -> Result<AcquiredFrame, AcquireError> {
        self.presentation().acquire()
    }

    /// Acquires the mock's next frame, panicking if the backend refuses.
    ///
    /// Use [`Mock::try_frame`] when the refusal is the point of the test.
    pub(crate) fn frame(&self) -> AcquiredFrame {
        self.try_frame()
            .expect("the mock acquires a frame unless a failure was injected")
    }

    /// A second token for the frame that is currently outstanding.
    ///
    /// Fault injection for the plan's once-per-frame present rule; see
    /// [`MockConfiguredPresentationBackend::duplicate_outstanding_frame`].
    pub(crate) fn duplicate_outstanding_frame(&self) -> Option<AcquiredFrame> {
        self.lease().duplicate_outstanding_frame()
    }
}
