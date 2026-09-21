//! Portable presentation lifecycle conformance workload.
//!
//! A presentation target is deliberately host-owned: an `HWND`,
//! `ANativeWindow`, `CAMetalLayer`, or browser canvas must not escape into the
//! RHI API merely so this test can run.  Therefore a backend fixture supplies
//! the already registered [`PresentationTarget`], a configuration selected from
//! that target's *current* facts, and a raster-capable logical lane.  Everything
//! after that boundary is one portable contract: configure, acquire, clear the
//! frame attachment, submit/present, observe the independent GPU and present
//! outcomes, reconfigure, and prove an abandoned frame releases the lease.
//!
//! The fixture must not replace this with a backend-local frame loop.  In
//! particular, a successful native `Present` is insufficient evidence: this
//! workload waits for both `CompletionPoint` and `PresentReceiptId`, because
//! those outcomes intentionally have independent terminal states in v13.

use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecorderDescriptor, StoreOp,
};
use crate::api::platform::Device;
use crate::api::presentation::{PresentState, PresentationConfiguration, PresentationTarget};
use crate::api::shader::ShaderLocation;
use crate::api::submission::{SubmissionLaneId, SubmissionPlanBuilder};
use crate::backend::test_harness::require_complete;

/// Host/backend inputs for [`clear_present_reconfigure`].
///
/// The target registration and configuration selection are fixture work because
/// both require native host facts.  `lane` remains a portable logical lane: a
/// fixture must obtain it from `Device::capabilities()`, never manufacture a
/// native queue identity for this workload.
pub(crate) struct PresentationFixture<'a> {
    /// Live public device created by the fixture's private provider bridge.
    pub(crate) device: &'a Device,
    /// Opaque target registered by the fixture's host integration.
    pub(crate) target: &'a PresentationTarget,
    /// A configuration freshly chosen from `presentation_capabilities(target)`.
    pub(crate) configuration: PresentationConfiguration,
    /// A lane advertising raster work.
    pub(crate) lane: SubmissionLaneId,
}

/// Runs the baseline portable surface lifecycle.
///
/// The first acquired frame is cleared and consumed by `present_after`.  The
/// function then waits for GPU completion *and* for `PresentState::Accepted`,
/// reconfigures the same lease, and finally acquires/abandons another frame.
/// The last pair catches a common backend error where only present releases the
/// outstanding-frame record.  Window resize, Android lifecycle callbacks and
/// browser canvas resize are deliberately outside this workload: they are host
/// fixtures which should invoke this same contract before/after their native
/// event.
///
/// # Panics
///
/// Panics when a fixture advertises a raster/presentation route but the route
/// rejects this baseline sequence, reports a non-complete GPU terminal state,
/// or returns a non-accepted presentation.  A missing target, host runtime, or
/// unpublished capability must be classified as `Skipped`/`Unsupported` by the
/// fixture before calling this function.
pub(crate) async fn clear_present_reconfigure(fixture: PresentationFixture<'_>, label: &str) {
    let PresentationFixture {
        device,
        target,
        configuration,
        lane,
    } = fixture;

    let mut surface = device
        .configure_presentation(target, &configuration)
        .await
        .unwrap_or_else(|error| {
            panic!("{label}: advertised presentation configuration failed: {error}")
        });
    let frame = surface.acquire().await.unwrap_or_else(|error| {
        panic!("{label}: configured target failed to acquire a frame: {error}")
    });

    // Take the attachment before `present_after` consumes the non-Clone frame.
    // This is intentionally clear-only: no backend shader ABI is needed to
    // establish that a FrameAttachment can be a final color target.
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            depth_slice: None,
            view: ColorAttachmentView::Frame(frame.attachment()),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        },
    );
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: presentation recorder creation failed: {error}"));
    {
        let raster = recorder
            .begin_raster(&scope)
            .unwrap_or_else(|error| panic!("{label}: frame raster scope failed: {error}"));
        raster
            .end()
            .unwrap_or_else(|error| panic!("{label}: frame raster scope close failed: {error}"));
    }
    // `scope` owns a clone of the attachment descriptor.  Drop it before
    // reconfiguration, so a fixture proves actual submitted-work retirement,
    // not an accidentally retained test-local frame reference.
    drop(scope);
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: presentation recording failed: {error}"));
    let mut plan = SubmissionPlanBuilder::new(device);
    let point = plan
        .add_batch(lane, vec![work])
        .unwrap_or_else(|error| panic!("{label}: presentation batch construction failed: {error}"));
    plan.present_after(frame, point)
        .unwrap_or_else(|error| panic!("{label}: present-after planning failed: {error}"));
    let receipt = device
        .submit(plan.build().unwrap_or_else(|error| {
            panic!("{label}: presentation plan validation failed: {error}")
        }))
        .await
        .unwrap_or_else(|error| panic!("{label}: presentation submission failed: {error}"));
    require_complete(device, receipt.completion(), label).await;
    assert_eq!(
        receipt.presents().len(),
        1,
        "{label}: one presented frame must produce one present receipt"
    );
    match device
        .wait_present(receipt.presents()[0].id())
        .await
        .unwrap_or_else(|error| panic!("{label}: presentation wait failed: {error}"))
    {
        PresentState::Accepted => {}
        terminal => panic!("{label}: clear/present reached {terminal:?}, not Accepted"),
    }

    surface
        .reconfigure(&configuration)
        .await
        .unwrap_or_else(|error| {
            panic!("{label}: reconfigure after terminal frame failed: {error}")
        });
    let frame = surface
        .acquire()
        .await
        .unwrap_or_else(|error| panic!("{label}: acquire after reconfigure failed: {error}"));
    frame
        .abandon()
        .await
        .unwrap_or_else(|error| panic!("{label}: explicit frame abandonment failed: {error}"));
    let frame = surface
        .acquire()
        .await
        .unwrap_or_else(|error| panic!("{label}: abandonment left the lease outstanding: {error}"));
    frame
        .abandon()
        .await
        .unwrap_or_else(|error| panic!("{label}: final frame abandonment failed: {error}"));
}
