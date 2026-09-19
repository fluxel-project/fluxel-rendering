//! The one-shot device request Direct3D 12 hands back, and the adapter that
//! carries its device onto the seam.
//!
//! [`crate::base::platform::DeviceRequestBackend`] is a *polled* handover because
//! some backends genuinely need one: WebGPU resolves an adapter and then a device
//! over several turns, and the portable layer must be able to wait without
//! knowing which backend it is talking to. Direct3D 12 has no such path —
//! `D3D12CreateDevice` returns when the device exists — so the request is
//! [`RequestProgress::Ready`] on its first and only poll.
//!
//! That is a claim about *this* backend, not a shortcut in the contract: a
//! backend that reported `Pending` here would be inventing a delay rather than
//! reporting one, and the two lines it would take to do so are exactly what this
//! file exists to make visible.

use std::sync::Arc;

use crate::api::capability::CapabilityFacts;
use crate::api::error::RhiResult;
use crate::api::identity::ObjectId;
use crate::api::platform::{AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::resource::buffer::BufferDescriptor;
use crate::api::submission::{CompletionState, SubmissionCapabilities};
use crate::base::command::{SubmissionOutcome, SubmissionRequest};
use crate::base::platform::{DeviceBackend, DeviceRequestBackend, RequestProgress};
use crate::base::resource::BufferBackend;

use super::device::Dx12Device;

/// A device request that has already produced its device.
///
/// Creation is the single native step and it happened in
/// [`super::provider::Dx12Provider::request_device`], so there is nothing left to
/// be pending about.
pub(super) struct Dx12Request {
    /// The device this request carries. Private, with a constructor rather than
    /// a public field: the only thing that may put a device in here is the
    /// provider that just created it, and a `Dx12Request` assembled anywhere else
    /// would be one whose `poll` reports `Ready` for a device nobody created.
    native: Arc<Dx12Device>,
}

impl Dx12Request {
    /// Wraps the device creation has just produced.
    ///
    /// `pub(super)` because [`super::provider`] is the only caller. Not
    /// `pub(crate)`: a request is an intermediate between one provider verb and
    /// the portable layer's poll, and nothing else in this backend has a reason
    /// to mint one.
    pub(super) fn new(native: Arc<Dx12Device>) -> Self {
        Self { native }
    }
}

impl DeviceRequestBackend for Dx12Request {
    fn poll(&mut self) -> RhiResult<RequestProgress> {
        // The one place this backend's device pointer is handed to the portable
        // layer. `Arc` rather than a fresh box so that whatever the portable side
        // keeps reaches the same native device.
        Ok(RequestProgress::Ready(Box::new(ArcDevice(Arc::clone(
            &self.native,
        )))))
    }
}

/// Adapts a shared [`Dx12Device`] into the owned box the seam hands over.
///
/// The seam transfers an owned `Box<dyn DeviceBackend>` and the portable
/// `Device` puts it behind an `Arc`, but this backend must keep its own handle to
/// observe a loss. This wrapper keeps the two from being two different devices.
///
/// The tuple field is `pub(super)` for one reason: this chapter's test set builds
/// one directly from the `Arc<Dx12Device>` it already holds, to check that the
/// forwarding below is not where a device's behavior changes. Reaching that
/// through a real `PlatformProvider` would test the portable layer instead.
pub(super) struct ArcDevice(pub(super) Arc<Dx12Device>);

impl DeviceBackend for ArcDevice {
    fn backend_kind(&self) -> BackendKind {
        self.0.backend_kind()
    }

    fn adapter_info(&self) -> &AdapterInfo {
        self.0.adapter_info()
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.0.capability_facts()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.0.submission_capabilities()
    }

    fn object_id(&self) -> ObjectId {
        self.0.object_id()
    }

    fn status(&self) -> DeviceStatus {
        self.0.status()
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.0.loss_info()
    }

    fn poll(&self) -> RhiResult<()> {
        self.0.poll()
    }

    fn wait_idle(&self) -> RhiResult<()> {
        self.0.wait_idle()
    }

    /// Forwarded rather than reimplemented, and that is the point of this
    /// wrapper: a device that reached the portable layer as `ArcDevice` and one
    /// the tests hold as `Arc<Dx12Device>` must allocate through the same code,
    /// or the terminal-failure path above would be exercised by nobody.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        self.0.create_buffer(descriptor)
    }

    /// Forwarded for the same reason `create_buffer` is, and with the same
    /// consequence if it were not: two entry points into the shader lowering would
    /// let the portable path and the test path disagree about what a module holds.
    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::base::shader::ShaderModuleBackend>> {
        self.0.create_shader(artifact)
    }

    /// Forwarded for the same reason `create_buffer` is: a device that reached the
    /// portable layer as `ArcDevice` and one the tests hold as `Arc<Dx12Device>`
    /// must allocate their descriptor slots from the *same* heap, or two bind
    /// groups built the same way would take different paths and only one of them
    /// would be exercised.
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::base::binding::BindGroupBackend>> {
        self.0.create_bind_group(descriptor)
    }

    /// Forwarded for the same reason `create_buffer` is, and here the consequence
    /// of not forwarding would be a wrong *answer* rather than a missing code path:
    /// a pipeline built through one entry point and a pipeline built through the
    /// other must be built against the same root signature layout, and a second
    /// implementation is where the two would drift.
    fn create_compute_pipeline(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::base::pipeline::ComputePipelineBackend>> {
        self.0.create_compute_pipeline(descriptor)
    }

    /// Forwarded for the same reason `create_buffer` is: a device that reached the
    /// portable layer as `ArcDevice` and one the tests hold as `Arc<Dx12Device>`
    /// must submit through the same code, or the terminal-failure path would be
    /// exercised by nobody.
    fn submit(&self, request: &SubmissionRequest<'_>) -> RhiResult<SubmissionOutcome> {
        self.0.submit(request)
    }

    fn completion(&self, serial: u64) -> CompletionState {
        self.0.completion(serial)
    }
}
