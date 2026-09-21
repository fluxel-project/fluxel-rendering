//! The Direct3D 12 logical device: one `ID3D12Device`, its liveness, and the
//! lowering verbs the portable layer reaches it through.
//!
//! This is the DX12 half of [`crate::api::platform::backend::DeviceBackend`]. What it
//! owns is the native device and the answers that belong to *that* device — its
//! adapter snapshot, its object id, whether it is still alive, and its capability
//! table. What it does not own is anything a sibling chapter already owns:
//! adapter choice is [`super::provider`]'s, the classification of a native
//! failure is [`crate::backend::dx12::ffi`]'s, allocation is
//! [`crate::backend::dx12::resource`]'s, recording and submission is
//! [`crate::backend::dx12::command`]'s, and each of those is called from here
//! rather than reimplemented.
//!
//! # Device loss, as Direct3D 12 reports it
//!
//! D3D12 has no device-loss callback: `DXGI_ERROR_DEVICE_REMOVED` arrives as the
//! return code of the next call that touches the device. There is therefore
//! nothing for [`DeviceBackend::poll`] to poll *for liveness* — the return code of
//! a real call is the only signal there is — and this chapter never caches a
//! liveness flag derived from anything else.
//!
//! What a device does carry is a one-way cell recording that loss once it has
//! been observed, because section 6.5 makes loss terminal for the whole identity
//! and [`DeviceBackend::status`] is how a caller reads it. Every native boundary
//! shares that authority: allocation, pipeline creation, submission, fence
//! progress, readback mapping, and presentation all publish the same stable first
//! reason and terminate pending asynchronous state.
//!
//! # Why the capability probe runs here, once, at creation
//!
//! [`super::facts::probe`] is called from [`super::provider`]'s `create_native`,
//! next to `CreateCommandQueue` and `CreateFence` and for the same reason: these
//! are the native questions a device either answers or does not, and a device
//! that cannot be asked is better refused at creation than discovered halfway
//! through a frame. A table filled lazily would put the first capability answer
//! on whichever call happened to arrive first.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::backend::DeviceBackend;
use crate::api::platform::{AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::resource::backend::BufferBackend;
use crate::api::resource::buffer::BufferDescriptor;
use crate::api::submission::{CompletionState, SubmissionCapabilities};

use crate::backend::dx12::binding::DescriptorHeap;
use crate::backend::dx12::command::Dx12CommandSpine;
use crate::backend::dx12::presentation::Dx12Presentation;
use crate::backend::dx12::{binding, pipeline, resource, shader};

/// A device's liveness, as this backend observes it.
struct Liveness {
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
}

/// The sole loss authority shared by a DX12 device and presentation leases.
pub(crate) struct Dx12LossState {
    liveness: Mutex<Liveness>,
    /// Backend-private subscribers which must terminate native asynchronous
    /// state on the *first* loss regardless of the native call that observed it
    /// (fence, allocation, pipeline creation, or presentation).
    handlers: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
}

impl Dx12LossState {
    pub(crate) fn new() -> Self {
        Self {
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                loss: None,
            }),
            handlers: Mutex::new(Vec::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn status(&self) -> DeviceStatus {
        self.lock().status
    }

    pub(crate) fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.lock().loss.clone()
    }

    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let mut liveness = self.lock();
        if !matches!(liveness.status, DeviceStatus::Active) {
            return;
        }
        liveness.status = DeviceStatus::Lost;
        liveness.loss = Some(info);
        drop(liveness);
        let handlers = self
            .handlers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for handler in handlers {
            handler();
        }
    }

    /// Registers a one-way loss cleanup. If loss won the race with setup, invoke
    /// it immediately so no pending native work can escape termination.
    pub(crate) fn register_handler(&self, handler: Arc<dyn Fn() + Send + Sync>) {
        // Registration can race `mark_lost` between publication and its status
        // check. Wrap the caller's cleanup so both paths may attempt delivery
        // without ever terminating the same ticket/waker registry twice.
        let invoked = Arc::new(AtomicBool::new(false));
        let invoked_once = Arc::clone(&invoked);
        let once: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            if !invoked_once.swap(true, Ordering::AcqRel) {
                handler();
            }
        });
        self.handlers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Arc::clone(&once));
        // Do not retain the handler lock while reading liveness: `mark_lost`
        // writes liveness before snapshotting handlers, and this order prevents
        // a presentation-loss/setup race from becoming a lock inversion.
        // Checking after publication closes the setup race: loss either sees
        // this handler in its snapshot, or it has already become visible here.
        if matches!(self.status(), DeviceStatus::Lost) {
            once();
        }
    }
}

/// The native device behind a portable [`crate::api::platform::Device`].
pub(crate) struct Dx12Device {
    /// The adapter snapshot, taken from the adapter that was actually selected.
    adapter: AdapterInfo,
    /// This device's process-local id.
    object: ObjectId,
    /// The Direct3D 12 device.
    ///
    /// Held rather than dropped because every chapter of this backend lowers
    /// through it, and because dropping it is what releases the adapter. It was
    /// named `_device` for as long as nothing read it — the underscore was the
    /// honest name for an allocation that had to outlive the call that made it,
    /// not a way to quiet a lint — and the name lost its underscore in the round
    /// that gave it its first reader, [`Self::create_buffer`]. An
    /// `#[expect(dead_code)]` was tried in that earlier state and is wrong in a
    /// way the gate catches: rustc does not report an unread field while nothing
    /// constructs the struct at all, so the expectation sits unfulfilled in a
    /// non-test build while being fulfilled in a test one, and no single
    /// attribute satisfies both.
    device: ID3D12Device,
    /// The single shader-visible CBV/SRV/UAV heap shared by all bind groups.
    descriptor_heap: Arc<DescriptorHeap>,
    /// The single shader-visible sampler heap shared by all bind groups.
    sampler_heap: Arc<DescriptorHeap>,
    /// The queue, fence and command-list ring every submission goes through.
    ///
    /// Held by value and never cloned: it owns the one queue this device has, and
    /// a second handle to the same queue would be a second path to `Signal` on
    /// the same fence, which is the only place a serial is minted.
    spine: Dx12CommandSpine,
    presentation: Dx12Presentation,
    loss: Arc<Dx12LossState>,
    /// The contract this device reports.
    ///
    /// Filled by [`super::facts::probe`] from the live `ID3D12Device` beside it,
    /// which is where the per-table detail lives — read that module's doc for
    /// what each table is derived from and which questions this backend still
    /// cannot answer.
    ///
    /// Two things belong here rather than there, because they are about the shape
    /// of this struct and not about Direct3D 12. The first is the comment above:
    /// *when* the probe runs. The second is what an absent entry still means, so
    /// it is not discovered later. The table is a snapshot of *this device*, and
    /// where it is incomplete the incompleteness is `CapabilityFacts`'s own
    /// documented behaviour rather than something this backend invents: `route`,
    /// `texture_support`, `view_compatibility` and `binding_limit` answer
    /// conservatively when they have no entry, which costs throughput and not
    /// correctness. `buffer_support` is the exception that panics, and the probe
    /// fills its key space completely for exactly that reason. `binding_support`
    /// is now in the same position despite answering rather than panicking,
    /// because a binding answer of `Unsupported` refuses a legal layout rather
    /// than merely declining to advertise one — so its key space, too, is filled
    /// rather than sampled.
    facts: CapabilityFacts,
    /// The lanes this device offers.
    submission: SubmissionCapabilities,
}

impl Dx12Device {
    #[cfg(test)]
    pub(crate) fn register_test_presentation_target(
        &self,
        hwnd: windows::Win32::Foundation::HWND,
    ) -> crate::api::presentation::PresentationTarget {
        self.presentation.register_test_hwnd(hwnd)
    }
    /// Assembles the device [`super::provider`] has just created natively.
    ///
    /// Every part is made by the caller, because every part is made *there*: the
    /// adapter snapshot from the adapter that was selected, the spine from the
    /// device handle, and the table from the probe that read it. What this
    /// constructor adds is the two things that are neither — the object id, which
    /// belongs to the object being built rather than to anything the caller
    /// measured, and the liveness cell, which starts `Active` because nothing has
    /// yet observed otherwise and is private to this module.
    ///
    /// `pub(super)` rather than crate-wide: the only caller is the provider, and
    /// a device assembled from anywhere else would be one whose object id and
    /// adapter snapshot were not minted against a real DXGI adapter.
    pub(super) fn new(
        adapter: AdapterInfo,
        device: ID3D12Device,
        descriptor_heap: Arc<DescriptorHeap>,
        sampler_heap: Arc<DescriptorHeap>,
        spine: Dx12CommandSpine,
        presentation: Dx12Presentation,
        loss: Arc<Dx12LossState>,
        facts: CapabilityFacts,
        submission: SubmissionCapabilities,
    ) -> Self {
        Self {
            adapter,
            object: ObjectId::next(),
            device,
            descriptor_heap,
            sampler_heap,
            spine,
            presentation,
            loss,
            facts,
            submission,
        }
    }

    /// Records that this device is gone, with the reason.
    ///
    /// One-way, like the loss it records: section 6.5 makes device loss terminal
    /// for the whole identity, so there is no matching `mark_active`.
    ///
    /// Crate-private and reached from every backend path that observes a terminal
    /// `HRESULT`, including presentation leases that do not hold `Dx12Device`.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        self.loss.mark_lost(info);
    }

    /// The sole device-layer authority for terminal native failures. Every
    /// native-call boundary must pass its failure here before exposing it: DX12
    /// reports removal from whichever call happens to notice first, while v13
    /// requires `status`, `loss_info`, pending futures and later operations to
    /// agree on one terminal identity.
    fn observe_failure(
        &self,
        failure: crate::backend::dx12::failure::Dx12Failure,
        operation: &'static str,
    ) -> RhiError {
        if failure.is_terminal() {
            self.mark_lost(DeviceLossInfo::new(format!(
                "Direct3D 12 reported a terminal failure in {operation}: {}",
                failure.message()
            )));
        }
        if let Some(info) = self.loss_info() {
            RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned()).at(operation)
        } else {
            failure.into_rhi(operation)
        }
    }

    fn observe_native_failure(
        &self,
        failure: crate::backend::dx12::ffi::NativeError,
        operation: &'static str,
    ) -> RhiError {
        if failure.failure().is_terminal() {
            self.mark_lost(DeviceLossInfo::new(format!(
                "Direct3D 12 reported a terminal failure in {operation}: {}",
                failure.as_error()
            )));
        }
        if let Some(info) = self.loss_info() {
            RhiError::new(RhiErrorKind::DeviceLost, info.message().to_owned()).at(operation)
        } else {
            failure.into_rhi()
        }
    }
}

impl DeviceBackend for Dx12Device {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Dx12
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter
    }

    fn capability_facts(&self) -> CapabilityFacts {
        self.facts.clone()
    }

    fn submission_capabilities(&self) -> SubmissionCapabilities {
        self.submission.clone()
    }

    fn object_id(&self) -> ObjectId {
        self.object
    }

    fn status(&self) -> DeviceStatus {
        self.loss.status()
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.loss.loss_info()
    }

    fn poll(&self) -> RhiResult<()> {
        // Direct3D 12 reports a removed device from the next call that touches it
        // rather than through a callback, so there is no *liveness* state for a
        // poll to advance: the return code of a real call is the only signal
        // there is, and nothing caches a liveness flag (see the module note
        // above).
        //
        // What a poll does advance is the submission side. A readback's bytes
        // become readable when the GPU finishes writing them, and the fence is
        // where that becomes known; this is the portable layer's only progress
        // verb (section 6.7 keeps a blocking wait out of the frame loop), so it is
        // the only place those bytes can be published. `advance` reads the fence
        // and copies out whatever it reports finished — it never waits.
        self.spine
            .advance()
            .map_err(|failure| self.observe_failure(failure, "Device::poll"))
    }

    fn wait_idle(&self) -> RhiResult<()> {
        // Section 6.7: shutdown, recovery, and diagnostics only. The wait is a
        // bounded one on a fence-signalled event rather than an `INFINITE` block,
        // because a removed device leaves fence values that will never be written
        // and a library must not turn that into a hung host.
        self.spine
            .wait_idle()
            .map_err(|failure| self.observe_failure(failure, "Device::wait_idle"))
    }

    /// Allocates one buffer and routes any native failure through the device-wide
    /// loss authority.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        resource::create_buffer(&self.device, descriptor)
            .map(|buffer| Box::new(buffer) as Box<dyn BufferBackend>)
            .map_err(|failure| self.observe_native_failure(failure, "Dx12Device::create_buffer"))
    }

    fn map_buffer(
        &self,
        buffer: &crate::api::resource::Buffer,
        mode: crate::api::resource::MapMode,
        range: crate::api::resource::BufferRange,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
        let native = buffer
            .native()
            .as_any()
            .downcast_ref::<resource::Dx12Buffer>()
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "DX12 received a buffer without a DX12 allocation",
                )
                .at("Dx12Device::map_buffer")
            })?;
        self.spine.map_buffer(native, mode, range)
    }

    fn create_query_set(
        &self,
        descriptor: &crate::api::query::QuerySetDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::QuerySetBackend>> {
        resource::create_query_set(&self.device, descriptor)
            .map(|query| Box::new(query) as Box<dyn crate::api::resource::backend::QuerySetBackend>)
            .map_err(|failure| self.observe_native_failure(failure, "Dx12Device::create_query_set"))
    }

    fn create_texture(
        &self,
        descriptor: &crate::api::resource::texture::TextureDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureBackend>> {
        resource::create_texture(&self.device, descriptor)
            .map(|texture| {
                Box::new(texture) as Box<dyn crate::api::resource::backend::TextureBackend>
            })
            .map_err(|failure| self.observe_native_failure(failure, "Dx12Device::create_texture"))
    }

    fn create_texture_view(
        &self,
        texture: &crate::api::resource::Texture,
        descriptor: &crate::api::resource::view::TextureViewDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::TextureViewBackend>> {
        let native = texture
            .native()
            .as_any()
            .downcast_ref::<resource::Dx12Texture>()
            .ok_or_else(|| {
                crate::api::RhiError::new(
                    crate::api::RhiErrorKind::BackendFailure,
                    "DX12 received a texture without a DX12 native allocation",
                )
                .at("Dx12Device::create_texture_view")
            })?;
        resource::create_texture_view(&self.device, native, texture.descriptor(), descriptor)
            .map(|view| {
                Box::new(view) as Box<dyn crate::api::resource::backend::TextureViewBackend>
            })
            .map_err(|failure| {
                self.observe_native_failure(failure, "Dx12Device::create_texture_view")
            })
    }

    fn create_sampler(
        &self,
        descriptor: &crate::api::resource::sampler::SamplerDescriptor,
    ) -> RhiResult<Box<dyn crate::api::resource::backend::SamplerBackend>> {
        resource::create_sampler(&self.device, descriptor)
            .map(|sampler| {
                Box::new(sampler) as Box<dyn crate::api::resource::backend::SamplerBackend>
            })
            .map_err(|failure| self.observe_native_failure(failure, "Dx12Device::create_sampler"))
    }

    /// Prepares one shader entry point, and cannot fail.
    ///
    /// The least eventful method on this trait, and the one most worth a note,
    /// because the reason it cannot fail is a property of Direct3D 12 rather than a
    /// gap: there is no shader-module object to create and no
    /// `CheckFeatureSupport` question that could refuse. The lowering keeps the
    /// artifact's bytes alive for `D3D12_SHADER_BYTECODE` and the driver's verdict
    /// arrives at pipeline creation. [`shader`] states this at length, and
    /// the length is deliberate — "the module was created" reads like "the shader
    /// compiled", and that misreading is the one this method must not invite.
    ///
    /// There is also no device-liveness check to add here, unlike
    /// [`Self::create_buffer`]: nothing in this method touches the device. The
    /// portable layer's `require_active` has already refused a lost device, and a
    /// backend that re-checked would be discipline 2's duplicate opinion.
    fn create_shader_request(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::shader::backend::ShaderModuleBackend,
                >,
        >,
    > {
        Ok(crate::api::platform::backend::ready_creation_request(
            Box::new(shader::create_shader(artifact)),
        ))
    }

    /// Writes a validated group into the shared view and sampler descriptor heaps.
    fn create_bind_group(
        &self,
        descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::api::binding::backend::BindGroupBackend>> {
        binding::create_bind_group(
            &self.device,
            &self.descriptor_heap,
            &self.sampler_heap,
            descriptor,
        )
        .map(|group| Box::new(group) as Box<dyn crate::api::binding::backend::BindGroupBackend>)
        .map_err(|failure| self.observe_failure(failure, "Dx12Device::create_bind_group"))
    }

    /// Builds the root signature and compute PSO, observing terminal driver
    /// failures through the same device loss authority as every other native
    /// creation path.
    fn create_compute_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::ComputePipelineBackend,
                >,
        >,
    > {
        pipeline::create_compute_pipeline(&self.device, descriptor)
            .map(|pipeline| {
                Box::new(pipeline) as Box<dyn crate::api::pipeline::backend::ComputePipelineBackend>
            })
            .map_err(|failure| self.observe_failure(failure, "Dx12Device::create_compute_pipeline"))
            .map(crate::api::platform::backend::ready_creation_request)
    }

    fn create_raster_pipeline_request(
        &self,
        descriptor: &crate::api::pipeline::RasterPipelineDescriptor,
    ) -> RhiResult<
        Box<
            dyn crate::api::platform::backend::CreationRequestBackend<
                    dyn crate::api::pipeline::backend::RasterPipelineBackend,
                >,
        >,
    > {
        pipeline::create_raster_pipeline(&self.device, descriptor)
            .map(|pipeline| {
                Box::new(pipeline) as Box<dyn crate::api::pipeline::backend::RasterPipelineBackend>
            })
            .map_err(|failure| self.observe_failure(failure, "Dx12Device::create_raster_pipeline"))
            .map(crate::api::platform::backend::ready_creation_request)
    }

    fn create_pipeline_cache(
        &self,
        descriptor: &crate::api::pipeline::PipelineCacheDescriptor,
    ) -> RhiResult<(
        Box<dyn crate::api::pipeline::backend::PipelineCacheBackend>,
        crate::api::pipeline::PipelineCacheValidationKey,
    )> {
        pipeline::create_pipeline_cache(&self.device, Arc::clone(&self.loss), descriptor)
    }

    fn presentation(&self) -> Option<&dyn crate::api::presentation::backend::PresentationBackend> {
        Some(&self.presentation)
    }

    /// Lowers a plan onto the spine's queue.
    ///
    /// The two directions of section 41.3 meet here. Phase A — everything
    /// recorded, nothing committed — is [`Dx12CommandSpine::submit`]'s, and its
    /// `Err` genuinely proves no native work was accepted. Phase B — once
    /// anything is accepted, no `Err` may claim otherwise — is also the spine's,
    /// which is why a post-commit `Signal` failure comes back as `Ok` and is
    /// reported through [`Self::completion`] instead.
    ///
    /// Phase-A failures flow through the shared loss observer. A terminal Phase-B
    /// `Signal` failure is recorded inside the spine after native acceptance and
    /// therefore still returns `Ok`, while status/completion become terminal.
    fn submit(
        &self,
        request: &crate::api::submission::backend::SubmissionRequest<'_>,
    ) -> RhiResult<crate::api::submission::backend::SubmissionOutcome> {
        match self.spine.submit(request) {
            Ok(outcome) => Ok(outcome),
            Err(failure) => Err(self.observe_failure(failure, "Dx12Device::submit")),
        }
    }

    /// Reports one serial's state, asking the spine first.
    ///
    /// The order is section 41.8's, and it is the whole reason this is not a
    /// two-line delegate. Points already reported `Complete` must stay `Complete`
    /// after a loss — the work really did finish, and a caller that saw it finish
    /// must not be told it did not. So the spine's answer wins when it is
    /// `Complete`, and only a spine answer of `Pending` or `Failed` can be
    /// upgraded to `DeviceLost` by the liveness cell.
    ///
    /// The upgrade is what closes the loop the spine opens: a serial past the
    /// first unobservable one is answered `Failed` by [`Dx12CommandSpine`], which
    /// the portable layer would surface as a backend failure even though the
    /// device is gone. With the liveness cell consulted last, the same serial
    /// answers `DeviceLost` — the terminal state section 41.8 requires and the one
    /// a caller branches on to recover.
    fn completion(&self, serial: u64) -> CompletionState {
        let spine = self.spine.completion(serial);
        if matches!(spine, CompletionState::Complete) {
            return spine;
        }
        match self.loss_info() {
            Some(info) => CompletionState::DeviceLost(info),
            None => spine,
        }
    }

    fn completion_or_register_waker(
        &self,
        serial: u64,
        waker: &std::task::Waker,
    ) -> CompletionState {
        let spine = self.spine.completion_or_register_waker(serial, waker);
        if matches!(spine, CompletionState::Complete) {
            return spine;
        }
        match self.loss_info() {
            Some(info) => CompletionState::DeviceLost(info),
            None => spine,
        }
    }
}

#[cfg(test)]
mod loss_tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[test]
    fn loss_authority_notifies_each_handler_once_and_preserves_first_reason() {
        let loss = Dx12LossState::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let observer_calls = Arc::clone(&calls);
        loss.register_handler(Arc::new(move || {
            observer_calls.fetch_add(1, Ordering::SeqCst);
        }));

        loss.mark_lost(DeviceLossInfo::new("first loss".to_owned()));
        loss.mark_lost(DeviceLossInfo::new("later loss".to_owned()));

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(matches!(loss.status(), DeviceStatus::Lost));
        assert_eq!(loss.loss_info().unwrap().message(), "first loss");
    }

    #[test]
    fn handler_registered_after_loss_is_terminated_immediately() {
        let loss = Dx12LossState::new();
        loss.mark_lost(DeviceLossInfo::new("lost".to_owned()));
        let calls = Arc::new(AtomicUsize::new(0));
        let observer_calls = Arc::clone(&calls);
        loss.register_handler(Arc::new(move || {
            observer_calls.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
