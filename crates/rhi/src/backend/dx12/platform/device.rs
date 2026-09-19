//! The Direct3D 12 logical device: one `ID3D12Device`, its liveness, and the
//! lowering verbs the portable layer reaches it through.
//!
//! This is the DX12 half of [`crate::base::platform::DeviceBackend`]. What it
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
//! and [`DeviceBackend::status`] is how a caller reads it. The two verbs that can
//! observe a terminal `HRESULT` — allocation and submission — are the only
//! writers, and each writes only on the `Terminal` classification: marking a
//! device lost because one allocation ran out of memory would retire a usable
//! device on a transient failure.
//!
//! # Why the capability probe runs here, once, at creation
//!
//! [`super::facts::probe`] is called from [`super::provider`]'s `create_native`,
//! next to `CreateCommandQueue` and `CreateFence` and for the same reason: these
//! are the native questions a device either answers or does not, and a device
//! that cannot be asked is better refused at creation than discovered halfway
//! through a frame. A table filled lazily would put the first capability answer
//! on whichever call happened to arrive first.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::api::capability::CapabilityFacts;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::ObjectId;
use crate::api::platform::{AdapterInfo, BackendKind, DeviceLossInfo, DeviceStatus};
use crate::api::resource::buffer::BufferDescriptor;
use crate::api::submission::{CompletionState, SubmissionCapabilities};
use crate::base::platform::DeviceBackend;
use crate::base::resource::BufferBackend;

use crate::backend::dx12::command::{Dx12CommandSpine, SpineFailure};
use crate::backend::dx12::{resource, shader};

/// A device's liveness, as this backend observes it.
struct Liveness {
    status: DeviceStatus,
    loss: Option<DeviceLossInfo>,
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
    /// The queue, fence and command-list ring every submission goes through.
    ///
    /// Held by value and never cloned: it owns the one queue this device has, and
    /// a second handle to the same queue would be a second path to `Signal` on
    /// the same fence, which is the only place a serial is minted.
    spine: Dx12CommandSpine,
    liveness: Mutex<Liveness>,
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
        spine: Dx12CommandSpine,
        facts: CapabilityFacts,
        submission: SubmissionCapabilities,
    ) -> Self {
        Self {
            adapter,
            object: ObjectId::next(),
            device,
            spine,
            facts,
            submission,
            liveness: Mutex::new(Liveness {
                status: DeviceStatus::Active,
                loss: None,
            }),
        }
    }

    /// Borrows the liveness cell, surviving a poisoned lock.
    ///
    /// Recovering rather than propagating is correct here and only here: the
    /// guarded value is two plain fields with no invariant a panicking holder
    /// could have left half-written, so letting one panic hide the first is
    /// strictly worse than reading the fields.
    fn liveness(&self) -> std::sync::MutexGuard<'_, Liveness> {
        self.liveness
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Records that this device is gone, with the reason.
    ///
    /// One-way, like the loss it records: section 6.5 makes device loss terminal
    /// for the whole identity, so there is no matching `mark_active`.
    ///
    /// Crate-private and reached from the failure path that observes a terminal
    /// `HRESULT`. That path is now written for allocation — see
    /// [`Self::create_buffer`] — and the command lowering will be the next
    /// caller; until it lands, allocation failures are the only ones that can
    /// end a device's identity here.
    pub(crate) fn mark_lost(&self, info: DeviceLossInfo) {
        let mut liveness = self.liveness();
        liveness.status = DeviceStatus::Lost;
        liveness.loss = Some(info);
    }
}

impl DeviceBackend for Dx12Device {
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
        self.liveness().status
    }

    fn loss_info(&self) -> Option<DeviceLossInfo> {
        self.liveness().loss.clone()
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
        self.spine.advance();
        Ok(())
    }

    fn wait_idle(&self) -> RhiResult<()> {
        // Section 6.7: shutdown, recovery, and diagnostics only. The wait is a
        // bounded one on a fence-signalled event rather than an `INFINITE` block,
        // because a removed device leaves fence values that will never be written
        // and a library must not turn that into a hung host.
        self.spine.wait_idle().map_err(SpineFailure::into_rhi)
    }

    /// Allocates one buffer, and is the only place in this backend that acts on a
    /// terminal native failure.
    ///
    /// The `HRESULT` alone cannot answer the question that matters here: a
    /// removed device reports that from an arbitrary call, and this one is as
    /// likely as any other, so a failure that looks like a plain allocation
    /// refusal may be the device ending. [`crate::backend::dx12::ffi::NativeError`] carries the
    /// classification beside the error so this reads it once rather than
    /// re-deriving it, and `Terminal` is the only case that is recorded: marking
    /// a device lost because one allocation ran out of memory would retire a
    /// usable device on a transient failure, which is the expensive direction of
    /// the mistake and the same reasoning [`crate::backend::dx12::ffi::NativeFailure`] gives for
    /// treating a hung device as alive.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> RhiResult<Box<dyn BufferBackend>> {
        match resource::create_buffer(&self.device, descriptor) {
            Ok(buffer) => Ok(Box::new(buffer) as Box<dyn BufferBackend>),
            Err(native) => {
                if native.failure().is_terminal() {
                    // Built before the error is consumed, because the summary is
                    // about the same failure the error reports and the port read
                    // that follows must answer the same thing.
                    let summary = format!(
                        "Direct3D 12 reported a terminal failure while allocating a buffer, \
                         and section 6.5 makes loss terminal for the identity: {}",
                        native.as_error()
                    );
                    self.mark_lost(DeviceLossInfo::new(summary));
                }
                Err(native.into_rhi())
            }
        }
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
    fn create_shader(
        &self,
        artifact: &crate::api::shader::ShaderArtifact,
    ) -> RhiResult<Box<dyn crate::base::shader::ShaderModuleBackend>> {
        Ok(Box::new(shader::create_shader(artifact)))
    }

    /// Refuses, because this backend's descriptor lowering is not written.
    ///
    /// # Why this is a refusal and not a stop
    ///
    /// The portable contract is complete for this verb and Direct3D 12 can express
    /// it, so the honest answer is that the *lowering* is missing — which is
    /// [`RhiErrorKind::Unsupported`] and not a panic (discipline 3 in
    /// `crate::base`). A caller that gets this learns it must not build the group;
    /// a caller that got a fabricated object would learn it only at the first
    /// dispatch that read a descriptor nothing had written.
    ///
    /// # What has to be built, and what it owns
    ///
    /// A shader-visible CBV/SRV/UAV descriptor heap with slot allocation and
    /// release, a sampler heap, and one
    /// [`crate::base::binding::BindGroupBackend`] per packet that writes each
    /// entry's view into its allocated slots at `space = group index`,
    /// `register = slot id`, `class = BindingKind` — read-write storage buffers and
    /// storage textures as UAVs, read-only ones and sampled textures as SRVs,
    /// uniform buffers as CBVs, samplers on the other heap. The object returned
    /// must hold an `Arc` to every resource it wrote an address for, because a
    /// native descriptor holds a *pointer* and section 22.2 makes the bind group
    /// the owner of everything it binds.
    ///
    /// Nothing about the portable half is in question here: the layout match, the
    /// range rules and the device's four binding limits have all run in
    /// `Device::create_bind_group` before this is reached.
    fn create_bind_group(
        &self,
        _descriptor: &crate::api::binding::BindGroupDescriptor,
    ) -> RhiResult<Box<dyn crate::base::binding::BindGroupBackend>> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this backend cannot lower a bind group yet: Direct3D 12 expresses it as a range of \
             a shader-visible descriptor heap, and the heap, its slot allocation and the \
             per-entry view writes are not built",
        )
        .at("Dx12Device::create_bind_group"))
    }

    /// Refuses, because this backend's root-signature and pipeline-state lowering
    /// is not written.
    ///
    /// The refusal is [`RhiErrorKind::Unsupported`] rather than a native-failure
    /// kind, and the distinction is the honest one: nothing was handed to the
    /// driver, so there is no driver verdict to report. [`Self::create_buffer`]'s
    /// note records that a failure here can also mean the device ended, and this
    /// method never gets far enough to find out.
    ///
    /// # What has to be built
    ///
    /// Direct3D 12 has no bind-group-layout object and no pipeline-interface
    /// object, so both portable verbs before this one do their native work here, at
    /// once: the whole ordered group sequence is lowered into one root signature
    /// (`space = group index`, `register = slot id`, one descriptor table per
    /// group), and the compute state object binds that root signature to the
    /// entry point's DXIL through `D3D12_SHADER_BYTECODE`.
    ///
    /// The driver's verdict arrives from `CreateComputePipelineState`, and that is
    /// why this method's failure is the first in the crate that can be about the
    /// *program* rather than about a descriptor — see the seam's documentation. A
    /// native failure must be reported as it arrives, with its own kind, and never
    /// folded into `InvalidUsage`.
    fn create_compute_pipeline(
        &self,
        _descriptor: &crate::api::pipeline::ComputePipelineDescriptor,
    ) -> RhiResult<Box<dyn crate::base::pipeline::ComputePipelineBackend>> {
        Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "this backend cannot lower a compute pipeline yet: Direct3D 12 builds one from a \
             root signature over the whole ordered group sequence plus the entry point's DXIL, \
             and neither the root-signature builder nor the pipeline-state builder is written",
        )
        .at("Dx12Device::create_compute_pipeline"))
    }

    /// Lowers a plan onto the spine's queue, and is the second place in this
    /// backend that acts on a terminal native failure.
    ///
    /// The two directions of section 41.3 meet here. Phase A — everything
    /// recorded, nothing committed — is [`Dx12CommandSpine::submit`]'s, and its
    /// `Err` genuinely proves no native work was accepted. Phase B — once
    /// anything is accepted, no `Err` may claim otherwise — is also the spine's,
    /// which is why a post-commit `Signal` failure comes back as `Ok` and is
    /// reported through [`Self::completion`] instead.
    ///
    /// What is left for this layer is the one question only the device can
    /// answer: whether a failure ended it. `Terminal` marks the device lost, for
    /// the same reason and in the same shape as [`Self::create_buffer`] — a
    /// failure that looks like a plain refusal may be the device ending, and
    /// `ffi::NativeFailure` already carries the classification so this reads it
    /// once rather than re-deriving it from a message.
    fn submit(
        &self,
        request: &crate::base::command::SubmissionRequest<'_>,
    ) -> RhiResult<crate::base::command::SubmissionOutcome> {
        match self.spine.submit(request) {
            Ok(outcome) => Ok(outcome),
            Err(failure) => {
                if failure.is_terminal() {
                    // Built before the failure is consumed, because the summary is
                    // about the same failure the error reports and the port read
                    // that follows must answer the same thing.
                    let summary = format!(
                        "Direct3D 12 reported a terminal failure while lowering a plan, and \
                         section 6.5 makes loss terminal for the identity: {}",
                        failure.message()
                    );
                    self.mark_lost(DeviceLossInfo::new(summary));
                }
                Err(failure.into_rhi())
            }
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
}
