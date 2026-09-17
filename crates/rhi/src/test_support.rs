//! Doc-hidden conformance observation helpers for workspace hardware fixtures.
//!
//! This non-default module exposes neither host mapping, command recording,
//! nor native handles. It is intentionally limited to observing completed
//! fixture work and injecting private conformance faults; production callers
//! must not use it as a general readback or submission API.
//!
//! # The two entries that open something, and why the sentence above survives them
//!
//! `observe_desktop_gl4_context` (Windows, `native-gl-wgl` builds only) is not
//! an observation of completed work: it
//! opens a desktop GL context, because the GL-family providers had no reachable
//! entry point at all and a release gate that asks for real-hardware evidence
//! could not otherwise collect any.  It is named here rather than folded in
//! silently, and what it hands back is what keeps the rule intact -- a report
//! of what the driver answered, and no context, provider, device, or native
//! handle for a caller to hold.  The context it opens is dropped before the
//! call returns, so there is nothing to misuse even by accident.
//!
//! `drive_desktop_gl4_draws` is the second, and it opens a context for a reason
//! the first cannot serve: a *cost* cannot be observed, it has to be incurred.
//! Observing and dropping cannot measure what a frame does, so this entry keeps
//! the provider stack alive across a compiled graph, an executor and a
//! submission, and hands back counters and durations.  The same rule holds for
//! the same reason -- the context, the provider and the device all die inside
//! the call, and the caller gets numbers -- and the one thing it adds is that a
//! caller may choose the execution mode, which is a string here precisely so
//! that the crate-private type behind it stays crate-private.

use crate::{
    BufferUploadError, BufferUploadStage, Device, TextureUploadError, TextureUploadStage,
    UploadedBuffer, UploadedTexture,
};

/// Opens a real desktop GL context over the caller's drawable and reports it.
///
/// The window is the caller's, named through the standard raw-handle traits;
/// this crate never creates one, and the ecosystem's host is one of the
/// producers that satisfies them.  The entry is scoped -- it opens the context,
/// observes it and drops it inside one call -- and it hands back a report rather
/// than a context, provider, device, or native handle, so no borrowed provider
/// stack ever escapes.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub use crate::webgl2::conformance::{DesktopGl4ContextReport, observe_desktop_gl4_context};

/// Drives one measured workload through the compatibility adapter over a real
/// desktop GL context, and reports what the run cost.
///
/// The window is the caller's, named through the same standard raw-handle traits
/// [`observe_desktop_gl4_context`] takes, and the context, provider and device
/// are all dropped before this returns.  What comes back is counters and
/// durations -- the numbers a candidate optimization is accepted or rejected on
/// -- plus the mode the run actually used, so that a cached-versus-uncached
/// differential can be shown to have differed rather than assumed to have.
///
/// The mode is spelled as a string (`"optimized"` or `"oracle"`) and a
/// misspelling is refused: the type behind it is crate-private because which
/// mode a renderer runs is not a choice the common contract offers, and a
/// differential whose two halves silently ran the same mode would be worse than
/// one that failed.
#[cfg(all(windows, feature = "native-gl-wgl", feature = "test-support"))]
pub use crate::webgl2::compat::{DesktopGl4DrawReport, DomainTally, drive_desktop_gl4_draws};

/// RAII latch for exactly one later successful DX12 presentation completion.
///
/// While this handle lives, polling and waiting for the consumed presentation
/// completion report `Pending`; dropping it releases that completion without
/// changing native fence state. It is intentionally unavailable to production
/// builds and never affects copy/upload completions. On unsupported platforms
/// it is a no-op so portable conformance code remains compilable.
#[doc(hidden)]
pub struct NextDx12PresentationCompletionLatch {
    #[cfg(all(windows, feature = "dx12"))]
    held: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for NextDx12PresentationCompletionLatch {
    fn drop(&mut self) {
        #[cfg(all(windows, feature = "dx12"))]
        self.held.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Holds the next successful DX12 presentation completion until the returned
/// handle is dropped.
///
/// This is a narrowly scoped frames-in-flight oracle: the hold is consumed
/// after `submit` and `present` both succeed, rather than at generic queue
/// submission, so upload/copy work cannot consume it.
#[doc(hidden)]
#[must_use]
pub fn hold_next_dx12_presentation_completion() -> NextDx12PresentationCompletionLatch {
    #[cfg(all(windows, feature = "dx12"))]
    {
        let held = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        crate::imp::arm_next_dx12_presentation_completion_hold(&held);
        NextDx12PresentationCompletionLatch { held }
    }
    #[cfg(not(all(windows, feature = "dx12")))]
    {
        NextDx12PresentationCompletionLatch {}
    }
}

/// Clears diagnostics collected after `Device::open` enabled capture.
pub fn clear_validation_diagnostics(device: &Device) {
    #[cfg(windows)]
    crate::imp::clear_validation_diagnostics(&device.inner);
    #[cfg(not(windows))]
    let _ = device;
}

/// Returns validation warnings and errors collected for `device`.
#[must_use]
pub fn validation_diagnostics(device: &Device) -> Vec<String> {
    #[cfg(windows)]
    {
        crate::imp::validation_diagnostics(&device.inner)
    }
    #[cfg(not(windows))]
    {
        let _ = device;
        Vec::new()
    }
}

/// Makes the next native submission fail before queue acceptance.
///
/// This conformance-only hook is one-shot and has no production build
/// surface. It exercises the path which must release reservations because
/// native work was never accepted.
pub fn inject_submit_rejected_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_rejected_once();
}

/// Makes the submission after exactly `successful_submits` accepted
/// submissions fail before queue acceptance.
///
/// Passing zero is identical to [`inject_submit_rejected_once`]. This is
/// conformance-only and fixtures must serialize configuration with their
/// existing guard.
pub fn inject_submit_rejected_after(successful_submits: usize) {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_rejected_after(successful_submits);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
    let _ = successful_submits;
}

/// Makes the next native submission report accepted-but-unknown failure.
///
/// This conformance-only hook is one-shot and exercises quarantine of all
/// submitted leases: callers must not infer a final resource state.
pub fn inject_submit_accepted_unknown_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_accepted_unknown_once();
}

/// Makes the submission after exactly `successful_submits` accepted
/// submissions report accepted-but-unknown failure.
///
/// Passing zero is identical to [`inject_submit_accepted_unknown_once`].
/// This is conformance-only and fixtures must serialize configuration with
/// their existing guard.
pub fn inject_submit_accepted_unknown_after(successful_submits: usize) {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_accepted_unknown_after(successful_submits);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
    let _ = successful_submits;
}

/// Makes the next nonblocking native completion observation report `Pending`.
///
/// This conformance-only hook is one-shot and is consumed by the same
/// completion-status path used by renderer polling. It never alters a
/// later blocking readback wait.
pub fn inject_completion_pending_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_completion_pending_once();
}

/// Makes the next successful DX12 present report accepted-but-unknown
/// completion after its native image has been submitted and presented.
///
/// Unlike the generic submission hook, this is consumed exclusively by the
/// presentation lowering; it exercises surface-ticket quarantine rather than
/// pretending that work was rejected before queue acceptance.
pub fn inject_dx12_presentation_accepted_unknown_once() {
    #[cfg(all(windows, feature = "dx12"))]
    crate::imp::inject_dx12_presentation_accepted_unknown_once();
}

/// Reads a finalized immutable upload using its exact published state.
///
/// The helper is only for CPU-oracle hardware fixtures. It consumes the
/// upload's `CopyDestination` state as the true incoming state before the
/// private readback transition, so it cannot repair a wrong upload state.
pub fn readback_uploaded_buffer(
    device: &Device,
    uploaded: &UploadedBuffer,
) -> Result<Vec<u8>, BufferUploadError> {
    if uploaded.buffer().device_identity() != device.identity() {
        return Err(BufferUploadError::ForeignDevice);
    }
    if !uploaded
        .buffer()
        .allowed_usage()
        .contains(fluxel_rendergraph::BufferUsageKind::CopySource)
    {
        return Err(BufferUploadError::InvalidRequest(
            crate::InvalidBufferUploadReason::CopySourceUsageRequired,
        ));
    }
    #[cfg(windows)]
    {
        crate::imp::readback_buffer_for_test(
            &device.inner,
            uploaded.buffer().native(),
            uploaded.lease().into(),
            uploaded.outgoing_state(),
            uploaded.buffer().descriptor().buffer.size,
        )
        .map_err(|reason| BufferUploadError::Native {
            backend: device.hardware().backend,
            stage: BufferUploadStage::Completion,
            reason,
        })
    }
    #[cfg(not(windows))]
    {
        let _ = uploaded;
        Err(BufferUploadError::Native {
            backend: device.hardware().backend,
            stage: BufferUploadStage::Completion,
            reason: "native readback is only supported on Windows".into(),
        })
    }
}

/// Reads a finalized immutable texture using its exact exported incoming
/// state. The returned tuple is `(tight_rgba8, padded_native_rows,
/// bytes_per_row)` for hardware CPU-oracle fixtures only.
///
/// This helper consumes [`UploadedTexture::outgoing_state`] as its actual
/// incoming state and restores that state afterward; it never assumes or
/// silently repairs an incorrect graph export.
pub fn readback_uploaded_texture(
    device: &Device,
    uploaded: &UploadedTexture,
) -> Result<(Vec<u8>, Vec<u8>, u32), TextureUploadError> {
    if uploaded.texture().device_identity() != device.identity() {
        return Err(TextureUploadError::ForeignDevice);
    }
    if !uploaded
        .texture()
        .allowed_usage()
        .contains(fluxel_rendergraph::TextureUsageKind::CopySource)
    {
        return Err(TextureUploadError::InvalidRequest(
            crate::InvalidTextureUploadReason::UnexpectedUsage,
        ));
    }
    #[cfg(windows)]
    {
        let readback = crate::imp::readback_texture_for_test(
            &device.inner,
            uploaded.texture().native(),
            uploaded.lease().into(),
            uploaded.texture().descriptor().texture,
            uploaded.outgoing_state(),
        )
        .map_err(|reason| TextureUploadError::Native {
            backend: device.hardware().backend,
            stage: TextureUploadStage::Completion,
            reason,
        })?;
        Ok((readback.tight, readback.padded, readback.bytes_per_row))
    }
    #[cfg(not(windows))]
    {
        let _ = uploaded;
        Err(TextureUploadError::Native {
            backend: device.hardware().backend,
            stage: TextureUploadStage::Completion,
            reason: "native readback is only supported on Windows".into(),
        })
    }
}
