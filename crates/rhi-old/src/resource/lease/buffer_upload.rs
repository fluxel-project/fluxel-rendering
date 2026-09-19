//! Buffer immutable-upload completion state and its terminal lifetime rule.
//!
//! An incomplete accepted upload deliberately leaks its completion bundle on drop:
//! native work may still reference the staging allocation and destination lease.

use super::*;

impl PendingBufferUpload {
    /// Returns the completion state without blocking the calling thread.
    pub fn status(&self) -> Result<CompletionStatus, BufferUploadError> {
        crate::imp::completion_status(&self.completion.0).map_err(|reason| {
            BufferUploadError::Native {
                backend: self.backend,
                stage: BufferUploadStage::Completion,
                reason,
            }
        })
    }

    /// Waits no longer than `timeout` for the accepted upload submission.
    ///
    /// A timeout returns [`CompletionStatus::Pending`]; it does not cancel the
    /// upload or release its staging allocation.
    pub fn wait(
        &self,
        timeout: core::time::Duration,
    ) -> Result<CompletionStatus, BufferUploadError> {
        crate::imp::wait_completion(&self.completion.0, timeout).map_err(|reason| {
            BufferUploadError::Native {
                backend: self.backend,
                stage: BufferUploadStage::Completion,
                reason,
            }
        })
    }

    /// Finalizes this upload only after native completion is proven.
    ///
    /// Pending and failed submissions are returned intact so their completion
    /// and keepalive storage remain owned by the caller.
    pub fn finalize(mut self) -> Result<UploadedBuffer, IncompleteBufferUpload> {
        let status = crate::imp::completion_status(&self.completion.0).unwrap_or(
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost),
        );
        if status == CompletionStatus::Complete {
            Ok(UploadedBuffer {
                buffer: self
                    .buffer
                    .take()
                    .expect("pending upload retains its destination until completion"),
            })
        } else {
            Err(IncompleteBufferUpload {
                upload: self,
                status,
            })
        }
    }
}

impl Drop for PendingBufferUpload {
    fn drop(&mut self) {
        // Dropping an application-level pending operation must neither block
        // the caller nor release storage still referenced by accepted native
        // work. A leaked Arc clone keeps the submission bundle, target lease,
        // and private staging allocation quarantined whenever completion is
        // not proven. Complete bundles take the ordinary immediate cleanup
        // path when this value's original completion field is dropped.
        if !matches!(
            crate::imp::completion_status(&self.completion.0),
            Ok(CompletionStatus::Complete)
        ) {
            let _quarantined = std::mem::ManuallyDrop::new(self.completion.clone());
        }
    }
}

impl IncompleteBufferUpload {
    /// Returns the observed non-complete state.
    #[must_use]
    pub fn status(&self) -> CompletionStatus {
        self.status
    }

    /// Returns the retained upload so it can be polled or waited again.
    #[must_use]
    pub fn upload(&self) -> &PendingBufferUpload {
        &self.upload
    }

    /// Returns ownership of the retained upload.
    #[must_use]
    pub fn into_pending(self) -> PendingBufferUpload {
        self.upload
    }
}

impl UploadedBuffer {
    /// Returns the immutable destination buffer.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the state that later graph imports must use as their incoming state.
    #[must_use]
    pub const fn outgoing_state(&self) -> ResourceAccessState {
        ResourceAccessState::CopyDestination
    }

    /// Returns an independent lifetime lease for later graph submission.
    #[must_use]
    pub fn lease(&self) -> BufferLease {
        self.buffer.lease()
    }
}

impl fmt::Debug for PendingBufferUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingBufferUpload")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for IncompleteBufferUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncompleteBufferUpload")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for UploadedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadedBuffer")
            .field("buffer", &self.buffer)
            .field("outgoing_state", &self.outgoing_state())
            .finish()
    }
}
