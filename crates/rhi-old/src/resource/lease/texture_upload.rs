//! Texture immutable-upload completion state and its terminal lifetime rule.
//!
//! An incomplete accepted upload deliberately leaks its completion bundle on drop:
//! native work may still reference the staging allocation and destination lease.

use super::*;

impl PendingTextureUpload {
    /// Returns the accepted upload's current completion state without blocking.
    pub fn status(&self) -> Result<CompletionStatus, TextureUploadError> {
        crate::imp::completion_status(&self.completion.0).map_err(|reason| {
            TextureUploadError::Native {
                backend: self.backend,
                stage: TextureUploadStage::Completion,
                reason,
            }
        })
    }

    /// Waits at most `timeout` for this accepted upload to complete.
    pub fn wait(
        &self,
        timeout: core::time::Duration,
    ) -> Result<CompletionStatus, TextureUploadError> {
        crate::imp::wait_completion(&self.completion.0, timeout).map_err(|reason| {
            TextureUploadError::Native {
                backend: self.backend,
                stage: TextureUploadStage::Completion,
                reason,
            }
        })
    }

    /// Converts a proven-complete upload into its graph-importable texture.
    pub fn finalize(mut self) -> Result<UploadedTexture, IncompleteTextureUpload> {
        let status = crate::imp::completion_status(&self.completion.0).unwrap_or(
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost),
        );
        if status == CompletionStatus::Complete {
            Ok(UploadedTexture {
                texture: self
                    .texture
                    .take()
                    .expect("pending upload retains its destination until completion"),
            })
        } else {
            Err(IncompleteTextureUpload {
                upload: self,
                status,
            })
        }
    }
}

impl Drop for PendingTextureUpload {
    fn drop(&mut self) {
        // Accepted native work may still read the submission bundle, texture
        // lease, and staging allocation after the caller drops this handle.
        // Keep a completion clone deliberately undisposed until the backend
        // proves completion: releasing that ownership early would turn a
        // non-blocking cancellation path into use-after-free at the native
        // boundary. Completed uploads retain ordinary immediate cleanup.
        if !matches!(
            crate::imp::completion_status(&self.completion.0),
            Ok(CompletionStatus::Complete)
        ) {
            let _quarantined = std::mem::ManuallyDrop::new(self.completion.clone());
        }
    }
}

impl IncompleteTextureUpload {
    /// Returns the observed non-complete state.
    #[must_use]
    pub fn status(&self) -> CompletionStatus {
        self.status
    }
    /// Returns the retained operation for a later poll or wait.
    #[must_use]
    pub fn upload(&self) -> &PendingTextureUpload {
        &self.upload
    }
    /// Returns ownership of the retained upload.
    #[must_use]
    pub fn into_pending(self) -> PendingTextureUpload {
        self.upload
    }
}

impl UploadedTexture {
    /// Returns the immutable destination texture.
    #[must_use]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }
    /// Returns the state that graph imports must use as their incoming state.
    #[must_use]
    pub const fn outgoing_state(&self) -> ResourceAccessState {
        ResourceAccessState::CopyDestination
    }
    /// Acquires an independent lifetime lease for later graph submission.
    #[must_use]
    pub fn lease(&self) -> TextureLease {
        self.texture.lease()
    }
}

impl fmt::Debug for PendingTextureUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingTextureUpload")
            .field("texture", &self.texture)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for IncompleteTextureUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncompleteTextureUpload")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for UploadedTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadedTexture")
            .field("texture", &self.texture)
            .field("outgoing_state", &self.outgoing_state())
            .finish()
    }
}
