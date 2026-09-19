//! Immutable buffer and texture upload construction.
use super::super::validation::*;
use super::super::*;
impl Device {
    /// Creates a device-local RGBA8 texture and starts one immutable whole-image upload.
    ///
    /// Input bytes are tightly packed RGBA8 rows. The private native boundary
    /// supplies required row padding; callers never provide a native pitch.
    pub fn upload_immutable_texture(
        &self,
        descriptor: TextureDescriptor,
        bytes: &[u8],
    ) -> Result<PendingTextureUpload, TextureUploadError> {
        validate_immutable_texture_upload_descriptor(descriptor, bytes)?;
        let texture = self
            .create_texture(descriptor)
            .map_err(TextureUploadError::Resource)?;
        let completion = crate::imp::upload_immutable_texture(
            &self.inner,
            texture.native(),
            texture.lease().into(),
            descriptor.texture,
            bytes,
        )
        .map_err(|(stage, reason)| TextureUploadError::Native {
            backend: self.hardware.backend,
            stage,
            reason,
        })?;
        Ok(PendingTextureUpload {
            texture: Some(texture),
            completion: NativeCompletion(completion),
            backend: self.hardware.backend,
        })
    }

    /// Creates a device-local buffer and starts one immutable copy upload.
    ///
    /// The descriptor size must exactly match `bytes`, the byte length must be
    /// non-zero and four-byte aligned, and the declared usage must include
    /// copy-destination access. The returned value does not make the buffer
    /// graph-importable until [`PendingBufferUpload::finalize`] reports a
    /// completed submission.
    pub fn upload_immutable_buffer(
        &self,
        descriptor: BufferDescriptor,
        bytes: &[u8],
    ) -> Result<PendingBufferUpload, BufferUploadError> {
        validate_immutable_upload_descriptor(descriptor, bytes)?;
        let buffer = self
            .create_buffer(descriptor)
            .map_err(BufferUploadError::Resource)?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::CopyDestination)
        {
            return Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::CopyDestinationUsageRequired,
            ));
        }
        let completion = crate::imp::upload_immutable_buffer(
            &self.inner,
            buffer.native(),
            buffer.lease().into(),
            bytes,
        )
        .map_err(|(stage, reason)| BufferUploadError::Native {
            backend: self.hardware.backend,
            stage,
            reason,
        })?;
        Ok(PendingBufferUpload {
            buffer: Some(buffer),
            completion: NativeCompletion(completion),
            backend: self.hardware.backend,
        })
    }
}
