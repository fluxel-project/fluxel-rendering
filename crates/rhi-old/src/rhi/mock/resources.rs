//! The mock's buffers, textures, views, samplers, and upload jobs.
//!
//! Each type here is the backend half of one portable façade and does one
//! thing: keep the descriptor it was handed and answer with it. Descriptor
//! normalization, capability checks, and cross-device checks all happen in the
//! `Device` façade before a backend is called, so repeating them here would
//! create a second place that decides legality.

use crate::rhi::platform::{DeviceIdentity, ObjectId, RhiResult};
use crate::rhi::submission::CompletionPoint;
use crate::rhi::resource::{
    BufferBackend, BufferDescriptor, ReadbackData, ReadbackRequest, ReadbackStatus,
    ReadbackTicketBackend, SamplerBackend, SamplerDescriptor, Texture, TextureBackend,
    TextureDescriptor, TextureViewBackend, TextureViewDescriptor, UploadDescriptor,
    UploadJobBackend,
};

/// The backend half of the mock's buffers.
pub(super) struct MockBuffer {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: BufferDescriptor,
}

impl BufferBackend for MockBuffer {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's textures.
pub(super) struct MockTexture {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: TextureDescriptor,
}

impl TextureBackend for MockTexture {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's texture views.
///
/// It holds the texture façade rather than a raw identity because the view
/// contract is expressed in terms of the texture it views.
pub(super) struct MockTextureView {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) texture: Texture,
    pub(super) descriptor: TextureViewDescriptor,
}

impl TextureViewBackend for MockTextureView {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn texture(&self) -> &Texture {
        &self.texture
    }

    fn descriptor(&self) -> &TextureViewDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's samplers.
pub(super) struct MockSampler {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: SamplerDescriptor,
}

impl SamplerBackend for MockSampler {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &SamplerDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's upload jobs.
pub(super) struct MockUploadJob {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) descriptor: UploadDescriptor,
}

impl UploadJobBackend for MockUploadJob {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn descriptor(&self) -> &UploadDescriptor {
        &self.descriptor
    }
}

/// The backend half of the mock's readback tickets.
///
/// The mock has no memory to read back, so a ticket is always
/// [`ReadbackStatus::Pending`] and never yields data. Reporting bytes the
/// backend never produced would let a test assert against fabricated values,
/// which is the one thing a contract fixture must not do.
pub(super) struct MockReadbackTicket {
    pub(super) id: ObjectId,
    pub(super) device: DeviceIdentity,
    pub(super) request: ReadbackRequest,
}

impl ReadbackTicketBackend for MockReadbackTicket {
    fn id(&self) -> ObjectId {
        self.id
    }

    fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    fn request(&self) -> &ReadbackRequest {
        &self.request
    }

    fn status(&self) -> ReadbackStatus {
        ReadbackStatus::Pending
    }

    fn completion(&self) -> Option<CompletionPoint> {
        None
    }

    fn try_read(&self) -> RhiResult<Option<ReadbackData<'_>>> {
        Ok(None)
    }
}
