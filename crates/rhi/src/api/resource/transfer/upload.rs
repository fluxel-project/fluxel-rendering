//! Section 17: upload jobs — caller bytes into a resource.

use super::validate_texture_region;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::resource::buffer::{
    Buffer, BufferRange, BufferUsage, validate_buffer_ownership, validate_buffer_range,
};
use crate::api::resource::route::BufferCopyLayoutLimits;
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureSubresourceLayers, source_bytes_required,
    validate_host_texel_layout,
};
use crate::api::resource::texture::{
    Extent3d, Texture, TextureDescriptor, TextureDimension, TextureUsage,
    validate_texture_ownership,
};
use std::sync::Arc;

/// A buffer upload: caller bytes written into a byte range of a buffer.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BufferUploadDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The buffer to write into.
    pub dst: Buffer,
    /// The byte offset to start writing at.
    pub dst_offset: u64,

    /// Retained immutable source bytes。
    pub bytes: Arc<[u8]>,
}
/// A texture upload: caller bytes written into a texture region.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureUploadDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The texture to write into.
    pub dst: Texture,

    /// Which mip level and array layers are written.
    pub subresource: TextureSubresourceLayers,
    /// Where in the level the region starts.
    pub origin: Origin3d,
    /// Size of the region in texels.
    pub extent: Extent3d,

    /// CPU source layout, not native GPU copy layout.
    pub source_layout: HostTexelLayout,

    /// Retained immutable source bytes。
    pub bytes: Arc<[u8]>,
}
/// Either upload, as one value.
///
/// Exists so that a job, a capture record, and a statistics entry can carry
/// "the mutation" without caring which kind it was. The two variants are not
/// unified further: a buffer range and a texture region share no fields, and a
/// single struct with six optional members would make "a texture upload with a
/// `dst_offset`" expressible.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum UploadDescriptor {
    /// A buffer upload.
    Buffer(BufferUploadDescriptor),
    /// A texture upload.
    Texture(TextureUploadDescriptor),
}
/// A prepared, repeatable resource mutation.
///
/// Section 17.2 keeps the source payload: an upload job owns its retained bytes
/// and is used by `&UploadJob`, so the same job may be encoded repeatedly, and
/// each encode is an independent mutation. That is why the job is not consumed
/// by encoding and why its bytes are an [`Arc`] rather than a borrow — the
/// payload must outlive the descriptor that named it.
#[derive(Clone)]
pub struct UploadJob {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: UploadDescriptor,
}
impl UploadJob {
    /// Assembles a prepared upload.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`crate::api::platform::Device::create_buffer_upload`] and
    /// `create_texture_upload` may produce one. Those verbs wait on
    /// `api::platform`, which is not declared yet.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_buffer_upload and create_texture_upload call this \
                      once api::platform is declared"
        )
    )]
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, descriptor: UploadDescriptor) -> Self {
        Self {
            id,
            device,
            descriptor,
        }
    }

    /// This job's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that prepared this job.
    ///
    /// Section 18.7 puts an `UploadJob` on the same side of a device loss as the
    /// resources it writes: it belongs to the lost device, and using it against
    /// a newly requested device is [`RhiErrorKind::WrongDevice`] rather than an
    /// attempt to revive it.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// Capture/tooling can obtain the complete portable mutation descriptor.
    pub fn descriptor(&self) -> &UploadDescriptor {
        &self.descriptor
    }
}
/// Checks a buffer upload.
///
/// Section 17.3's buffer list:
///
/// ```text
/// dst usage includes COPY_DST
/// dst_offset + bytes.len() does not overflow
/// dst_offset + bytes.len() <= dst.size
/// bytes non-empty
/// BufferCopyLayoutLimits of RouteQuery::BufferToBuffer satisfied
/// DeviceIdentity matches
/// ```
///
/// The alignment limits are a parameter rather than something read from a
/// device: the caller of this function is the device façade, which has already
/// asked the route question and only calls an upload legal when the route is
/// supported. A device whose `RouteQuery::BufferToBuffer` answers `Unsupported`
/// refuses the upload with [`RhiErrorKind::Unsupported`] without reaching here.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_buffer_upload validates through this once api::platform is \
                  declared"
    )
)]
pub(crate) fn validate_buffer_upload(
    desc: &BufferUploadDescriptor,
    target: DeviceIdentity,
    limits: &BufferCopyLayoutLimits,
) -> RhiResult<()> {
    // The identity comparison comes first although section 17.3 lists it last:
    // root section 3.1 requires it in O(1) before anything else, and a
    // wrong-device answer must not be reachable only after a caller has already
    // acted on a range verdict.
    validate_buffer_ownership(&desc.dst, target)?;

    let dst = desc.dst.descriptor();
    if !dst.usage.contains(BufferUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the destination buffer was not created with COPY_DST usage",
        )
        .with_object(desc.dst.id()));
    }
    if desc.bytes.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "an upload must carry at least one byte",
        ));
    }

    // `bytes.len()` is a `usize`; every supported target is 64-bit or narrower,
    // so the widening is lossless and cannot wrap.
    let written = BufferRange::new(desc.dst_offset, desc.bytes.len() as u64);
    validate_buffer_range(written, dst.size)?;
    limits.validate(written.offset, written.size)
}
/// Checks a texture upload.
///
/// Section 17.3's texture list:
///
/// ```text
/// dst usage includes COPY_DST
/// TextureSubresourceLayers valid
/// origin/extent valid
/// dst.sample_count == 1
/// source_layout sufficiently covers source bytes
/// format/aspect valid
/// corresponding upload/copy route realizable
/// DeviceIdentity matches
/// ```
///
/// The native staging alignment is deliberately **not** part of this list:
/// section 17.3 states that upload does not require the caller to meet it, and
/// that the RHI may repack a normal CPU layout into private staging. The last
/// entry in the list — the route being realizable — is the device's, and is
/// asked by the `Device::create_texture_upload` façade rather than here.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_texture_upload validates through this once api::platform is \
                  declared"
    )
)]
pub(crate) fn validate_texture_upload(
    desc: &TextureUploadDescriptor,
    target: DeviceIdentity,
) -> RhiResult<()> {
    validate_texture_ownership(&desc.dst, target)?;
    let base = desc.dst.descriptor();
    if !base.usage.contains(TextureUsage::COPY_DST) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the destination texture was not created with COPY_DST usage",
        )
        .with_object(desc.dst.id()));
    }
    if base.sample_count != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a texture upload requires a single-sampled destination, not {} samples",
                base.sample_count
            ),
        )
        .with_object(desc.dst.id()));
    }
    validate_texture_region(base, desc.subresource, desc.origin, desc.extent)?;

    validate_host_texel_layout(desc.source_layout, desc.extent, base.format)?;

    let image_count = image_count(base, desc.subresource, desc.extent);
    if let Some(required) =
        source_bytes_required(desc.source_layout, desc.extent, image_count, base.format)
    {
        if (desc.bytes.len() as u64) < required {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the source layout describes {required} bytes but only {} were provided",
                    desc.bytes.len()
                ),
            ));
        }
    }
    Ok(())
}
/// How many source images a texture upload's region contains.
///
/// An "image" is one run of rows that `rows_per_image` separates from the next.
/// A 3D texture's Z slices are images, a 2D texture's array layers are images,
/// and neither is both — section 14.4 forbids conflating those two axes, and
/// [`validate_origin_extent`] has already pinned the layer count to one for a
/// 3D texture.
fn image_count(
    base: &TextureDescriptor,
    subresource: TextureSubresourceLayers,
    extent: Extent3d,
) -> u32 {
    match base.dimension {
        TextureDimension::D3 => extent.depth,
        _ => subresource.layer_count,
    }
}
