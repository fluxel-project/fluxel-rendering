//! Section 17: upload jobs — caller bytes into a resource.

use super::validate_texture_region;
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::resource::buffer::{
    Buffer, BufferRange, BufferUsage, validate_buffer_ownership, validate_buffer_range,
};
use crate::api::resource::route::{BufferCopyLayoutLimits, RouteQuery, RouteSupport};
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
    /// [`crate::api::platform::Device::create_texture_upload`] may produce one.
    /// Those verbs exist and are the only callers this is written for, but they stop
    /// before the staging bytes are accepted — nothing can mint the identity below
    /// until a backend staging path does — so nothing calls this yet.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_buffer_upload and create_texture_upload call this once the backend staging path lands and can mint the job's identity"
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

// The two creation verbs of this chapter, written here for the reason adjudication
// A28 records: section 17.3 declares both beside the object they produce, so the
// definition site is the owner. An inherent impl block attaches to `Device`
// wherever it is written, so callers and intra-doc links that name
// `crate::api::platform::Device::create_buffer_upload` or `create_texture_upload`
// still resolve here.
impl Device {
    /// Prepares a buffer upload.
    ///
    /// Section 17.3's buffer verb. It is an inherent method written in the upload
    /// chapter rather than in `api::platform` because section 17.3 declares it beside
    /// the object it produces: the definition site is the owner (adjudication A28).
    ///
    /// The descriptor arrives **by value**, as section 17.3 declares it, and the
    /// reason is section 17.2's: the job owns its retained source bytes and the same
    /// job may be encoded repeatedly, so nothing is borrowed from a caller who would
    /// then have to keep the descriptor alive.
    ///
    /// Section 17.3's route entry is asked *here* rather than inside the validator,
    /// because it is a probed device fact and the validator takes its answer as a
    /// parameter. The order matters: a device with no direct buffer-to-buffer route
    /// refuses the upload outright, since section 9.4 forbids substituting a staging
    /// CPU round-trip for a route that does not exist, and a route that exists but
    /// states no buffer copy layout cannot answer the alignment rule at all.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::WrongDevice`] when the destination buffer belongs to another
    /// device — section 3.1's comparison runs first, before any other verdict about
    /// the range or the route. [`RhiErrorKind::Unsupported`] when the device has no
    /// direct buffer-to-buffer route, or states no copy alignment for it.
    /// [`RhiErrorKind::InvalidUsage`] when the destination was not created with
    /// `COPY_DST`, when the payload is empty, when the written range leaves the
    /// buffer, or when it does not meet the copy alignment.
    pub fn create_buffer_upload(&self, desc: BufferUploadDescriptor) -> RhiResult<UploadJob> {
        // Section 3.1's comparison is repeated here even though
        // `validate_buffer_upload` performs it too, and the repetition is the point:
        // this verb has to ask the device for the copy layout *before* it can call
        // the validator, and without this line a foreign buffer on a device with no
        // buffer-to-buffer route would be refused as `Unsupported` — a verdict about
        // the route — before anyone checked whose buffer it was. The `# Errors`
        // section above promises this ordering, so the check belongs ahead of the
        // route read, not only inside the validator. The validator keeps its own
        // copy because it is callable on its own; an O(1) identity comparison run
        // twice costs nothing next to answering a caller with the wrong kind.
        validate_buffer_ownership(&desc.dst, self.identity())?;

        // Section 6.5's liveness verdict, after the ownership comparison and
        // before the route read. A destination belonging to another device is
        // `WrongDevice` even when this device is also lost.
        self.require_active()?;

        let route = self.capabilities().route(&RouteQuery::BufferToBuffer);
        let capabilities = match route {
            RouteSupport::Supported(capabilities) => capabilities,
            RouteSupport::Unsupported => {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    "this device has no direct buffer-to-buffer copy route, and section 9.4 \
                     forbids substituting one, so an upload into a buffer cannot be prepared",
                ));
            }
        };
        let limits = capabilities.buffer_copy_layout().ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                "the buffer-to-buffer route reports no buffer copy layout, so this device \
                 states no offset or size alignment for the copy an upload would need",
            )
        })?;
        validate_buffer_upload(&desc, self.identity(), &limits)?;
        unimplemented!(
            "Device::create_buffer_upload needs a backend staging path to write the {} \
             retained bytes at offset {} of a buffer on device {:?}; the portable \
             contract is fixed and its refusal paths above are built, but no backend \
             port is built. The capability snapshot this verb asks the copy route and \
             its alignment from is a backend-port deliverable as well, so on today's \
             tree the call stops inside Device::capabilities before reaching this point",
            desc.bytes.len(),
            desc.dst_offset,
            self.identity()
        )
    }

    /// Prepares a texture upload.
    ///
    /// Section 17.3's texture verb. It is an inherent method written in the upload
    /// chapter rather than in `api::platform` because section 17.3 declares it beside
    /// the object it produces: the definition site is the owner (adjudication A28).
    ///
    /// The descriptor arrives **by value** for the reason
    /// [`Device::create_buffer_upload`] gives: section 17.2 makes the job own its
    /// retained bytes.
    ///
    /// Section 17.3's list is checked in the order that keeps the portable verdicts
    /// ahead of the device's: the identity comparison, the usage bit, the
    /// single-sampled destination, the region against the texture's own descriptor,
    /// the host layout, and the byte count — none of which needs a device fact — and
    /// then the route. The route key is built from the destination texture's
    /// dimensionality and format and the aspect the subresource names, which is the
    /// same key the buffer-to-texture *copy* command asks about, so the preflight
    /// here and the copy's own check cannot disagree about what was asked.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::WrongDevice`] when the destination texture belongs to another
    /// device. [`RhiErrorKind::Unsupported`] when the device has no direct
    /// buffer-to-texture route for this key. [`RhiErrorKind::InvalidUsage`] for every
    /// region, usage, layout, or byte-count violation listed above.
    pub fn create_texture_upload(&self, desc: TextureUploadDescriptor) -> RhiResult<UploadJob> {
        validate_texture_upload(&desc, self.identity())?;

        // Section 6.5's liveness verdict, after the ownership comparison inside
        // the validator and before the route read.
        self.require_active()?;

        let route = self.capabilities().route(&RouteQuery::BufferToTexture {
            dimension: desc.dst.descriptor().dimension,
            format: desc.dst.descriptor().format,
            aspect: desc.subresource.aspect,
        });
        if !route.is_supported() {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "this device has no direct buffer-to-texture copy route for this \
                 destination's shape, and section 9.4 forbids substituting one, so an \
                 upload into this texture cannot be prepared",
            ));
        }
        unimplemented!(
            "Device::create_texture_upload needs a backend staging path to write the {} \
             retained bytes into a texture region on device {:?}; the portable contract \
             is fixed and its refusal paths above are built, but no backend port is \
             built. The capability snapshot this verb asks the copy route from is a \
             backend-port deliverable as well, so on today's tree the call stops inside \
             Device::capabilities before reaching this point",
            desc.bytes.len(),
            self.identity()
        )
    }
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
