//! Portable upload, copy, and readback conformance workloads.
//!
//! The functions in this module deliberately own the complete public-RHI
//! resource and command sequence.  A backend fixture supplies only a ready
//! [`Device`], a COPY-capable submission lane, and its normal submit/wait
//! bridge; it must not duplicate the upload/copy/readback workload.  This
//! keeps a differing result attributable to lowering rather than to subtly
//! different backend test programs.
//!
//! The workloads use `Rgba8Unorm` and tightly packed host rows. Upload is
//! allowed to repack those host bytes for native footprint alignment. Readback
//! intentionally exposes its returned row layout, so the assertion compares
//! only each logical texel row and never mistakes native staging padding for
//! an API failure.

use std::sync::Arc;

use crate::api::command::{
    BufferCopy, BufferTextureCopy, RecordedWork, RecorderDescriptor, TextureCopy,
};
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::platform::Device;
use crate::api::resource::buffer::{BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::subresource::{
    HostTexelLayout, Origin3d, TextureAspect, TextureSubresourceLayers,
};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{
    BufferUploadDescriptor, ReadbackRequest, ReadbackTicket, ReadbackViewData,
    TextureUploadDescriptor,
};

/// Byte count used by the portable buffer transfer workload.
///
/// It is intentionally larger than one cache line and has no native alignment
/// requirement: the upload API accepts host bytes and the copy route is
/// responsible for its device-specific placement constraints.
pub(crate) const BUFFER_TRANSFER_SIZE: u64 = 4096;

/// Fixed dimensions used by the portable RGBA8 texture workload.
pub(crate) const TEXTURE_TRANSFER_WIDTH: u32 = 8;
/// Fixed dimensions used by the portable RGBA8 texture workload.
pub(crate) const TEXTURE_TRANSFER_HEIGHT: u32 = 4;

/// Dimensions for the complete texture-copy route workload. Unlike the basic
/// texture case this deliberately crosses texture -> buffer -> texture, so it
/// closes every direct portable copy verb without involving a shader fallback.
const TEXTURE_BUFFER_ROUND_TRIP_WIDTH: u32 = 4;
const TEXTURE_BUFFER_ROUND_TRIP_HEIGHT: u32 = 2;
const TEXTURE_BUFFER_ROUND_TRIP_ROW_PITCH: u32 = 256;

/// One portable buffer transfer recording and the bytes it must publish after
/// its accepted batch reaches completion.
pub(crate) struct BufferTransferRecording {
    /// Upload -> buffer copy -> readback work, ready for a COPY-capable lane.
    work: Option<RecordedWork>,
    /// Readback ticket for the copy destination.
    pub(crate) ticket: ReadbackTicket,
    /// Deterministic payload that CPU readback must equal byte-for-byte.
    pub(crate) expected: Vec<u8>,
}

impl BufferTransferRecording {
    /// Transfers the single-use portable recording into a fixture submission
    /// while retaining its ticket and oracle for the common assertion.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("buffer transfer recording was submitted more than once")
    }
}

/// One portable texture transfer recording and the logical texel bytes it must
/// publish after its accepted batch reaches completion.
pub(crate) struct TextureTransferRecording {
    /// Upload work. Fixtures submit it first, so the second work proves that
    /// resource state survives an accepted submission boundary.
    upload_work: Option<RecordedWork>,
    /// Texture copy -> readback work, submitted after [`Self::upload_work`].
    work: Option<RecordedWork>,
    /// Readback ticket for the texture-copy destination.
    pub(crate) ticket: ReadbackTicket,
    /// Deterministic tightly packed RGBA8 texels expected from readback's valid
    /// logical row bytes.
    pub(crate) expected: Vec<u8>,
}

impl TextureTransferRecording {
    /// Moves the upload recording into the fixture's first submission.
    ///
    /// The public workload deliberately crosses an accepted submission
    /// boundary.  Keeping the move behind this narrow method lets fixtures
    /// provide queue setup without re-implementing the portable commands.
    pub(crate) fn take_upload_work(&mut self) -> RecordedWork {
        self.upload_work
            .take()
            .expect("texture transfer upload recording was submitted more than once")
    }

    /// Moves the copy/readback recording into the fixture's second submission.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("texture transfer copy/readback recording was submitted more than once")
    }
}

/// One recording covering upload, texture copy, texture-to-buffer,
/// buffer-to-texture, and texture readback. It is public-RHI workload state;
/// a backend fixture supplies only a compatible COPY lane and submit bridge.
pub(crate) struct TextureBufferRoundTripRecording {
    work: Option<RecordedWork>,
    pub(crate) ticket: ReadbackTicket,
    expected: Vec<u8>,
}

impl TextureBufferRoundTripRecording {
    /// Moves the one completed recording into a submission batch.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("texture/buffer round-trip recording was submitted more than once")
    }
}

/// Records a buffer upload, device-side buffer copy, and buffer readback.
///
/// Fixture inputs: a live public [`Device`], then (outside this helper) a lane
/// whose published work domains accept copy records, and the fixture's normal
/// async submit/completion bridge.  No native handle, backend enum, or shader
/// artifact is involved.
pub(crate) fn record_buffer_upload_copy_readback(
    device: &Device,
    label: &'static str,
) -> BufferTransferRecording {
    let usage = BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST);
    let source = device
        .create_buffer(&BufferDescriptor::new(BUFFER_TRANSFER_SIZE, usage))
        .unwrap_or_else(|error| panic!("{label}: source buffer creation failed: {error}"));
    let destination = device
        .create_buffer(&BufferDescriptor::new(BUFFER_TRANSFER_SIZE, usage))
        .unwrap_or_else(|error| panic!("{label}: destination buffer creation failed: {error}"));
    let expected = (0..BUFFER_TRANSFER_SIZE)
        .map(|index| (index.wrapping_mul(73).wrapping_add(19) & 0xff) as u8)
        .collect::<Vec<_>>();
    let upload = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Label(Some(format!("{label} source upload"))),
            dst: source.clone(),
            dst_offset: 0,
            bytes: Arc::from(expected.as_slice()),
        })
        .unwrap_or_else(|error| panic!("{label}: buffer upload creation failed: {error}"));
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    recorder
        .encode_upload(&upload)
        .unwrap_or_else(|error| panic!("{label}: buffer upload recording failed: {error}"));
    recorder
        .copy_buffer(&BufferCopy {
            src: source,
            src_offset: 0,
            dst: destination.clone(),
            dst_offset: 0,
            size: BUFFER_TRANSFER_SIZE,
        })
        .unwrap_or_else(|error| panic!("{label}: buffer copy recording failed: {error}"));
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some(format!("{label} destination readback"))),
            src: destination,
            range: BufferRange::new(0, BUFFER_TRANSFER_SIZE),
        })
        .unwrap_or_else(|error| panic!("{label}: buffer readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: buffer recording completion failed: {error}"));
    BufferTransferRecording {
        work: Some(work),
        ticket,
        expected,
    }
}

/// Records a tightly packed RGBA8 texture upload, texture copy, and texture
/// readback across two accepted portable submissions.
///
/// The resource uses only `COPY_DST | COPY_SRC`; a backend that publishes the
/// corresponding routes must lower the full sequence without a shader or CPU
/// fallback.  The fixture inputs are the same as
/// [`record_buffer_upload_copy_readback`].
pub(crate) fn record_texture_upload_copy_readback(
    device: &Device,
    label: &'static str,
) -> TextureTransferRecording {
    let usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let source = device
        .create_texture(&TextureDescriptor::new_2d(
            TEXTURE_TRANSFER_WIDTH,
            TEXTURE_TRANSFER_HEIGHT,
            TextureFormat::Rgba8Unorm,
            usage,
        ))
        .unwrap_or_else(|error| panic!("{label}: source texture creation failed: {error}"));
    let destination = device
        .create_texture(&TextureDescriptor::new_2d(
            TEXTURE_TRANSFER_WIDTH,
            TEXTURE_TRANSFER_HEIGHT,
            TextureFormat::Rgba8Unorm,
            usage,
        ))
        .unwrap_or_else(|error| panic!("{label}: destination texture creation failed: {error}"));
    let expected = (0..(TEXTURE_TRANSFER_WIDTH * TEXTURE_TRANSFER_HEIGHT * 4))
        .map(|index| ((index * 29 + 7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let layers = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let origin = Origin3d { x: 0, y: 0, z: 0 };
    let extent = Extent3d::d2(TEXTURE_TRANSFER_WIDTH, TEXTURE_TRANSFER_HEIGHT);
    let bytes_per_row = TEXTURE_TRANSFER_WIDTH * 4;
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some(format!("{label} source upload"))),
            dst: source.clone(),
            subresource: layers,
            origin,
            extent,
            source_layout: HostTexelLayout {
                bytes_per_row,
                rows_per_image: TEXTURE_TRANSFER_HEIGHT,
            },
            bytes: Arc::from(expected.as_slice()),
        })
        .unwrap_or_else(|error| panic!("{label}: texture upload creation failed: {error}"));
    let mut upload_recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    upload_recorder
        .encode_upload(&upload)
        .unwrap_or_else(|error| panic!("{label}: texture upload recording failed: {error}"));
    let upload_work = upload_recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: texture upload completion failed: {error}"));
    // This separate recording is intentional: it verifies that the backend
    // carries the uploaded image's final state/layout into a later submission.
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: copy recorder creation failed: {error}"));
    recorder
        .copy_texture(&TextureCopy {
            src: source,
            src_subresource: layers,
            src_origin: origin,
            dst: destination.clone(),
            dst_subresource: layers,
            dst_origin: origin,
            extent,
        })
        .unwrap_or_else(|error| panic!("{label}: texture copy recording failed: {error}"));
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some(format!("{label} destination readback"))),
            src: destination,
            subresource: layers,
            origin,
            extent,
        })
        .unwrap_or_else(|error| panic!("{label}: texture readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: texture recording completion failed: {error}"));
    TextureTransferRecording {
        upload_work: Some(upload_work),
        work: Some(work),
        ticket,
        expected,
    }
}

/// Records every direct portable texture copy route in one deterministic chain:
/// tightly packed host upload -> texture copy -> texture-to-buffer ->
/// buffer-to-texture -> texture readback.
///
/// The buffer row pitch is explicitly 256 bytes. This is legal portable input
/// and is intentionally not inferred from DX12's footprint rules: it makes the
/// workload equally meaningful on Vulkan, Metal, WebGPU, and GL backends that
/// publish these routes.
pub(crate) fn record_texture_buffer_round_trip(
    device: &Device,
    label: &'static str,
) -> TextureBufferRoundTripRecording {
    let usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let make_texture = || {
        device
            .create_texture(&TextureDescriptor::new_2d(
                TEXTURE_BUFFER_ROUND_TRIP_WIDTH,
                TEXTURE_BUFFER_ROUND_TRIP_HEIGHT,
                TextureFormat::Rgba8Unorm,
                usage,
            ))
            .unwrap_or_else(|error| panic!("{label}: texture creation failed: {error}"))
    };
    let source = make_texture();
    let middle = make_texture();
    let destination = make_texture();
    let staging = device
        .create_buffer(&BufferDescriptor::new(
            u64::from(TEXTURE_BUFFER_ROUND_TRIP_ROW_PITCH)
                * u64::from(TEXTURE_BUFFER_ROUND_TRIP_HEIGHT),
            BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
        ))
        .unwrap_or_else(|error| panic!("{label}: copy buffer creation failed: {error}"));
    let expected = (1u8..=32).collect::<Vec<_>>();
    let layers = TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let origin = Origin3d { x: 0, y: 0, z: 0 };
    let extent = Extent3d::d2(
        TEXTURE_BUFFER_ROUND_TRIP_WIDTH,
        TEXTURE_BUFFER_ROUND_TRIP_HEIGHT,
    );
    let upload = device
        .create_texture_upload(TextureUploadDescriptor {
            label: Label(Some(format!("{label} upload"))),
            dst: source.clone(),
            subresource: layers,
            origin,
            extent,
            source_layout: HostTexelLayout {
                bytes_per_row: TEXTURE_BUFFER_ROUND_TRIP_WIDTH * 4,
                rows_per_image: TEXTURE_BUFFER_ROUND_TRIP_HEIGHT,
            },
            bytes: Arc::from(expected.as_slice()),
        })
        .unwrap_or_else(|error| panic!("{label}: texture upload creation failed: {error}"));
    let buffer_copy = BufferTextureCopy {
        buffer: staging,
        buffer_offset: 0,
        bytes_per_row: TEXTURE_BUFFER_ROUND_TRIP_ROW_PITCH,
        rows_per_image: TEXTURE_BUFFER_ROUND_TRIP_HEIGHT,
        texture: middle.clone(),
        texture_subresource: layers,
        texture_origin: origin,
        extent,
    };
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    recorder
        .encode_upload(&upload)
        .unwrap_or_else(|error| panic!("{label}: texture upload recording failed: {error}"));
    recorder
        .copy_texture(&TextureCopy {
            src: source,
            src_subresource: layers,
            src_origin: origin,
            dst: middle.clone(),
            dst_subresource: layers,
            dst_origin: origin,
            extent,
        })
        .unwrap_or_else(|error| panic!("{label}: texture copy recording failed: {error}"));
    recorder
        .copy_texture_to_buffer(&buffer_copy)
        .unwrap_or_else(|error| panic!("{label}: texture-to-buffer recording failed: {error}"));
    recorder
        .copy_buffer_to_texture(&BufferTextureCopy {
            texture: destination.clone(),
            ..buffer_copy
        })
        .unwrap_or_else(|error| panic!("{label}: buffer-to-texture recording failed: {error}"));
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some(format!("{label} readback"))),
            src: destination,
            subresource: layers,
            origin,
            extent,
        })
        .unwrap_or_else(|error| panic!("{label}: texture readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    TextureBufferRoundTripRecording {
        work: Some(work),
        ticket,
        expected,
    }
}

/// Asserts that an accepted buffer-transfer workload published exactly its
/// uploaded bytes. Call only after the fixture has observed `Complete` for the
/// submitted batch.
pub(crate) async fn assert_buffer_transfer(recording: &BufferTransferRecording, label: &str) {
    let view = recording
        .ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: buffer readback failed: {error}"));
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("{label}: buffer transfer returned texture readback data");
    };
    assert_eq!(
        bytes,
        recording.expected.as_slice(),
        "{label}: buffer bytes differ"
    );
}

/// Asserts that an accepted RGBA8 texture-transfer workload returned exactly
/// the uploaded texels in its valid logical row bytes.
///
/// Native row padding is explicitly represented by `ReadbackTexelLayout`; it
/// is not an API failure and is therefore skipped before comparing texels.
pub(crate) async fn assert_texture_transfer(recording: &TextureTransferRecording, label: &str) {
    let view = recording
        .ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: texture readback failed: {error}"));
    let ReadbackViewData::Texture { bytes, layout } = view.data() else {
        panic!("{label}: texture transfer returned buffer readback data");
    };
    let logical_row_bytes = (TEXTURE_TRANSFER_WIDTH * 4) as usize;
    let native_row_bytes = layout.bytes_per_row as usize;
    assert!(
        native_row_bytes >= logical_row_bytes,
        "{label}: readback row pitch is smaller than its valid RGBA8 texel row"
    );
    assert!(
        layout.rows_per_image >= TEXTURE_TRANSFER_HEIGHT,
        "{label}: readback rows-per-image is smaller than the requested region"
    );
    let required_size = native_row_bytes
        .checked_mul((TEXTURE_TRANSFER_HEIGHT - 1) as usize)
        .and_then(|prefix| prefix.checked_add(logical_row_bytes))
        .expect("fixed transfer dimensions fit usize");
    assert!(
        bytes.len() >= required_size,
        "{label}: readback byte span is shorter than its declared row layout"
    );
    let mut actual = Vec::with_capacity(recording.expected.len());
    for row in 0..TEXTURE_TRANSFER_HEIGHT as usize {
        let start = row * native_row_bytes;
        actual.extend_from_slice(&bytes[start..start + logical_row_bytes]);
    }
    assert_eq!(actual, recording.expected, "{label}: texture texels differ");
}

/// Asserts the valid texels from the complete texture/buffer round trip.
///
/// Returned texture readback rows can have native padding, so only the valid
/// sixteen RGBA8 bytes of each requested row participate in comparison.
pub(crate) async fn assert_texture_buffer_round_trip(
    recording: &TextureBufferRoundTripRecording,
    label: &str,
) {
    let view = recording
        .ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: texture readback failed: {error}"));
    let ReadbackViewData::Texture { bytes, layout } = view.data() else {
        panic!("{label}: texture/buffer round trip returned buffer readback data");
    };
    let logical_row_bytes = (TEXTURE_BUFFER_ROUND_TRIP_WIDTH * 4) as usize;
    let native_row_bytes = layout.bytes_per_row as usize;
    assert!(
        native_row_bytes >= logical_row_bytes,
        "{label}: returned row pitch is smaller than valid RGBA8 bytes"
    );
    let required_size = native_row_bytes
        .checked_mul((TEXTURE_BUFFER_ROUND_TRIP_HEIGHT - 1) as usize)
        .and_then(|prefix| prefix.checked_add(logical_row_bytes))
        .expect("fixed round-trip dimensions fit usize");
    assert!(
        bytes.len() >= required_size,
        "{label}: texture readback is shorter than its declared row layout"
    );
    let mut actual = Vec::with_capacity(recording.expected.len());
    for row in 0..TEXTURE_BUFFER_ROUND_TRIP_HEIGHT as usize {
        let start = row * native_row_bytes;
        actual.extend_from_slice(&bytes[start..start + logical_row_bytes]);
    }
    assert_eq!(
        actual, recording.expected,
        "{label}: texture/buffer direct-copy chain changed texels"
    );
}
