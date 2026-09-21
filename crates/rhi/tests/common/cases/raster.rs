//! Portable off-screen raster/readback conformance workload.
//!
//! This module intentionally contains no backend shader representation or
//! provider construction.  A fixture supplies a pipeline whose portable target
//! signature is `Rgba8Unorm` and, when needed, a vertex binding.  The workload
//! then owns the attachment, raster scope, draw, readback request, and CPU-side
//! pixel assertion.  Keeping those steps here makes one result comparable
//! across every backend that publishes the same capability route.

use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
    RecordedWork, RecorderDescriptor, StoreOp,
};
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::pipeline::RasterPipeline;
use crate::api::platform::Device;
use crate::api::resource::buffer::BufferBinding;
use crate::api::resource::subresource::{Origin3d, TextureAspect, TextureSubresourceLayers};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackTicket, ReadbackViewData};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::ShaderLocation;

/// Fixture-provided portable inputs for one basic color raster draw.
///
/// The fixture is responsible only for creating code-form-specific shaders,
/// compiling the resulting portable pipeline, and (if its vertex shader uses
/// one) uploading a portable vertex buffer.  It must not record native work or
/// duplicate the attachment/readback sequence in a backend-local test.
#[derive(Clone)]
pub(crate) struct OffscreenRasterFixture {
    /// A pipeline with one `Rgba8Unorm` color target at location zero.
    pub(crate) pipeline: RasterPipeline,
    /// Vertex stream for slot zero, or `None` for a vertex-index shader.
    pub(crate) vertex: Option<BufferBinding>,
    /// Range passed unchanged to the portable `draw` command.
    pub(crate) vertices: core::ops::Range<u32>,
    /// Instance range passed unchanged to the portable `draw` command.
    pub(crate) instances: core::ops::Range<u32>,
    /// Texel coordinate whose expected result distinguishes clear from draw.
    pub(crate) sample: (u32, u32),
    /// Expected straight `Rgba8Unorm` bytes at [`Self::sample`].
    pub(crate) expected: [u8; 4],
}

/// Portable output of [`record_offscreen_raster_rgba8`].
pub(crate) struct OffscreenRasterRecording {
    /// Complete work for a raster-capable submission lane.
    work: Option<RecordedWork>,
    /// Ticket for the full color attachment readback.
    pub(crate) ticket: ReadbackTicket,
    /// Attachment extent, retained to validate the requested CPU sample.
    pub(crate) extent: Extent3d,
    /// Texel coordinate whose expected result distinguishes clear from draw.
    ///
    /// This deliberately retains the assertion data, rather than the fixture:
    /// after recording, `RecordedWork` is the sole owner of the pipeline and
    /// resource backing needed by accepted GPU work.  A fixture test can drop
    /// its shader/pipeline handles before submission to exercise that contract.
    pub(crate) sample: (u32, u32),
    /// Expected straight `Rgba8Unorm` bytes at [`Self::sample`].
    pub(crate) expected: [u8; 4],
}

impl OffscreenRasterRecording {
    /// Transfers the one recorded workload into a submission plan.
    ///
    /// Keeping the assertion metadata in `self` permits a fixture to submit
    /// the work and then use the same recording for the portable readback
    /// assertion without retaining caller-side pipeline objects.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("one portable raster recording may be submitted only once")
    }
}

/// Records one `Rgba8Unorm` off-screen clear, draw, and texture readback.
///
/// `extent` is deliberately supplied by the caller rather than fixed to 4x4 or
/// 8x8.  This lets a single workload expose backend copy-pitch behaviour while
/// its assertion uses the public [`ReadbackViewData`] layout instead of making
/// a DX12-specific tightly-packed assumption.
pub(crate) fn record_offscreen_raster_rgba8(
    device: &Device,
    fixture: OffscreenRasterFixture,
    extent: Extent3d,
    label: &'static str,
) -> OffscreenRasterRecording {
    assert!(
        extent.width > 0 && extent.height > 0 && extent.depth == 1,
        "{label}: portable off-screen raster case requires one non-empty 2D layer"
    );
    let OffscreenRasterFixture {
        pipeline,
        vertex,
        vertices,
        instances,
        sample,
        expected,
    } = fixture;
    assert!(
        sample.0 < extent.width && sample.1 < extent.height,
        "{label}: requested CPU sample lies outside the portable attachment"
    );

    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            extent.width,
            extent.height,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .unwrap_or_else(|error| panic!("{label}: RGBA8 attachment creation failed: {error}"));
    let view = device
        .create_texture_view(
            &texture,
            &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)
                .expect("a whole 2D color view descriptor is valid"),
        )
        .unwrap_or_else(|error| panic!("{label}: RGBA8 attachment view creation failed: {error}"));
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            depth_slice: None,
            view: ColorAttachmentView::Texture(view),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        },
    );
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    {
        let mut raster = recorder
            .begin_raster(&scope)
            .unwrap_or_else(|error| panic!("{label}: raster scope creation failed: {error}"));
        raster.set_pipeline(&pipeline).unwrap_or_else(|error| {
            panic!("{label}: portable raster pipeline bind failed: {error}")
        });
        if let Some(vertex) = &vertex {
            raster
                .set_vertex_buffer(0, vertex)
                .unwrap_or_else(|error| panic!("{label}: portable vertex binding failed: {error}"));
        }
        raster
            .draw(vertices, instances)
            .unwrap_or_else(|error| panic!("{label}: portable draw recording failed: {error}"));
        raster
            .end()
            .unwrap_or_else(|error| panic!("{label}: raster scope close failed: {error}"));
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some(format!("{label} color output"))),
            src: texture,
            subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent,
        })
        .unwrap_or_else(|error| panic!("{label}: color readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: portable recording completion failed: {error}"));
    OffscreenRasterRecording {
        work: Some(work),
        ticket,
        extent,
        sample,
        expected,
    }
}

/// Asserts the fixture's expected texel after its accepted work completed.
///
/// The public readback contract explicitly permits padded rows.  This helper
/// therefore derives the byte offset from `ReadbackTexelLayout::bytes_per_row`
/// rather than assuming Vulkan/Metal packing or D3D12's 256-byte footprint.
pub(crate) async fn assert_offscreen_raster_rgba8(
    recording: &OffscreenRasterRecording,
    label: &str,
) {
    let view = recording
        .ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: color readback failed: {error}"));
    let ReadbackViewData::Texture { bytes, layout } = view.data() else {
        panic!("{label}: off-screen color readback returned buffer data");
    };
    let (x, y) = recording.sample;
    assert!(
        x < recording.extent.width && y < recording.extent.height,
        "{label}: fixture sample escaped the retained attachment extent"
    );
    let offset = usize::try_from(
        u64::from(y)
            .checked_mul(u64::from(layout.bytes_per_row))
            .and_then(|row| row.checked_add(u64::from(x) * 4))
            .expect("portable RGBA8 sample offset must not overflow"),
    )
    .expect("portable readback allocation must fit host address space");
    let actual: [u8; 4] = bytes
        .get(offset..offset + 4)
        .unwrap_or_else(|| {
            panic!("{label}: readback layout does not contain requested RGBA8 texel")
        })
        .try_into()
        .expect("exact RGBA8 texel");
    assert_eq!(
        actual, recording.expected,
        "{label}: raster output differs at ({x}, {y}); readback row pitch was {} bytes",
        layout.bytes_per_row,
    );
}
