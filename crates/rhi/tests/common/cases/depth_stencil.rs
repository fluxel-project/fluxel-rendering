//! Portable depth/stencil attachment conformance workloads.
//!
//! This module deliberately contains neither shader code nor native-provider
//! setup.  A backend fixture supplies the already-created raster pipeline and
//! vertex buffer; this module owns the portable resources, capability gate,
//! command recording, texture readbacks, and CPU-visible assertions.  Keeping
//! that split makes one result comparable across every backend rather than
//! accidentally testing slightly different DX12, Vulkan, Metal, or GL cases.

use crate::api::command::RecordedWork;
use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
    DepthStencilAttachment, LoadOp, RasterScopeDescriptor, RecorderDescriptor,
    StencilAttachmentMode, StoreOp,
};
use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::identity::Label;
use crate::api::pipeline::RasterPipeline;
use crate::api::platform::Device;
use crate::api::resource::buffer::{Buffer, BufferBinding, BufferRange};
use crate::api::resource::subresource::{Origin3d, TextureAspect, TextureSubresourceLayers};
use crate::api::resource::texture::{Extent3d, TextureDescriptor, TextureDimension, TextureUsage};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackTicket, ReadbackViewData};
use crate::api::resource::view::{TextureView, TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::ShaderLocation;

const EXTENT: u32 = 8;
const COLOR_LAYERS: TextureSubresourceLayers = TextureSubresourceLayers {
    aspect: TextureAspect::Color,
    mip_level: 0,
    base_layer: 0,
    layer_count: 1,
};

/// Why a portable depth/stencil workload was deliberately not recorded.
///
/// This is an `Unsupported` test result, not a successful no-op.  Fixtures
/// should report it as capability-gated and must not substitute another depth
/// format or sample count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DepthStencilCaseSkip {
    /// The precise RGBA8 attachment plus readback route is not published.
    ColorReadbackRoute,
    /// The requested depth/stencil attachment format is not published.
    DepthStencilRoute(TextureFormat),
}

/// Backend-code-form-independent inputs for strict depth-boundary rendering.
///
/// The fixture's pipeline must use a strict `Less` depth compare, write depth,
/// consume the supplied position-only vertex buffer, and output
/// `admitted_color` for the triangle.  The case makes the two boundaries
/// observable: `z = 0` is rejected after a zero clear, and admitted after a
/// one clear.  That proves the attachment clear, depth state, draw, and
/// readback path together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DepthCompareCase {
    /// Exact attachment format used by both pipeline and texture.
    pub(crate) depth_format: TextureFormat,
    /// The fragment color the fixture's shader writes for an admitted sample.
    pub(crate) admitted_color: [u8; 4],
}

impl DepthCompareCase {
    /// A conventional normalized-color case for a `Depth32Float` pipeline.
    pub(crate) const fn depth32_float(admitted_color: [u8; 4]) -> Self {
        Self {
            depth_format: TextureFormat::Depth32Float,
            admitted_color,
        }
    }
}

/// Work and readbacks produced by [`record_depth_compare_boundaries`].
pub(crate) struct DepthCompareRecording {
    /// The one portable recording, ready for a raster/copy-capable lane.
    work: Option<RecordedWork>,
    /// Color output from the zero-clear (rejected) depth boundary.
    pub(crate) rejected: ReadbackTicket,
    /// Color output from the one-clear (admitted) depth boundary.
    pub(crate) admitted: ReadbackTicket,
    case: DepthCompareCase,
}

impl DepthCompareRecording {
    /// Transfers the single-use command recording into a submission plan while
    /// retaining its readback tickets for the post-completion assertion.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("depth comparison recording was submitted more than once")
    }
}

/// Portable inputs for a clear/store depth and optional stencil attachment
/// scope.  It is useful where a backend exposes a combined format but does not
/// yet provide a fixture shader for the stricter depth-compare workload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DepthStencilClearStoreCase {
    /// Exact depth-only or combined depth/stencil attachment format.
    pub(crate) format: TextureFormat,
    /// Clear value for the depth plane. `None` leaves the plane unbound.
    pub(crate) depth_clear: Option<f32>,
    /// Clear value for the stencil plane. `None` leaves the plane unbound.
    pub(crate) stencil_clear: Option<u32>,
    /// Color clear used as a readback-visible proof that the scope executed.
    pub(crate) color: [u8; 4],
}

/// Work and color readback produced by [`record_depth_stencil_clear_store`].
pub(crate) struct DepthStencilClearStoreRecording {
    /// The one portable recording, ready for a raster/copy-capable lane.
    work: Option<RecordedWork>,
    /// Readback of the stored color attachment.
    pub(crate) ticket: ReadbackTicket,
}

impl DepthStencilClearStoreRecording {
    /// Transfers the single-use command recording into a submission plan while
    /// retaining its readback ticket for the post-completion assertion.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("depth/stencil clear-store recording was submitted more than once")
    }
}

/// Records strict depth comparison at the exact clear boundaries.
///
/// The fixture creates only the shader/pipeline and position vertex buffer;
/// this routine retains all public RHI recording semantics.  A returned skip
/// means the adapter did not publish the exact route, while creation or record
/// failure after a positive fact is a conformance failure.
pub(crate) fn record_depth_compare_boundaries(
    device: &Device,
    pipeline: &RasterPipeline,
    vertex: &Buffer,
    case: DepthCompareCase,
    label: &'static str,
) -> Result<DepthCompareRecording, DepthStencilCaseSkip> {
    require_attachment_routes(device, case.depth_format)?;

    let depth = create_depth_stencil_texture(device, case.depth_format, label);
    let depth_view = whole_view(device, &depth, label, "depth attachment");
    let (rejected_target, rejected_view) = color_target(device, label);
    let (admitted_target, admitted_view) = color_target(device, label);
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    for (scope, phase) in [
        (
            depth_scope(rejected_view, depth_view.clone(), 0.0),
            "zero-clear rejection",
        ),
        (
            depth_scope(admitted_view, depth_view, 1.0),
            "one-clear admission",
        ),
    ] {
        let mut raster = recorder
            .begin_raster(&scope)
            .unwrap_or_else(|error| panic!("{label}: {phase} scope failed: {error}"));
        raster
            .set_pipeline(pipeline)
            .unwrap_or_else(|error| panic!("{label}: {phase} pipeline failed: {error}"));
        raster
            .set_vertex_buffer(
                0,
                &BufferBinding::new(vertex.clone(), BufferRange::new(0, 24)),
            )
            .unwrap_or_else(|error| panic!("{label}: {phase} vertex binding failed: {error}"));
        raster
            .draw(0..3, 0..1)
            .unwrap_or_else(|error| panic!("{label}: {phase} draw failed: {error}"));
        raster
            .end()
            .unwrap_or_else(|error| panic!("{label}: {phase} scope close failed: {error}"));
    }
    let rejected = encode_color_readback(&mut recorder, rejected_target, label, "rejected");
    let admitted = encode_color_readback(&mut recorder, admitted_target, label, "admitted");
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    Ok(DepthCompareRecording {
        work: Some(work),
        rejected,
        admitted,
        case,
    })
}

/// Asserts the two exact depth outcomes after the caller has observed work
/// completion.  The readback layout is consumed rather than assuming D3D12's
/// 256-byte footprint or Vulkan's often-tight rows.
pub(crate) async fn assert_depth_compare_boundaries(
    recording: &DepthCompareRecording,
    label: &str,
) {
    assert_eq!(
        center_rgba8(&recording.rejected, label, "rejected").await,
        [0, 0, 0, 255],
        "{label}: z=0 must fail strict Less against a zero depth clear"
    );
    assert_eq!(
        center_rgba8(&recording.admitted, label, "admitted").await,
        recording.case.admitted_color,
        "{label}: z=0 must pass strict Less against a one depth clear"
    );
}

/// Records an empty raster scope that clears and stores the requested
/// depth/stencil planes along with a readback-visible color attachment.
pub(crate) fn record_depth_stencil_clear_store(
    device: &Device,
    case: DepthStencilClearStoreCase,
    label: &'static str,
) -> Result<DepthStencilClearStoreRecording, DepthStencilCaseSkip> {
    require_attachment_routes(device, case.format)?;
    let depth_stencil = create_depth_stencil_texture(device, case.format, label);
    let depth_stencil_view = whole_view(device, &depth_stencil, label, "depth/stencil attachment");
    let (color, color_view) = color_target(device, label);
    let depth = case
        .depth_clear
        .map(|clear| DepthAttachmentMode::ReadWrite {
            load: LoadOp::Clear(clear),
            store: StoreOp::Store,
        });
    let stencil = case
        .stencil_clear
        .map(|clear| StencilAttachmentMode::ReadWrite {
            load: LoadOp::Clear(clear),
            store: StoreOp::Store,
        });
    let scope = RasterScopeDescriptor::new()
        .with_color(
            ShaderLocation::new(0),
            ColorAttachment {
                depth_slice: None,
                view: ColorAttachmentView::Texture(color_view),
                load: LoadOp::Clear(ColorClearValue::Float(color_to_float(case.color))),
                store: StoreOp::Store,
                resolve: None,
            },
        )
        .with_depth_stencil(DepthStencilAttachment {
            view: depth_stencil_view,
            depth,
            stencil,
        });
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    recorder
        .begin_raster(&scope)
        .unwrap_or_else(|error| panic!("{label}: depth/stencil scope failed: {error}"))
        .end()
        .unwrap_or_else(|error| panic!("{label}: depth/stencil scope close failed: {error}"));
    let ticket = encode_color_readback(&mut recorder, color, label, "clear/store");
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    Ok(DepthStencilClearStoreRecording {
        work: Some(work),
        ticket,
    })
}

/// Asserts that a completed clear/store scope produced its requested color.
pub(crate) async fn assert_depth_stencil_clear_store(
    ticket: &ReadbackTicket,
    expected_color: [u8; 4],
    label: &str,
) {
    assert_eq!(
        first_rgba8(ticket, label, "clear/store").await,
        expected_color,
        "{label}: stored color clear differs"
    );
}

fn require_attachment_routes(
    device: &Device,
    depth_format: TextureFormat,
) -> Result<(), DepthStencilCaseSkip> {
    let color = TextureSupportQuery::new(
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        1,
    );
    if !device.capabilities().texture_support(&color).is_supported() {
        return Err(DepthStencilCaseSkip::ColorReadbackRoute);
    }
    let depth = TextureSupportQuery::new(
        TextureDimension::D2,
        depth_format,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        1,
    );
    if !device.capabilities().texture_support(&depth).is_supported() {
        return Err(DepthStencilCaseSkip::DepthStencilRoute(depth_format));
    }
    Ok(())
}

fn create_depth_stencil_texture(
    device: &Device,
    format: TextureFormat,
    label: &str,
) -> crate::api::resource::texture::Texture {
    device
        .create_texture(&TextureDescriptor::new_2d(
            EXTENT,
            EXTENT,
            format,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        ))
        .unwrap_or_else(|error| {
            panic!("{label}: queried depth/stencil texture creation failed: {error}")
        })
}

fn color_target(
    device: &Device,
    label: &str,
) -> (crate::api::resource::texture::Texture, TextureView) {
    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            EXTENT,
            EXTENT,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC),
        ))
        .unwrap_or_else(|error| panic!("{label}: queried color texture creation failed: {error}"));
    let view = whole_view(device, &texture, label, "color attachment");
    (texture, view)
}

fn whole_view(
    device: &Device,
    texture: &crate::api::resource::texture::Texture,
    label: &str,
    kind: &str,
) -> TextureView {
    let descriptor = TextureViewDescriptor::whole(texture, TextureViewDimension::D2)
        .unwrap_or_else(|error| panic!("{label}: {kind} view descriptor failed: {error}"));
    device
        .create_texture_view(texture, &descriptor)
        .unwrap_or_else(|error| panic!("{label}: {kind} view creation failed: {error}"))
}

fn depth_scope(color: TextureView, depth: TextureView, clear_depth: f32) -> RasterScopeDescriptor {
    RasterScopeDescriptor::new()
        .with_color(
            ShaderLocation::new(0),
            ColorAttachment {
                depth_slice: None,
                view: ColorAttachmentView::Texture(color),
                load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                store: StoreOp::Store,
                resolve: None,
            },
        )
        .with_depth_stencil(DepthStencilAttachment {
            view: depth,
            depth: Some(DepthAttachmentMode::ReadWrite {
                load: LoadOp::Clear(clear_depth),
                store: StoreOp::Store,
            }),
            stencil: None,
        })
}

fn encode_color_readback(
    recorder: &mut crate::api::command::CommandRecorder,
    color: crate::api::resource::texture::Texture,
    label: &str,
    phase: &str,
) -> ReadbackTicket {
    recorder
        .encode_readback(ReadbackRequest::Texture {
            label: Label(Some(format!("{label} {phase} color"))),
            src: color,
            subresource: COLOR_LAYERS,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            extent: Extent3d::d2(EXTENT, EXTENT),
        })
        .unwrap_or_else(|error| panic!("{label}: {phase} color readback recording failed: {error}"))
}

async fn center_rgba8(ticket: &ReadbackTicket, label: &str, phase: &str) -> [u8; 4] {
    rgba8_at(ticket, EXTENT / 2, EXTENT / 2, label, phase).await
}

async fn first_rgba8(ticket: &ReadbackTicket, label: &str, phase: &str) -> [u8; 4] {
    rgba8_at(ticket, 0, 0, label, phase).await
}

async fn rgba8_at(ticket: &ReadbackTicket, x: u32, y: u32, label: &str, phase: &str) -> [u8; 4] {
    let view = ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: {phase} readback failed: {error}"));
    let ReadbackViewData::Texture { bytes, layout } = view.data() else {
        panic!("{label}: {phase} expected texture readback");
    };
    let offset = (y as usize)
        .checked_mul(layout.bytes_per_row as usize)
        .and_then(|row| row.checked_add((x as usize) * 4))
        .unwrap_or_else(|| panic!("{label}: {phase} texel offset overflow"));
    bytes
        .get(offset..offset + 4)
        .unwrap_or_else(|| {
            panic!("{label}: {phase} texel lies outside published layout {layout:?}")
        })
        .try_into()
        .expect("RGBA8 texel width")
}

fn color_to_float(color: [u8; 4]) -> [f32; 4] {
    color.map(|channel| channel as f32 / 255.0)
}
