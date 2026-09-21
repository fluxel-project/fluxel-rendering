//! Shared backend-injected hardware conformance cases.
//!
//! This file deliberately lives under `crates/rhi/tests/`: the cases are about
//! the portable RHI contract, not one backend's implementation. It is included
//! as a crate module during native tests because ordinary Cargo integration-test
//! binaries cannot construct the intentionally crate-private native provider.
//! Backend fixtures inject only provider creation, shader code form, and host
//! surface glue; recording, submission plan construction, completion and CPU
//! assertions belong here as common test behaviour.

/// Portable workloads, split by RHI domain. These modules never name native
/// provider, shader-code-form, or surface-host types.
pub(crate) mod cases {
    pub(crate) mod compute;
    pub(crate) mod core_device;
    pub(crate) mod depth_stencil;
    pub(crate) mod presentation;
    pub(crate) mod raster;
    pub(crate) mod transfer;
}

use std::sync::Arc;

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, ComputeScopeDescriptor, LoadOp,
    RasterScopeDescriptor, RecordedWork, RecorderDescriptor, StoreOp,
};
use crate::api::format::TextureFormat;
use crate::api::identity::Label;
use crate::api::pipeline::ComputePipeline;
use crate::api::platform::Device;
use crate::api::query::{QuerySetDescriptor, QueryType};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::texture::{TextureDescriptor, TextureUsage};
use crate::api::resource::transfer::{
    BufferUploadDescriptor, ReadbackRequest, ReadbackTicket, ReadbackViewData,
};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::ShaderLocation;

/// The portable output of one shared indirect-compute recording case.
///
/// Backends prepare only their code-form-specific pipeline and bind group. The
/// common case owns the RHI-level argument upload, `dispatch_indirect`, buffer
/// readback, and the explicit command-recording lifetime.
pub(crate) struct IndirectComputeRecording {
    /// Completed portable work ready for a compatible plan lane.
    work: Option<RecordedWork>,
    /// Ticket that observes the storage-buffer output once the plan completes.
    pub(crate) ticket: ReadbackTicket,
}

impl IndirectComputeRecording {
    /// Moves the single recorded workload into its submission plan.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("one indirect-compute recording may be submitted only once")
    }
}

/// Portable recording for the deterministic empty-occlusion-query case.
///
/// An empty raster interval is intentional: its resolved `u64` must be zero
/// on every backend, while still requiring native begin/end, query-set hazard,
/// resolve, transfer, completion, and readback lowering.  Fixtures provide a
/// device and a compatible lane only; they must not reimplement this workload.
pub(crate) struct EmptyOcclusionQueryRecording {
    /// Work containing raster query begin/end, query resolve, and readback.
    work: Option<RecordedWork>,
    /// Readback of the one portable `u64` occlusion result.
    pub(crate) ticket: ReadbackTicket,
}

impl EmptyOcclusionQueryRecording {
    /// Moves the single recorded workload into its submission plan.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("one occlusion-query recording may be submitted only once")
    }
}

/// Records one empty raster occlusion-query round trip using only portable RHI
/// commands. Callers submit it on a lane that supports raster and copy work.
pub(crate) fn record_empty_occlusion_query(
    device: &Device,
    label: &'static str,
) -> EmptyOcclusionQueryRecording {
    use crate::api::platform::OptionalFeature;

    assert!(
        device
            .capabilities()
            .supports_feature(OptionalFeature::OcclusionQuery),
        "{label}: backend published no occlusion-query capability"
    );
    assert!(
        device
            .capabilities()
            .supports_feature(OptionalFeature::QueryResolve),
        "{label}: backend published occlusion without the required resolve route"
    );

    let set = device
        .create_query_set(&QuerySetDescriptor::new(QueryType::Occlusion, 1))
        .unwrap_or_else(|error| panic!("{label}: published query set creation failed: {error}"));
    let target = device
        .create_texture(&TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        ))
        .unwrap_or_else(|error| panic!("{label}: color attachment creation failed: {error}"));
    let color_view = device
        .create_texture_view(
            &target,
            &TextureViewDescriptor::whole(&target, TextureViewDimension::D2)
                .expect("whole portable 2D color view"),
        )
        .unwrap_or_else(|error| panic!("{label}: color attachment view creation failed: {error}"));
    let destination = device
        .create_buffer(&BufferDescriptor::new(
            8,
            BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
        ))
        .unwrap_or_else(|error| {
            panic!("{label}: query resolve destination creation failed: {error}")
        });
    let scope = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment {
            depth_slice: None,
            view: ColorAttachmentView::Texture(color_view),
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
        raster
            .begin_query(&set, 0)
            .unwrap_or_else(|error| panic!("{label}: query begin failed: {error}"));
        raster
            .end_query(&set, 0)
            .unwrap_or_else(|error| panic!("{label}: query end failed: {error}"));
        raster
            .end()
            .unwrap_or_else(|error| panic!("{label}: raster scope close failed: {error}"));
    }
    recorder
        .resolve_query_set(&set, 0, 1, &destination, 0)
        .unwrap_or_else(|error| panic!("{label}: query resolve recording failed: {error}"));
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some(format!("{label} result"))),
            src: destination,
            range: BufferRange::new(0, 8),
        })
        .unwrap_or_else(|error| panic!("{label}: query result readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    EmptyOcclusionQueryRecording {
        work: Some(work),
        ticket,
    }
}

/// Asserts the portable `u64` result layout for an empty occlusion interval.
pub(crate) async fn assert_empty_occlusion_result(ticket: &ReadbackTicket, label: &str) {
    // Exercise the public async mapping lease instead of observing backend
    // readiness through `try_read`; dropping this view closes native mappings.
    let view = ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: query readback failed: {error}"));
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("{label}: query resolve returned a texture readback");
    };
    assert_eq!(
        bytes.len(),
        8,
        "{label}: one occlusion slot must be one u64"
    );
    assert_eq!(
        u64::from_le_bytes(bytes.try_into().expect("exact portable u64 result")),
        0,
        "{label}: an empty raster interval cannot produce an occlusion sample"
    );
}

/// Records the common `upload [1, 1, 1] -> indirect compute -> readback` case.
///
/// This is deliberately one common sequence, so a result difference between
/// DX12, Vulkan, Metal, WebGPU, or GL is evidence about lowering rather than a
/// subtly different test program. The backend-specific fixture has already
/// created `pipeline`, `group`, and `output` through the public API.
pub(crate) fn record_single_indirect_compute(
    device: &Device,
    pipeline: &ComputePipeline,
    group: &BindGroup,
    output: Buffer,
    output_size: u64,
    label: &'static str,
) -> IndirectComputeRecording {
    assert!(
        device
            .capabilities()
            .supports_feature(crate::api::platform::OptionalFeature::IndirectDispatch),
        "{label}: backend published no indirect-dispatch capability"
    );
    let arguments = device
        .create_buffer(&BufferDescriptor::new(
            12,
            BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
        ))
        .unwrap_or_else(|error| {
            panic!("{label}: indirect argument buffer creation failed: {error}")
        });
    let upload = device
        .create_buffer_upload(BufferUploadDescriptor {
            label: Label(Some(format!("{label} arguments"))),
            dst: arguments.clone(),
            dst_offset: 0,
            bytes: Arc::from([1u32, 1, 1].map(u32::to_le_bytes).concat()),
        })
        .unwrap_or_else(|error| {
            panic!("{label}: indirect argument upload creation failed: {error}")
        });
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    recorder
        .encode_upload(&upload)
        .unwrap_or_else(|error| panic!("{label}: argument upload recording failed: {error}"));
    {
        let mut compute = recorder
            .begin_compute(&ComputeScopeDescriptor::new())
            .unwrap_or_else(|error| panic!("{label}: compute scope failed: {error}"));
        compute
            .set_pipeline(pipeline)
            .unwrap_or_else(|error| panic!("{label}: pipeline binding failed: {error}"));
        compute
            .set_bind_group(BindGroupIndex::new(0), group, &[])
            .unwrap_or_else(|error| panic!("{label}: bind-group binding failed: {error}"));
        compute
            .dispatch_indirect(&arguments, 0)
            .unwrap_or_else(|error| panic!("{label}: indirect dispatch recording failed: {error}"));
        compute
            .end()
            .unwrap_or_else(|error| panic!("{label}: compute scope close failed: {error}"));
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some(format!("{label} output"))),
            src: output,
            range: BufferRange::new(0, output_size),
        })
        .unwrap_or_else(|error| panic!("{label}: output readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    IndirectComputeRecording {
        work: Some(work),
        ticket,
    }
}

/// Reads one completed buffer ticket as little-endian `u32` words and compares
/// it with the fixture's backend-code-form-specific expected result.
pub(crate) async fn assert_readback_u32_words(
    ticket: &ReadbackTicket,
    expected: &[u32],
    label: &str,
) {
    let view = ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: readback failed: {error}"));
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("{label}: buffer readback returned texture data");
    };
    let actual = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("one u32")))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "{label}: indirect compute output differs");
}
