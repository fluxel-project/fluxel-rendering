//! Metal occlusion-query storage.
//!
//! An occlusion slot is the RHI's fixed one `u64` result word.  Metal writes
//! visibility results into a shared `MTLBuffer`; command lowering will select
//! it on the render-pass descriptor and resolve its byte range.  Timestamp and
//! pipeline-statistics sets intentionally remain refused here: their native
//! counter APIs require separate availability probes and lowering paths.

use std::any::Any;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::query::{QuerySetDescriptor, QueryType};
use crate::api::resource::backend::QuerySetBackend;

/// Native visibility-result storage behind an occlusion `QuerySet`.
pub(super) struct MetalQuerySet {
    /// Shared because Metal's visibility-result buffer is written by the GPU
    /// and resolved/read back through the normal RHI transfer path.
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) count: u32,
}

// Like buffers and textures, query storage is immutable as a Rust object after
// creation; command encoder mutation is serialized by the device command spine.
unsafe impl Send for MetalQuerySet {}
unsafe impl Sync for MetalQuerySet {}

impl QuerySetBackend for MetalQuerySet {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) fn create_query_set(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &QuerySetDescriptor,
) -> RhiResult<MetalQuerySet> {
    if !matches!(descriptor.ty, QueryType::Occlusion) {
        let what = match descriptor.ty {
            QueryType::Timestamp => return timestamp_unavailable(),
            QueryType::PipelineStatistics(_) => "pipeline-statistics query sets",
            QueryType::Occlusion => unreachable!(),
        };
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!("Metal lowering is unavailable for {what}"),
        )
        .at("MetalDevice::create_query_set"));
    }
    const OCCLUSION_SLOT_BYTES: u64 = 8;
    let bytes = u64::from(descriptor.count)
        .checked_mul(OCCLUSION_SLOT_BYTES)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "occlusion query storage size overflows",
            )
            .at("MetalDevice::create_query_set")
        })?;
    let bytes = usize::try_from(bytes).map_err(|_| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "occlusion query storage exceeds host address space",
        )
        .at("MetalDevice::create_query_set")
    })?;
    let raw = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::OutOfMemory,
                "Metal visibility-result buffer allocation failed",
            )
            .at("MetalDevice::create_query_set")
        })?;
    if let Some(label) = descriptor.label.as_deref() {
        raw.setLabel(Some(&NSString::from_str(label)));
    }
    Ok(MetalQuerySet {
        raw,
        count: descriptor.count,
    })
}

/// Refuses timestamp creation before native allocation.
///
/// Metal has `sampleTimestamps:gpuTimestamp:`, which correlates CPU and GPU
/// clocks but does not report a stable GPU tick period. The RHI's timestamp
/// capability requires exact `period_nanos`, so this is not a substitute.
/// `wgpu-hal` 30.0.1 describes its Intel=83.333ns / otherwise=1ns choice as
/// "dangerous" knowledge (`metal/adapter.rs:81-108`); Fluxel does not publish
/// that heuristic as a device fact. The current objc2-metal dependency also
/// intentionally does not enable its `MTLCounters` feature, preventing an
/// accidental counter-buffer implementation without that missing contract.
fn timestamp_unavailable() -> RhiResult<MetalQuerySet> {
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        "Metal exposes CPU/GPU timestamp correlation but no stable GPU tick period; Fluxel refuses to guess period_nanos",
    )
    .at("MetalDevice::create_query_set"))
}
