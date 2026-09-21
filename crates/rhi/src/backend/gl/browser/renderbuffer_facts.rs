//! Browser renderbuffer storage evidence and allocation admission.
//!
//! A renderbuffer fact is recorded only from a real allocation: the probe
//! creates a scratch renderbuffer, gives it storage, binds it as a scratch
//! framebuffer attachment and reads completeness back. That scratch bind is
//! this module's own probe and says nothing about the contract: the shared
//! framebuffer contract does carry a renderbuffer attachment target, so a
//! render pass may attach one exactly as it attaches a texture.
//!
//! What enters the table is what the driver answered: a completed probe records
//! its answer, a refused one records `renderable: false` so a later reader can
//! tell "refused here" from "never asked", and a probe that could not run
//! records nothing at all, which is what makes an allocation fail closed
//! instead of being attempted and hoped for. Sample counts above the discovered
//! ceiling are dropped while recording, so the recorded set is a subset of the
//! set the table's own limit validation would accept.
//!
//! This module does not own the renderbuffer object lifetime (that is
//! `provider.rs`) and does not change where a texture view attaches (that is
//! `format_map.rs`).

use js_sys::Int32Array;
use wasm_bindgen::JsCast;
use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::{
    GlError, GlFormat, GlFormatCapabilities, GlFormatEvidence, GlFormatResourceKind, GlFormatTable,
    GlLimits, GlRenderBufferDesc,
};
use super::format_map;

/// Edge length of the scratch storage every probe allocates.
///
/// Four texels is the smallest extent that exercises a real allocation and a
/// real attachment on any WebGL2 implementation, and it keeps a probe set that
/// runs once per context negligible.
const PROBE_EXTENT: i32 = 4;

/// One renderbuffer-capable format and the internal constant it allocates as.
struct RenderbufferFormat {
    format: GlFormat,
    internal: u32,
}

/// The formats this backend records renderbuffer facts for.
///
/// It is deliberately the same set the native provider records renderbuffer
/// facts for, so one descriptor cannot name a renderbuffer class on one
/// provider and an unallocatable one on the other. Compressed formats are
/// absent for the same reason they can never be renderbuffers at all.
const RENDERBUFFER_FORMATS: [RenderbufferFormat; 3] = [
    RenderbufferFormat {
        format: GlFormat::Rgba8Unorm,
        internal: Gl::RGBA8,
    },
    RenderbufferFormat {
        format: GlFormat::Rgba8Srgb,
        internal: Gl::SRGB8_ALPHA8,
    },
    RenderbufferFormat {
        format: GlFormat::Depth32Float,
        internal: Gl::DEPTH_COMPONENT32F,
    },
];

/// Why a renderbuffer descriptor cannot be allocated on this context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RenderbufferRejection {
    /// The extent exceeds the discovered renderbuffer ceiling.
    ExtentExceedsLimit,
    /// The sample count exceeds the discovered multisample ceiling.
    SampleCountExceedsLimit,
    /// No exact fact exists for this format at this sample count here.
    NoFormatFact,
    /// The exact fact exists and says this format is not renderable here.
    NotRenderable,
    /// The format has no renderbuffer storage mapping in this backend.
    NoStorageMapping,
}

impl RenderbufferRejection {
    /// The structured provider error for one rejection.
    pub(super) fn error(self, operation: &'static str) -> GlError {
        match self {
            Self::ExtentExceedsLimit => GlError::Validation {
                operation,
                message: "renderbuffer extent exceeds the discovered limit".into(),
            },
            Self::SampleCountExceedsLimit => GlError::Validation {
                operation,
                message: "renderbuffer sample count exceeds the discovered limit".into(),
            },
            // A missing fact is not bad input; it is this context having no
            // evidence that the allocation exists, so it stays `Unsupported`
            // rather than being reported as a caller mistake.
            Self::NoFormatFact => GlError::Unsupported {
                operation,
                reason: "no exact renderbuffer format fact for this context",
            },
            Self::NotRenderable => GlError::Unsupported {
                operation,
                reason: "format lacks renderable evidence at this sample count",
            },
            Self::NoStorageMapping => GlError::Unsupported {
                operation,
                reason: "format has no proven renderbuffer storage mapping",
            },
        }
    }
}

/// Decides one renderbuffer allocation from recorded facts only.
///
/// Pure by construction: it reads the discovery snapshot and the descriptor and
/// touches no browser object, so the accept/reject matrix is testable without a
/// live context. Returns the internal constant to allocate with.
pub(super) fn admit(
    limits: &GlLimits,
    formats: &GlFormatTable,
    desc: GlRenderBufferDesc,
) -> Result<u32, RenderbufferRejection> {
    if desc.width > limits.max_renderbuffer_size || desc.height > limits.max_renderbuffer_size {
        return Err(RenderbufferRejection::ExtentExceedsLimit);
    }
    if desc.samples > limits.max_samples {
        return Err(RenderbufferRejection::SampleCountExceedsLimit);
    }
    let facts = formats
        .get_for(
            GlFormatResourceKind::Renderbuffer,
            desc.format,
            desc.samples,
        )
        .ok_or(RenderbufferRejection::NoFormatFact)?;
    if !facts.renderable {
        return Err(RenderbufferRejection::NotRenderable);
    }
    storage_mapping(desc.format).ok_or(RenderbufferRejection::NoStorageMapping)
}

/// The internal constant this format allocates a renderbuffer as.
fn storage_mapping(format: GlFormat) -> Option<u32> {
    RENDERBUFFER_FORMATS
        .iter()
        .find(|entry| entry.format == format)
        .map(|entry| entry.internal)
}

/// Records a renderbuffer fact for every format/count pair this context answered.
pub(super) fn record_facts(
    raw: &Gl,
    limits: &GlLimits,
    formats: &mut GlFormatTable,
) -> Result<(), GlError> {
    for entry in &RENDERBUFFER_FORMATS {
        let mut counts = vec![1];
        if let Some(answered) = supported_sample_counts(raw, entry.internal) {
            counts.extend(
                answered
                    .into_iter()
                    .filter(|count| *count > 1 && *count <= limits.max_samples),
            );
        }
        counts.sort_unstable();
        counts.dedup();
        for sample_count in counts {
            let answer = probe(raw, entry, sample_count);
            let Some(renderable) = answer else {
                continue;
            };
            formats
                .record(GlFormatCapabilities {
                    format: entry.format,
                    resource_kind: GlFormatResourceKind::Renderbuffer,
                    sample_count,
                    evidence: GlFormatEvidence::OperationProbed,
                    // A renderbuffer is never sampled and never filtered: it is
                    // written by rendering and read by a framebuffer operation.
                    // The copy fields stay false for the same reason, because a
                    // shared copy word describes a texture transfer.
                    sampled: false,
                    filterable: false,
                    renderable,
                    blendable: renderable && !is_depth_format(entry.format),
                    storage_read: false,
                    storage_write: false,
                    copy_source: false,
                    copy_destination: false,
                })
                .map_err(|error| GlError::Driver {
                    operation: "record WebGL2 renderbuffer format",
                    message: format!("{error:?}"),
                })?;
        }
    }
    Ok(())
}

const fn is_depth_format(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Depth16Unorm | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32Float
    )
}

/// The sample counts the driver itself reports for one internal format.
///
/// `SAMPLES` is the registry parameter that answers exactly this question for a
/// renderbuffer target, so the answer is a recorded fact from the driver rather
/// than a portable guess. An answer that is not an integer array means the
/// question was not answered, which is reported as no evidence instead of an
/// empty set of counts.
fn supported_sample_counts(raw: &Gl, internal: u32) -> Option<Vec<u32>> {
    let value = raw
        .get_internalformat_parameter(Gl::RENDERBUFFER, internal, Gl::SAMPLES)
        .ok()?;
    let answered = value.dyn_into::<Int32Array>().ok()?;
    let counts = (0..answered.length())
        .map(|index| answered.get_index(index))
        .filter(|count| *count > 0)
        .map(|count| count as u32)
        .collect();
    Some(counts)
}

/// Allocates real storage, binds it as an attachment, and reads completeness.
///
/// `None` means the probe could not run — a browser exception or a driver error
/// while probing — and never that rendering is unsupported; `Some(false)` is a
/// real negative observation and is recorded as one. Both scratch objects are
/// released on every path, and the driver error this probe may generate stays
/// consumed here so it cannot be reported later as another call's failure.
fn probe(raw: &Gl, entry: &RenderbufferFormat, sample_count: u32) -> Option<bool> {
    let renderbuffer = raw.create_renderbuffer()?;
    raw.bind_renderbuffer(Gl::RENDERBUFFER, Some(&renderbuffer));
    if sample_count > 1 {
        raw.renderbuffer_storage_multisample(
            Gl::RENDERBUFFER,
            sample_count as i32,
            entry.internal,
            PROBE_EXTENT,
            PROBE_EXTENT,
        );
    } else {
        raw.renderbuffer_storage(Gl::RENDERBUFFER, entry.internal, PROBE_EXTENT, PROBE_EXTENT);
    }
    let answer = (raw.get_error() == Gl::NO_ERROR).then(|| {
        let framebuffer = raw.create_framebuffer()?;
        raw.bind_framebuffer(Gl::FRAMEBUFFER, Some(&framebuffer));
        // The depth formats attach where their texture siblings attach, so one
        // mapping decides for both and a renderbuffer can never be attached to
        // a point its format cannot serve.
        let attachment =
            format_map::depth_attachment_point(entry.format).unwrap_or(Gl::COLOR_ATTACHMENT0);
        raw.framebuffer_renderbuffer(
            Gl::FRAMEBUFFER,
            attachment,
            Gl::RENDERBUFFER,
            Some(&renderbuffer),
        );
        let status = raw.check_framebuffer_status(Gl::FRAMEBUFFER);
        let errored = raw.get_error() != Gl::NO_ERROR;
        raw.bind_framebuffer(Gl::FRAMEBUFFER, None);
        raw.delete_framebuffer(Some(&framebuffer));
        (!errored).then_some(status == Gl::FRAMEBUFFER_COMPLETE)
    });
    raw.bind_renderbuffer(Gl::RENDERBUFFER, None);
    raw.delete_renderbuffer(Some(&renderbuffer));
    let _ = raw.get_error();
    answer.flatten()
}
