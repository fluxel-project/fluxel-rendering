//! Lowering a GL-family discovery snapshot onto the common capability contract.
//!
//! The common contract's read of a device is a `DeviceCapabilities` value the
//! graph compiler validates against, and its builder starts **fail-closed**
//! (`rendergraph/src/rhi.rs:262-279`): immediate-context recording, no queues, no
//! formats, backend-managed transitions, single-queue ordering, no timestamps, no
//! transient reuse, zero colour attachments and zero compute-workgroup counts.
//! This module fills that description in from one discovery snapshot, and the rule
//! it obeys is the builder's own: **a fact is reported only where the snapshot
//! proves it, and every field left alone keeps the value that rejects work.**  A
//! GL-family context is not lowered by widening the default and then trimming; it
//! is lowered by adding exactly what was observed.
//!
//! The lowering is total.  It cannot fail, and it has nothing to report on
//! failure: an absent fact and an unproved fact are the same thing to the
//! compiler, so a capability description that could error would only move that
//! sameness into a second place.
//!
//! Three fields are more than a copy, and each is a decision worth reading:
//!
//! - **The compute workgroup count is reported only where dispatch was proved.**
//!   `GlLimits::max_compute_work_group_count` is queried on every accepted
//!   profile and is non-zero on a desktop context that has no compute, so copying
//!   it unconditionally would report a dispatch dimension for a context that
//!   cannot dispatch.  Layer 1 already guards this shape at
//!   `GlDiscoverySnapshot::max_multiview_view_count`; this is the same rule.
//! - **A format entry folds across sample counts.**  The common contract
//!   describes a format, while the GL table is keyed by `(format, sample count)`
//!   -- so the boolean facts fold with `any` and the count-sensitive half is
//!   carried by `attachment_sample_counts`, the only field shaped for it.  The
//!   fold does not need to prefer the single-sample entry: a multisample entry
//!   records its own `sampled: false`, so an `any` fold is already honest.
//! - **There is no surface.**  `DeviceCapabilities::surface` is `Option`, and this
//!   adapter has none until the presentation slice attaches one.  Its absence is
//!   a fact about this slice and not an omission, and it is the same reason the
//!   queue reports `present: false`.
//!
//! What is deliberately *not* claimed is `transient_resources`.  Those three rows
//! describe the lowering the compiler may assume for graph transients, and this
//! adapter has no transient-creation path until the raster slice exists; asserting
//! reuse before there is anything to reuse would be asserting a property of code
//! that does not exist.

use fluxel_rendergraph::{
    BufferCapabilities, DeviceCapabilities, DeviceLimits, QueueCapabilities, QueueDescriptor,
    QueueId, RecordingCapabilities, RecordingModel, SynchronizationCapabilities, TextureFormat,
    TextureFormatCapabilities, TimestampCapabilities, TransientResourceCapabilities,
    TransitionCapabilities,
};

use crate::webgl2::api::{
    GlCapability, GlDiscoverySnapshot, GlFormat, GlFormatCapabilities, GlFormatResourceKind,
    GlLimits,
};

/// Lowers one discovery snapshot onto the common capability contract.
pub(crate) fn capabilities(snapshot: &GlDiscoverySnapshot) -> DeviceCapabilities {
    let proved = snapshot.capabilities();
    let compute = proved.supports(GlCapability::Compute);
    let storage = proved.supports(GlCapability::StorageBuffer);
    // One indirect command reads its parameters from a buffer, so any of the
    // three indirect domains proves the buffer half.  They are separate
    // capabilities because they are separate command paths, not because they
    // read different buffers.
    let indirect = proved.supports(GlCapability::IndirectDraw)
        || proved.supports(GlCapability::IndirectDispatch)
        || proved.supports(GlCapability::MultiDrawIndirect);

    let mut builder = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, compute, true, false),
        ))
        // Set rather than inherited, even though both equal the builder's
        // default: the default is fail-closed by choice, and a value that happens
        // to agree with it today is not the same as a value read off the GL
        // context.  One context is one immediate recorder on one owning thread
        // (`OwnerThreadIdentity`), so there are no parallel encoders.
        .recording(RecordingCapabilities::new(
            RecordingModel::ImmediateContext,
            false,
        ))
        // Layer 2 owns the mirror that makes a redundant transition free, and
        // Layer 1 exposes an explicit memory barrier only for the optional
        // compute path.  There is no graph-lowered barrier to hand the compiler.
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        // Only the pass-boundary timestamp row is expressible in the common
        // contract, and placing one needs the pass identity this slice does not
        // have.  A timer query's counter width is not that fact.
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(limits(&snapshot.limits(), compute))
        .buffers(BufferCapabilities::new(storage, storage, indirect));

    for entry in texture_format_entries(snapshot) {
        builder = builder.texture_format(entry);
    }
    builder.build()
}

/// The limits graph validation reads, lowered from the queried GL limits.
fn limits(queried: &GlLimits, compute: bool) -> DeviceLimits {
    // A pass may use as many colour attachments as the smaller of the two queried
    // counts allows: one is the framebuffer's limit, the other the draw state's,
    // and neither is usable on its own.
    let max_color_attachments = queried.max_color_attachments.min(queried.max_draw_buffers);
    let limits = DeviceLimits::new(
        max_color_attachments,
        queried.uniform_buffer_offset_alignment,
    );
    if compute {
        limits.with_max_compute_workgroups_per_dimension(queried.max_compute_work_group_count)
    } else {
        // Left at the fail-closed zero on every dimension, which is what rejects
        // a dispatch on a context that never proved one.
        limits
    }
}

/// The per-format entries, folded from the GL table's `(format, count)` rows.
///
/// The output order is the GL table's own, which is stable `(format, resource
/// kind, count)` order, so the same snapshot always produces the same capability
/// value -- which matters because the value is comparable and the graph caches
/// compilation on it.
fn texture_format_entries(snapshot: &GlDiscoverySnapshot) -> Vec<TextureFormatCapabilities> {
    let mut folded: Vec<(TextureFormat, FormatFacts)> = Vec::new();
    for observed in snapshot.formats().iter() {
        // Renderbuffer facts are real but describe a different resource
        // contract: the common contract's format row is about textures, and the
        // GL table splits the two because WebGL2 exposes one without the other.
        if observed.resource_kind != GlFormatResourceKind::Texture {
            continue;
        }
        // The common contract has no name for every GL format, and an unnamed
        // format has no row to make a claim in.  Unnamed is unsupported.
        let Some(format) = common_format(observed.format) else {
            continue;
        };
        match folded.iter_mut().find(|(known, _)| *known == format) {
            Some((_, facts)) => facts.absorb(&observed),
            None => {
                let mut facts = FormatFacts::default();
                facts.absorb(&observed);
                folded.push((format, facts));
            }
        }
    }
    folded
        .into_iter()
        .map(|(format, facts)| facts.into_capabilities(format))
        .collect()
}

/// The common contract's name for a GL format, where it has one.
///
/// Four of the contract's five formats have a GL counterpart here; `Bgra8Unorm`
/// does not, because no GL-family profile this contract models has a BGRA texture
/// format in its table.  Returning `None` is the lowering's whole answer for it.
pub(super) fn common_format(format: GlFormat) -> Option<TextureFormat> {
    match format {
        GlFormat::Rgba8Unorm => Some(TextureFormat::Rgba8Unorm),
        GlFormat::Rgba8Srgb => Some(TextureFormat::Rgba8UnormSrgb),
        GlFormat::Rgba16Float => Some(TextureFormat::Rgba16Float),
        GlFormat::Depth32Float => Some(TextureFormat::Depth32Float),
        _ => None,
    }
}

/// Whether a GL format is an attachment's depth/stencil side rather than its
/// colour side.
///
/// The GL table's `renderable` fact does not distinguish the two sides, so the
/// side has to come from the format.  The three variants are the same three Layer
/// 1 classifies by when it checks a sample count against
/// `max_depth_texture_samples` (`api/formats.rs:530-533`), and the wildcard is
/// deliberate for the same reason it is there: this list is the accepted
/// profiles' depth-format set, and a fourth entry would be a profile change
/// rather than a lowering change.
pub(super) fn is_depth_stencil(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Depth16Unorm | GlFormat::Depth24PlusStencil8 | GlFormat::Depth32Float
    )
}

/// The folded boolean facts of one common texture format.
#[derive(Default)]
struct FormatFacts {
    sampled: bool,
    filterable: bool,
    storage_read: bool,
    storage_write: bool,
    color_attachment: bool,
    depth_stencil_attachment: bool,
    attachment_sample_counts: Vec<u32>,
    copy_source: bool,
    copy_destination: bool,
}

impl FormatFacts {
    /// Folds one observed `(format, sample count)` row into the format's facts.
    ///
    /// The storage half is read from the row without re-checking its evidence:
    /// the table refuses to record a storage fact that was not operation-probed
    /// (`GlFormatTableError::StorageRequiresOperationProbe`), so a check here
    /// would be a second definition of a rule the table already owns.
    fn absorb(&mut self, observed: &GlFormatCapabilities) {
        self.sampled |= observed.sampled;
        self.filterable |= observed.filterable;
        self.storage_read |= observed.storage_read;
        self.storage_write |= observed.storage_write;
        self.copy_source |= observed.copy_source;
        self.copy_destination |= observed.copy_destination;
        if observed.renderable {
            self.attachment_sample_counts.push(observed.sample_count);
            if is_depth_stencil(observed.format) {
                self.depth_stencil_attachment = true;
            } else {
                self.color_attachment = true;
            }
        }
    }

    fn into_capabilities(mut self, format: TextureFormat) -> TextureFormatCapabilities {
        // Ascending and deduplicated: the GL table is keyed by count, so the fold
        // already sees them in order, but the list is a set of legal counts and is
        // made one here rather than relied on to have arrived as one.
        self.attachment_sample_counts.sort_unstable();
        self.attachment_sample_counts.dedup();
        TextureFormatCapabilities::builder(format)
            .sampled(self.sampled, self.filterable)
            .storage(self.storage_read, self.storage_write)
            .attachments(
                self.color_attachment,
                self.depth_stencil_attachment,
                self.attachment_sample_counts,
            )
            .copies(self.copy_source, self.copy_destination)
            .build()
    }
}

#[cfg(test)]
mod tests;
