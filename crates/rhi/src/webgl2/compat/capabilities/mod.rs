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
//! Four fields are more than a copy, and each is a decision worth reading:
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
//! - **There is a surface where the drawable was read, and no surface where it
//!   was not.**  `DeviceCapabilities::surface` is `Option`, and the lowering
//!   answers it from the typed half of the drawable observation.  The format list
//!   has exactly one entry, the one the observed component widths name, and a
//!   drawable whose widths name no format this contract has leaves the field
//!   `None` -- which is not a partial answer, because the graph refuses a present
//!   root without a format list and refuses every use of a surface resource
//!   without the field (`compile/validation/roots.rs:134`,
//!   `compile/validation/capabilities.rs:263`).  The queue's `present` row follows
//!   the same fact rather than the other way round: a queue that presents frames
//!   the device cannot describe would be a row no present root could ever reach.
//! - **The buffer row is two decisions rather than a copy, and the indirect half
//!   of it has no consumer.**  The snapshot's one storage fact carries no
//!   direction, so it is reported in both, and the indirect flag is the union of
//!   three separate command-path rows.  What that union *authorizes* is the part
//!   worth reading: `ExecutionBackend` declares no indirect verb and neither
//!   command sink has one, so nothing in this adapter can issue an indirect draw
//!   or dispatch, and the flag authorizes a pass's *declaration* that a buffer is
//!   read as an indirect command source -- the resource role and the hazard --
//!   rather than a command.  The refusal a caller meets is therefore the
//!   compiler's, at `rendergraph/src/compile/validation/capabilities.rs`, and it
//!   arrives exactly where this flag is false.  A context that proved indirect
//!   really does permit the role, so answering false would misdescribe the
//!   context to fix a gap in the layer above it; the ledger's indirect row has no
//!   consumer in this contract, which is a statement about the contract.  Layer
//!   2's `GlOptionalIndirectBackend` (`webgl2/state/backend.rs`) is the seam the
//!   row is waiting for, and no entry point is bounded on it yet.
//!
//! What is deliberately *not* claimed is `transient_resources`.  Those three rows
//! are reuse facts -- pooling, in-frame reuse and aliasing -- and this adapter
//! answers all three no: a graph transient is one Layer 1 object created for the
//! request that asked for it (`create_transient_texture` /
//! `create_transient_buffer`), retained until its last lease drops and destroyed
//! then.  Nothing here pools an object across frames, hands back one a previous
//! frame used, or lets two of them share an allocation, so every row keeps the
//! value that rejects the lowering.

use fluxel_rendergraph::{
    BufferCapabilities, DeviceCapabilities, DeviceLimits, QueueCapabilities, QueueDescriptor,
    QueueId, RecordingCapabilities, RecordingModel, SurfaceCapabilities,
    SynchronizationCapabilities, TextureFormat, TextureFormatCapabilities, TimestampCapabilities,
    TransientResourceCapabilities, TransitionCapabilities,
};

use crate::webgl2::api::{
    GlCapability, GlDiscoverySnapshot, GlFormat, GlFormatCapabilities, GlFormatResourceKind,
    GlLimits, GlSurfaceFacts,
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
    //
    // The third term is unreachable through the shipped providers today, and is
    // kept anyway.  `ProbeReport::multi_draw_indirect` is
    // `ProbeAnswer::Unavailable` at its only construction site, with the reason
    // recorded there -- glow 0.18 binds no entry point for it -- and
    // `passed_probes_enable_core_proved_capabilities_on_desktop_46` asserts the
    // row stays disabled even when every probe passes.  This union reads the
    // *ledger*, and the ledger has the row, so a provider that one day answers
    // it should not need this line changed as well.  No test asserts the term,
    // because the state it would assert is one no provider can produce; the test
    // that covers this function's live rows is
    // `a_proved_indirect_command_row_reports_indirect_buffer_reads`.
    let indirect = proved.supports(GlCapability::IndirectDraw)
        || proved.supports(GlCapability::IndirectDispatch)
        || proved.supports(GlCapability::MultiDrawIndirect);

    let surface = surface(snapshot.surface_facts());

    let mut builder = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, compute, true, surface.is_some()),
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
    match surface {
        Some(surface) => builder.surface(surface).build(),
        None => builder.build(),
    }
}

/// The drawable's presentation facts, where the provider observed them.
///
/// `None` is the fail-closed answer, and it is what this returns unless the
/// drawable was actually read.  A surface is a claim about the default
/// framebuffer, and this contract's read of one is its format list; a device that
/// reported a surface with an empty list would be a device whose every present
/// root fails one check later, in the compiler, with a reason that blames the
/// graph.
///
/// # Why one format, and why only this one
///
/// The widths are the whole of the evidence, so the claim is exactly the format
/// they name: four eight-bit channels.  Nothing wider or narrower is claimed,
/// because a drawable this contract has no name for is a drawable a graph cannot
/// compile a present root against, and the outcome of claiming the nearest format
/// instead would be an acquired image whose descriptor and whose pixels disagree
/// -- the extent is the caller's and the format is the graph's, so a mismatch
/// here is not a driver's to resolve.
///
/// `Rgba8UnormSrgb` is deliberately not claimed even though a browser drawable
/// usually is sRGB: no accepted profile exposes the drawable's encoding, and a
/// graph compiled against `sRGB` would be told to write encoded values into a
/// texture the adapter creates without encoding.  That is a double-gamma error,
/// and an unclaimed format is a refused present root -- the two mistakes are not
/// the same size.  `Rgba16Float` is not claimed for the same class of reason:
/// component widths alone do not separate a float component from a
/// normalized-integer one, so a 16/16/16/16 drawable is not named by them.
///
/// The two operations are claimed together and unconditionally, because neither
/// is a fact about the drawable: the acquired image is created through the same
/// resource path as every other texture this adapter owns, in the same
/// attachment table and with the same usage lowering
/// (`compat/device/surface.rs::acquire_surface_texture`), so the two semantics
/// the graph permits on a surface resource -- a colour attachment and a copy
/// destination -- are served by machinery that was not built for the surface at
/// all.
fn surface(facts: GlSurfaceFacts) -> Option<SurfaceCapabilities> {
    let GlSurfaceFacts::Observed { color_bits } = facts else {
        return None;
    };
    let formats = match color_bits {
        [8, 8, 8, 8] => vec![TextureFormat::Rgba8Unorm],
        _ => return None,
    };
    Some(SurfaceCapabilities::new(formats, true, true))
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
