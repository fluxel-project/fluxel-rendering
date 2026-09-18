//! Step 11's lowering half: the discovered facts onto the common capability
//! contract.
//!
//! The graph compiler reads one `fluxel_rendergraph::DeviceCapabilities` value, and
//! its builder starts fail-closed (`rendergraph/src/rhi.rs`): immediate-context
//! recording, no queues, no formats, backend-managed transitions, single-queue
//! ordering, no timestamps, no transient reuse, zero colour attachments and zero
//! compute-workgroup counts. [`capabilities`] fills that description in from the
//! three discovery results this backend already owns, and it obeys the builder's
//! own rule: **a fact is reported only where discovery proved it, and every field
//! left alone keeps the value that rejects work.**
//!
//! The lowering is total, exactly as the GL family's is. It cannot fail, and it has
//! nothing to report on failure: an absent fact and an unproved fact are the same
//! thing to the compiler, so a capability description that could error would only
//! move that sameness into a second place.
//!
//! # The ledger is the capability input, not a second copy of the facts
//!
//! The queue-family rows, the optional command rows, the buffer-storage row and
//! the timestamp row are read from the same [`CapabilityLedger`] that `require`
//! negotiates from, rather than re-derived from the queue family and the limits.
//! That is what keeps one discovery answer: a row this backend has not recorded is
//! a domain nothing has proved, and the lowering reports the rejecting value for it
//! automatically -- and starts reporting the fact the moment the step that proves
//! the row records it.
//!
//! # The queue's flags are the ledger's answer, not a second opinion
//!
//! `raster`, `compute` and `copy` are the `Graphics`, `Compute` and `Copy` rows.
//! The queue physically exists because a device was created on a graphics family,
//! but what it may execute is the ledger's answer, so a row nothing proved leaves
//! its flag false instead of being assumed from the queue's existence. `copy` in
//! particular used to be a constant written here, which made the lowering claim a
//! capability the ledger did not carry; it is now the same kind of read as the
//! other two.
//!
//! # Four fields are decisions rather than copies
//!
//! - **Recording is deferred command buffers.** This backend records into a
//!   `VkCommandBuffer` (step 7) and submits it (step 9), which is
//!   [`RecordingModel::DeferredCommandBuffers`]; the GL family's immediate context
//!   is the other model. `parallel_independent_encoders` is false because the one
//!   recording encoder is sequential.
//! - **Transitions are graph-managed explicit.** [`super::barrier`] lowers the
//!   graph's semantic access states onto `vkCmdPipelineBarrier` (step 7), so a
//!   graph transition *is* a backend operation here. The GL family keeps them
//!   backend-managed because its `compat` layer mirrors state instead.
//! - **The compute workgroup count is reported only where compute was proved.** A
//!   driver reports a non-zero count on every device, so copying it unconditionally
//!   would name a dispatch dimension for a device whose ledger refused the compute
//!   row. This is the same shape the GL lowering reads out of its own limits.
//! - **Timestamps are pass boundaries, and only where the row was proved.** The
//!   `TimestampQuery` row comes from the selected family's own
//!   `timestamp_valid_bits` report (step 2), so an unproved row keeps
//!   [`TimestampCapabilities::Unsupported`] rather than naming a placement the
//!   family never reported.
//!
//! # What is deliberately not claimed
//!
//! `surface` is `None` and the queue's `present` row is false, because presentation
//! is a fact about one device/surface pair rather than about a device. The surface
//! half of step 10 reads it through a live `VkSurfaceKHR` and the fixed contract;
//! reporting one here would be a claim about a window this call never saw. The
//! surface facts join this lowering when the device owns both a surface and the
//! format table.
//!
//! `transient_resources` keeps all three rows false. `gpu-allocator` suballocates
//! device memory, but nothing in this layer pools an object across frames, reuses
//! one inside a frame, or aliases two objects over one allocation.
//!
//! `blendable` is the one per-format fact [`FormatCapabilities`] carries that
//! `TextureFormatCapabilities` has no field for. It is not lost in the fold -- it
//! simply has nowhere to be reported yet, and it stays in the evidence table for
//! the contract that first needs to ask about blending.

use fluxel_rendergraph::{
    BufferCapabilities, DeviceCapabilities, DeviceLimits, QueueCapabilities, QueueDescriptor,
    QueueId, RecordingCapabilities, RecordingModel, SynchronizationCapabilities, TextureFormat,
    TextureFormatCapabilities, TimestampCapabilities, TransientResourceCapabilities,
    TransitionCapabilities,
};

use crate::common::caps::{AdapterLimits, Capability, CapabilityLedger};
use crate::common::formats::{FormatCapabilities, FormatTable};

use super::format::{image_format, is_depth};

/// Lowers the discovered facts of one opened device onto the common contract.
///
/// The three inputs are one device's whole discovery: the ledger `require` reads,
/// the adapter's numeric facts, and the per-format evidence table step 11's other
/// half fills. Nothing here re-asks the driver, and nothing here can fail.
pub(crate) fn capabilities(
    ledger: &CapabilityLedger,
    adapter: &AdapterLimits,
    formats: &FormatTable,
) -> DeviceCapabilities {
    let raster = ledger.supports(Capability::Graphics);
    let compute = ledger.supports(Capability::Compute);
    let copy = ledger.supports(Capability::Copy);
    let storage = ledger.supports(Capability::StorageBuffer);
    // One indirect command reads its parameters from a buffer, so any of the three
    // indirect rows proves the buffer half. They are separate rows because they
    // are separate command paths, not because they read different buffers.
    let indirect = ledger.supports(Capability::IndirectDraw)
        || ledger.supports(Capability::IndirectDispatch)
        || ledger.supports(Capability::MultiDrawIndirect);
    // The one placement this backend can offer is a timestamp written at a pass
    // boundary, and only a proved row may name it.
    let timestamps = if ledger.supports(Capability::TimestampQuery) {
        TimestampCapabilities::PassBoundaries
    } else {
        TimestampCapabilities::Unsupported
    };

    let mut builder = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            // Every flag but present is the ledger's answer; present is false
            // because this is the headless lowering.
            QueueCapabilities::new(raster, compute, copy, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(timestamps)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(limits(adapter, compute))
        .buffers(BufferCapabilities::new(storage, storage, indirect));

    for entry in format_entries(formats) {
        builder = builder.texture_format(entry);
    }
    builder.build()
}

/// The limits graph validation reads, lowered from the adapter's own report.
///
/// `DeviceLimits` carries three fields, and only the alignment and colour count
/// are copies. The workgroup count is the one field gated on a proved fact, for the
/// reason the module doc states.
fn limits(adapter: &AdapterLimits, compute: bool) -> DeviceLimits {
    let limits = DeviceLimits::new(
        adapter.max_color_attachments,
        u64::from(adapter.min_uniform_buffer_offset_alignment),
    );
    if compute {
        limits.with_max_compute_workgroups_per_dimension(
            adapter.max_compute_workgroups_per_dimension,
        )
    } else {
        // Left at the fail-closed zero on every dimension, which is what rejects a
        // dispatch on a device that never proved one.
        limits
    }
}

/// The per-format entries, folded from the evidence table's `(format, count)` rows.
///
/// The output order is the table's own insertion order, which is deterministic
/// because the caller records from [`super::format::MAPPED`]. The capability value
/// is compared and cached, so its format order has to be a function of its contents
/// rather than of a hasher seed.
fn format_entries(formats: &FormatTable) -> Vec<TextureFormatCapabilities> {
    let mut folded: Vec<(TextureFormat, FormatFacts)> = Vec::new();
    for observed in formats.iter() {
        match folded
            .iter_mut()
            .find(|(known, _)| *known == observed.format)
        {
            Some((_, facts)) => facts.absorb(&observed),
            None => {
                let mut facts = FormatFacts::default();
                facts.absorb(&observed);
                folded.push((observed.format, facts));
            }
        }
    }
    folded
        .into_iter()
        .map(|(format, facts)| facts.into_capabilities(format))
        .collect()
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
    /// The boolean facts fold with `or` and the count-sensitive half is carried by
    /// the one field shaped for it. The fold does not need to prefer a single-sample
    /// row: this backend records only sample count one today, and a later
    /// multisampled row records its own facts rather than inheriting them.
    ///
    /// The storage half is read without re-checking its evidence, because the table
    /// already refuses to record a storage fact that was not operation-probed -- a
    /// check here would be a second definition of a rule the table owns.
    fn absorb(&mut self, observed: &FormatCapabilities) {
        self.sampled |= observed.sampled;
        self.filterable |= observed.filterable;
        self.storage_read |= observed.storage_read;
        self.storage_write |= observed.storage_write;
        self.copy_source |= observed.copy_source;
        self.copy_destination |= observed.copy_destination;
        if observed.renderable {
            self.attachment_sample_counts.push(observed.sample_count);
            // The evidence row's `renderable` covers both attachment kinds, because
            // `Vulkan` reports one flag for each and the portable table asks one
            // question. Which side of the attachment this format belongs to is
            // asked of the *mapped* `Vulkan` format, which keeps `format::is_depth`
            // the single source of truth for the depth fact.
            match image_format(observed.format) {
                Some(mapped) if is_depth(mapped) => self.depth_stencil_attachment = true,
                Some(_) => self.color_attachment = true,
                // A portable format this backend cannot create an image with has a
                // driver answer but no resource to attach. Unreachable from discovery
                // -- the query refuses an unmapped format before the table sees it --
                // and left as no claim rather than guessed at, the same direction
                // `image_format`'s `None` already takes.
                None => {}
            }
        }
    }

    /// Finishes one format's contract row.
    fn into_capabilities(mut self, format: TextureFormat) -> TextureFormatCapabilities {
        // Ascending and deduplicated: the list is a set of legal counts, and it is
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
mod tests {
    use super::*;
    use crate::common::caps::{CapabilityEvidence, CapabilityFact, OperationProbe};
    use crate::common::formats::FormatEvidence;

    /// A row proved the way this backend proves one: core, limits met, structural.
    fn proved() -> CapabilityFact {
        CapabilityFact {
            evidence: Some(CapabilityEvidence::Core),
            limits_satisfied: true,
            operation_probe: OperationProbe::NotRequired,
        }
    }

    fn ledger_with(rows: &[Capability]) -> CapabilityLedger {
        let mut ledger = CapabilityLedger::default();
        for row in rows {
            ledger.record(*row, proved());
        }
        ledger
    }

    fn adapter() -> AdapterLimits {
        AdapterLimits {
            max_color_attachments: 8,
            min_uniform_buffer_offset_alignment: 256,
            max_compute_workgroups_per_dimension: [65535, 65535, 65535],
            ..AdapterLimits::unavailable()
        }
    }

    /// One evidence row, with every fact off unless the test turns it on.
    fn row(
        format: TextureFormat,
        sample_count: u32,
        sampled: bool,
        renderable: bool,
    ) -> FormatCapabilities {
        FormatCapabilities {
            format,
            sample_count,
            evidence: FormatEvidence::OperationProbed,
            sampled,
            filterable: false,
            renderable,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        }
    }

    fn table(rows: &[FormatCapabilities]) -> FormatTable {
        let mut table = FormatTable::default();
        for entry in rows {
            table.record(*entry).expect("the fixture rows do not disagree");
        }
        table
    }

    fn entry(capabilities: &DeviceCapabilities, format: TextureFormat) -> TextureFormatCapabilities {
        capabilities
            .texture_formats
            .iter()
            .find(|entry| entry.format == format)
            .unwrap_or_else(|| panic!("{format:?} is reported"))
            .clone()
    }

    fn reported(capabilities: &DeviceCapabilities) -> Vec<TextureFormat> {
        capabilities
            .texture_formats
            .iter()
            .map(|entry| entry.format)
            .collect()
    }

    #[test]
    fn a_device_with_no_optional_row_reports_only_the_floor() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &FormatTable::default(),
        );

        assert_eq!(capabilities.queues.len(), 1, "one queue is created, one reported");
        let queue = capabilities.queues[0];
        assert!(queue.capabilities.raster, "the Graphics row was proved");
        assert!(
            !queue.capabilities.copy,
            "no Copy row was proved, so the queue's copy flag stays at the rejecting value"
        );
        assert!(
            !queue.capabilities.compute,
            "the ledger proved no compute, so the queue cannot execute any"
        );
        assert!(!queue.capabilities.present);
        assert_eq!(
            capabilities.timestamps,
            TimestampCapabilities::Unsupported,
            "no TimestampQuery row was proved"
        );
        assert_eq!(
            capabilities.buffers,
            BufferCapabilities::new(false, false, false),
            "no storage and no indirect row was proved"
        );
        assert_eq!(
            capabilities.limits.max_compute_workgroups_per_dimension, [0; 3],
            "the adapter reported a non-zero count, and no dispatch was ever proved"
        );
        assert_eq!(capabilities.surface, None);
    }

    #[test]
    fn a_proved_compute_row_reports_the_queue_and_the_workgroup_dimensions() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics, Capability::Compute]),
            &adapter(),
            &FormatTable::default(),
        );

        assert!(capabilities.queues[0].capabilities.compute);
        assert_eq!(
            capabilities.limits.max_compute_workgroups_per_dimension,
            adapter().max_compute_workgroups_per_dimension,
            "a proved dispatch reports the adapter's queried dimension"
        );
    }

    #[test]
    fn a_proved_copy_row_reports_the_queue_copy_flag() {
        // The flag is the ledger's row rather than a constant written beside the
        // queue, so the discriminating pair is a ledger with and without it.
        let without = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &FormatTable::default(),
        );
        assert!(!without.queues[0].capabilities.copy);

        let with = capabilities(
            &ledger_with(&[Capability::Graphics, Capability::Copy]),
            &adapter(),
            &FormatTable::default(),
        );
        assert!(with.queues[0].capabilities.copy);
    }

    #[test]
    fn a_proved_timestamp_row_reports_pass_boundaries() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics, Capability::TimestampQuery]),
            &adapter(),
            &FormatTable::default(),
        );
        assert_eq!(
            capabilities.timestamps,
            TimestampCapabilities::PassBoundaries,
            "a proved row may name the one placement this backend offers"
        );
    }

    #[test]
    fn a_proved_storage_row_reports_both_storage_directions() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics, Capability::StorageBuffer]),
            &adapter(),
            &FormatTable::default(),
        );

        assert_eq!(
            capabilities.buffers,
            // Both directions from the one proved row: the ledger carries storage
            // buffers as one domain, and reporting one direction without the other
            // would describe a distinction nothing observed.
            BufferCapabilities::new(true, true, false),
            "storage buffers reported, indirect still not proved"
        );
    }

    #[test]
    fn a_proved_indirect_row_reports_indirect_buffer_reads() {
        for row in [
            Capability::IndirectDraw,
            Capability::IndirectDispatch,
            Capability::MultiDrawIndirect,
        ] {
            let capabilities = capabilities(
                &ledger_with(&[Capability::Graphics, row]),
                &adapter(),
                &FormatTable::default(),
            );
            assert!(
                capabilities.buffers.indirect_read,
                "a proved {row:?} reads its parameters from a buffer"
            );
            assert!(
                !capabilities.buffers.storage_read && !capabilities.buffers.storage_write,
                "and proves nothing about the storage row, which is a different row"
            );
        }
    }

    #[test]
    fn the_shape_of_the_backend_is_reported_rather_than_inherited() {
        // These are the fields where this backend's answers differ from the
        // builder's fail-closed default or from the GL family's, so they are pinned
        // as read facts rather than left to agree with whatever the default is.
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &FormatTable::default(),
        );

        assert_eq!(
            capabilities.recording,
            RecordingCapabilities::new(RecordingModel::DeferredCommandBuffers, false),
            "this backend records a command buffer and submits it"
        );
        assert_eq!(
            capabilities.transitions,
            TransitionCapabilities::GraphManagedExplicit,
            "the barrier lowering turns a graph transition into a vkCmdPipelineBarrier"
        );
        assert_eq!(
            capabilities.synchronization,
            SynchronizationCapabilities::SingleQueueOrdering
        );
        assert_eq!(
            capabilities.transient_resources,
            TransientResourceCapabilities::new(false, false, false),
            "nothing pools, reuses or aliases an object yet"
        );
        assert_eq!(capabilities.timestamps, TimestampCapabilities::Unsupported);
    }

    #[test]
    fn the_colour_count_and_the_alignment_are_the_adapter_values() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &FormatTable::default(),
        );

        assert_eq!(capabilities.limits.max_color_attachments, 8);
        assert_eq!(
            capabilities.limits.min_uniform_buffer_offset_alignment,
            256,
            "the ledger's u32 alignment is widened, not narrowed"
        );
    }

    #[test]
    fn a_colour_format_and_a_depth_format_land_on_their_own_sides_of_an_attachment() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &table(&[
                row(TextureFormat::Rgba8Unorm, 1, true, true),
                row(TextureFormat::Depth32Float, 1, true, true),
            ]),
        );

        let colour = entry(&capabilities, TextureFormat::Rgba8Unorm);
        assert!(colour.color_attachment);
        assert!(
            !colour.depth_stencil_attachment,
            "one format is never both sides: Vulkan's renderable flag does not say which, so the mapped format does"
        );

        let depth = entry(&capabilities, TextureFormat::Depth32Float);
        assert!(depth.depth_stencil_attachment);
        assert!(!depth.color_attachment, "and the same the other way round");
    }

    #[test]
    fn a_format_is_folded_across_its_sample_counts() {
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &table(&[
                row(TextureFormat::Rgba8Unorm, 1, true, true),
                row(TextureFormat::Rgba8Unorm, 4, true, true),
            ]),
        );

        let colour = entry(&capabilities, TextureFormat::Rgba8Unorm);
        assert_eq!(
            colour.attachment_sample_counts,
            vec![1, 4],
            "both counts are legal attachments, ascending"
        );
        assert_eq!(
            reported(&capabilities),
            vec![TextureFormat::Rgba8Unorm],
            "two rows, one common format"
        );
    }

    #[test]
    fn the_storage_and_copy_facts_are_reported_from_the_row() {
        let mut observed = row(TextureFormat::Rgba8Unorm, 1, true, false);
        observed.storage_read = true;
        observed.storage_write = true;
        observed.copy_source = true;
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &table(&[observed]),
        );

        let colour = entry(&capabilities, TextureFormat::Rgba8Unorm);
        assert!(colour.storage_read && colour.storage_write);
        assert!(colour.copy_source && !colour.copy_destination);
        assert!(
            colour.attachment_sample_counts.is_empty(),
            "recorded, usable, and not an attachment at any count"
        );
    }

    #[test]
    fn a_format_no_discovery_recorded_is_absent_rather_than_reported() {
        // "Absent" and "recorded with every fact false" are different sentences for
        // the compiler, and this is the absent half of the pair.
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &table(&[row(TextureFormat::Rgba8Unorm, 1, true, true)]),
        );

        assert_eq!(reported(&capabilities), vec![TextureFormat::Rgba8Unorm]);
        assert!(
            !reported(&capabilities).contains(&TextureFormat::Rgba8UnormSrgb),
            "a format the table never recorded has no row, proved or otherwise"
        );
    }

    #[test]
    fn the_report_order_is_the_table_order() {
        // The value is compared and cached, so the order has to come from the
        // table rather than from a hasher. The table's order is its insertion order.
        let capabilities = capabilities(
            &ledger_with(&[Capability::Graphics]),
            &adapter(),
            &table(&[
                row(TextureFormat::Depth32Float, 1, true, true),
                row(TextureFormat::Rgba8Unorm, 1, true, true),
            ]),
        );

        assert_eq!(
            reported(&capabilities),
            vec![TextureFormat::Depth32Float, TextureFormat::Rgba8Unorm]
        );
    }

    #[test]
    fn a_real_adapter_lowers_to_the_contract() {
        // Step 11's lowering against the real driver. A machine with no Vulkan
        // adapter returns before asserting, because having no GPU is not this
        // test's subject.
        use crate::Validation;
        use crate::common::api::negotiate::CapabilitySource;
        use crate::native::vulkan::format::MAPPED;
        use crate::native::vulkan::open;

        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        // The device owns the per-format discovery now, so the lowering folds the
        // same table the storage-image row was read from rather than a second
        // recording of the same driver answers.
        let ledger = opened.device.ledger();
        let capabilities = capabilities(ledger, &opened.limits, opened.device.formats());

        assert_eq!(capabilities.queues.len(), 1, "one queue is reported");
        let queue = capabilities.queues[0];
        assert!(queue.capabilities.raster);
        assert!(
            queue.capabilities.copy,
            "the created device proves Copy from the API version, so the flag is set"
        );
        assert_eq!(
            queue.capabilities.compute,
            ledger.supports(Capability::Compute),
            "the queue and the ledger read the same fact"
        );
        assert_eq!(
            capabilities.buffers.indirect_read,
            ledger.supports(Capability::IndirectDispatch),
            "the indirect buffer flag is the ledger's row, not a second derivation"
        );
        assert_eq!(
            capabilities.buffers.storage_read,
            ledger.supports(Capability::StorageBuffer),
            "the storage half of the buffer row is the ledger's row too"
        );
        assert_eq!(
            capabilities.buffers.storage_write, capabilities.buffers.storage_read,
            "one ledger row serves both storage directions"
        );
        assert_eq!(
            capabilities.timestamps == TimestampCapabilities::PassBoundaries,
            opened.device.selected_queue().supports_timestamps(),
            "timestamps are reported exactly where the family reported valid bits"
        );
        assert!(!queue.capabilities.present);
        assert_eq!(capabilities.surface, None);

        assert_eq!(
            capabilities.limits.max_color_attachments,
            opened.limits.max_color_attachments
        );
        assert_eq!(
            capabilities.limits.min_uniform_buffer_offset_alignment,
            u64::from(opened.limits.min_uniform_buffer_offset_alignment)
        );
        if queue.capabilities.compute {
            assert_eq!(
                capabilities.limits.max_compute_workgroups_per_dimension,
                opened.limits.max_compute_workgroups_per_dimension
            );
        }

        assert_eq!(
            reported(&capabilities),
            MAPPED.to_vec(),
            "the lowering reports exactly the formats discovery asked about, in its own order"
        );

        // The facts below are the ones `Vulkan` makes mandatory for these formats
        // with optimal tiling, so a failure is a real disagreement rather than an
        // optional capability this board happens to lack.
        let colour = entry(&capabilities, TextureFormat::Rgba8Unorm);
        assert!(colour.sampled, "R8G8B8A8_UNORM is a mandatory sampled format");
        assert!(
            colour.filterable,
            "and mandatory with linear filtering under optimal tiling"
        );
        assert!(colour.color_attachment && !colour.depth_stencil_attachment);
        assert_eq!(colour.attachment_sample_counts, vec![1]);
        assert!(colour.copy_source && colour.copy_destination);

        let depth = entry(&capabilities, TextureFormat::Depth32Float);
        assert!(depth.depth_stencil_attachment && !depth.color_attachment);
        assert!(depth.copy_source && depth.copy_destination);
    }
}
