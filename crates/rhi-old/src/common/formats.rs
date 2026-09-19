//! The per-format evidence table a backend's discovery fills.
//!
//! # Why this shape, and not a bitflags set
//!
//! The native path being replaced reports per-format support as
//! `wgpu_hal::TextureFormatCapabilities` bitflags plus predicates in its lowering,
//! while the GL family already carries an *evidence-carrying* table keyed by
//! `(format, resource kind, sample count)` (`webgl2/api/formats.rs`). Those are two
//! different shapes rather than two spellings of one, so the unified lowering the
//! stage plan calls W9 cannot reconcile them by mapping names. This module promotes
//! the evidence-carrying shape, which is the one that can answer "was this fact
//! proved, and by what" as well as "is it true".
//!
//! # The one rule every field obeys
//!
//! A false field is a conservative unsupported fact, never an invitation to
//! emulate the operation, and a format no discovery examined is *absent* rather
//! than present-and-false. That is the same distinction the capability ledger
//! makes: an unexamined row and a refused row are one answer to a caller deciding
//! what to record, and two different sentences in a diagnostic.
//!
//! # What is deliberately not here
//!
//! - **The compressed-format rule.** ADR-0011's rule -- a compressed format claims
//!   sampling and copying and nothing else -- is enforced centrally by
//!   [`FormatTable::record`] once the portable vocabulary can name a compressed
//!   format, which is W8a in 0.17. Until then no such format exists to refuse, and
//!   a predicate written now would be vocabulary with no fact behind it.
//! - **The resource kind.** A renderbuffer is a GL-family resource rather than a
//!   texture, which is why the GL table keys on it. This table keys on
//!   `(format, sample count)`, and the GL backend keeps its renderbuffer rows as an
//!   internal detail mapping onto the same texture-shaped facts (W6).
//! - **Storage facts implied by other facts.** Read and write stay split because
//!   neither implies the other: a format may legalize one direction alone.

use fluxel_rendergraph::TextureFormat;

/// Origin of one exact per-format fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FormatEvidence {
    /// An unconditional guarantee of the API the backend selected.
    CoreGuaranteed,
    /// A named optional facility on this same context supplies the fact.
    ///
    /// Provenance for diagnostics only: it is never compared, parsed, or consulted
    /// when deciding anything, so a driver's or extension registry's spelling of a
    /// name can never become a semantic dependency of this crate.
    ExtensionAcquired(&'static str),
    /// A device-specific query or operation for exactly this format succeeded.
    ///
    /// This is the native backends' evidence: an explicit API establishes a format
    /// fact by asking the driver about that format, not by a profile guarantee.
    OperationProbed,
}

/// Exact use facts for one `(format, sample count)` pair.
///
/// Every field is a fact about using the format in that role, and every one of them
/// defaults to the value that rejects work when it is written by hand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FormatCapabilities {
    /// The portable format this row describes.
    pub format: TextureFormat,
    /// The exact sample count this row describes. Zero is invalid and refused.
    pub sample_count: u32,
    /// Whether this row is an API guarantee or concrete device evidence.
    pub evidence: FormatEvidence,
    /// Sampling from this format at this count is legal.
    pub sampled: bool,
    /// Linear filtering is legal for this format at this count.
    pub filterable: bool,
    /// Rendering into this format at this count is legal.
    pub renderable: bool,
    /// Blending into a colour attachment of this format is legal.
    pub blendable: bool,
    /// Shader reads through a storage image are legal.
    pub storage_read: bool,
    /// Shader writes through a storage image are legal.
    pub storage_write: bool,
    /// Copying from this format at this count is legal.
    pub copy_source: bool,
    /// Copying to this format at this count is legal.
    pub copy_destination: bool,
}

/// Why one exact format fact could not be recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FormatTableError {
    /// A zero sample count describes no resource at all.
    ZeroSampleCount {
        /// The format the invalid row named.
        format: TextureFormat,
    },
    /// A storage fact arrived without the operation evidence it requires.
    ///
    /// A storage direction is the one fact no profile guarantees, because whether a
    /// format may be read or written through a storage image is exactly what a
    /// device has to be asked. Accepting it as a guarantee would be a capability
    /// claim with no backer.
    StorageRequiresProbe {
        /// The format the refused row named.
        format: TextureFormat,
        /// The sample count the refused row named.
        sample_count: u32,
    },
    /// A second observation of the same pair disagreed with the first.
    ConflictingObservation {
        /// The format the two rows disagreed about.
        format: TextureFormat,
        /// The sample count the two rows disagreed about.
        sample_count: u32,
    },
}

/// Immutable facts, one row per `(format, sample count)` discovery examined.
///
/// # Why a sequence rather than a sorted map
///
/// [`TextureFormat`] is `#[non_exhaustive]` and deliberately **not** `Ord`: nothing
/// about a format's identity is ordered, so a map keyed by it would have to invent
/// an ordering. That is the mistake the resource table already paid for when it
/// reached for a `BTreeMap` over an id type that is only `Eq + Hash`. A `HashMap`
/// avoids the invented order but replaces it with a randomized one, and a capability
/// table is compared and cached, so its iteration order has to be a function of its
/// contents rather than of a hasher seed.
///
/// The order is therefore insertion order, and it is deterministic because the
/// caller records from its own fixed list of formats: the same discovery always
/// produces the same sequence. [`Self::record`] keeps an identical repeat in place
/// instead of appending it, so re-running a discovery cannot reorder anything.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FormatTable {
    entries: Vec<FormatCapabilities>,
}

impl FormatTable {
    /// Inserts one exact observation; a disagreeing repeat is refused.
    ///
    /// An identical repeat is accepted and keeps the row where it was, because two
    /// discoveries agreeing is not a contradiction. Two *different* observations of
    /// one `(format, sample count)` pair are, and silently keeping the later one
    /// would make the table depend on discovery order.
    pub(crate) fn record(
        &mut self,
        capabilities: FormatCapabilities,
    ) -> Result<(), FormatTableError> {
        if capabilities.sample_count == 0 {
            return Err(FormatTableError::ZeroSampleCount {
                format: capabilities.format,
            });
        }
        if (capabilities.storage_read || capabilities.storage_write)
            && capabilities.evidence != FormatEvidence::OperationProbed
        {
            return Err(FormatTableError::StorageRequiresProbe {
                format: capabilities.format,
                sample_count: capabilities.sample_count,
            });
        }
        if let Some(previous) = self.get(capabilities.format, capabilities.sample_count) {
            if previous != capabilities {
                return Err(FormatTableError::ConflictingObservation {
                    format: capabilities.format,
                    sample_count: capabilities.sample_count,
                });
            }
            return Ok(());
        }
        self.entries.push(capabilities);
        Ok(())
    }

    /// Returns the facts recorded for one exact pair.
    ///
    /// `None` means nothing examined this pair; `Some` with every field false means
    /// something examined it and the device proved nothing. The two are different
    /// sentences and are kept different here.
    pub(crate) fn get(
        &self,
        format: TextureFormat,
        sample_count: u32,
    ) -> Option<FormatCapabilities> {
        self.entries
            .iter()
            .copied()
            .find(|facts| facts.format == format && facts.sample_count == sample_count)
    }

    /// Iterates every recorded row in the order it was recorded.
    pub(crate) fn iter(&self) -> impl Iterator<Item = FormatCapabilities> + '_ {
        self.entries.iter().copied()
    }

    /// Whether at least one recorded row proves read **and** write storage access.
    ///
    /// This is the per-format half of a device-level storage-image capability: the
    /// capability is the domain, and one format a shader may both read and write
    /// through a storage image is what the domain needs. A row proving one direction
    /// alone does not satisfy it, because [`FormatCapabilities`] keeps the two split
    /// and neither implies the other.
    ///
    /// The evidence is deliberately not re-checked here. [`Self::record`] refuses a
    /// storage fact that arrived without an operation probe, so a check here would be
    /// a second definition of a rule the table already owns.
    pub(crate) fn has_storage_read_write(&self) -> bool {
        self.entries
            .iter()
            .any(|facts| facts.storage_read && facts.storage_write)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(format: TextureFormat, sample_count: u32) -> FormatCapabilities {
        FormatCapabilities {
            format,
            sample_count,
            evidence: FormatEvidence::CoreGuaranteed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: false,
            storage_write: false,
            copy_source: true,
            copy_destination: true,
        }
    }

    #[test]
    fn an_unexamined_pair_is_absent_rather_than_false() {
        let table = FormatTable::default();
        assert_eq!(table.get(TextureFormat::Rgba8Unorm, 1), None);
        assert_eq!(table.iter().count(), 0);
    }

    #[test]
    fn a_recorded_row_is_read_back_field_for_field() {
        let mut table = FormatTable::default();
        let facts = row(TextureFormat::Rgba8Unorm, 1);
        assert_eq!(table.record(facts), Ok(()));
        assert_eq!(table.get(TextureFormat::Rgba8Unorm, 1), Some(facts));
        // A sample count that was never recorded is a different pair.
        assert_eq!(table.get(TextureFormat::Rgba8Unorm, 4), None);
    }

    #[test]
    fn a_zero_sample_count_describes_no_resource() {
        let mut table = FormatTable::default();
        assert_eq!(
            table.record(row(TextureFormat::Rgba8Unorm, 0)),
            Err(FormatTableError::ZeroSampleCount {
                format: TextureFormat::Rgba8Unorm
            })
        );
        assert_eq!(table.iter().count(), 0);
    }

    #[test]
    fn a_storage_fact_without_an_operation_probe_is_refused() {
        let mut table = FormatTable::default();
        let mut facts = row(TextureFormat::Rgba8Unorm, 1);
        facts.storage_read = true;
        assert_eq!(
            table.record(facts),
            Err(FormatTableError::StorageRequiresProbe {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            })
        );
        assert_eq!(
            table.record(FormatCapabilities {
                evidence: FormatEvidence::ExtensionAcquired("a named facility"),
                ..facts
            }),
            Err(FormatTableError::StorageRequiresProbe {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            })
        );
        assert_eq!(
            table.record(FormatCapabilities {
                evidence: FormatEvidence::OperationProbed,
                ..facts
            }),
            Ok(())
        );
    }

    #[test]
    fn a_write_only_storage_fact_needs_the_same_proof_as_a_read() {
        let mut table = FormatTable::default();
        let mut facts = row(TextureFormat::Rgba8Unorm, 1);
        facts.storage_write = true;
        assert_eq!(
            table.record(facts),
            Err(FormatTableError::StorageRequiresProbe {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            })
        );
    }

    #[test]
    fn a_disagreeing_repeat_is_refused_and_the_first_row_stands() {
        let mut table = FormatTable::default();
        let first = row(TextureFormat::Rgba8Unorm, 1);
        assert_eq!(table.record(first), Ok(()));
        let second = FormatCapabilities {
            sampled: false,
            ..first
        };
        assert_eq!(
            table.record(second),
            Err(FormatTableError::ConflictingObservation {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            })
        );
        assert_eq!(table.get(TextureFormat::Rgba8Unorm, 1), Some(first));
    }

    #[test]
    fn an_identical_repeat_keeps_its_row_and_its_position() {
        let mut table = FormatTable::default();
        let rgba = row(TextureFormat::Rgba8Unorm, 1);
        let depth = row(TextureFormat::Depth32Float, 1);
        assert_eq!(table.record(rgba), Ok(()));
        assert_eq!(table.record(depth), Ok(()));
        assert_eq!(table.record(rgba), Ok(()));
        let formats: Vec<TextureFormat> = table.iter().map(|facts| facts.format).collect();
        assert_eq!(formats, vec![TextureFormat::Rgba8Unorm, TextureFormat::Depth32Float]);
    }

    #[test]
    fn a_proved_negative_is_not_an_absent_row() {
        // The distinction the module doc states: recording a row whose every fact is
        // false is a device answer, and the table must not collapse it into `None`.
        let mut table = FormatTable::default();
        let nothing = FormatCapabilities {
            format: TextureFormat::Rgba8UnormSrgb,
            sample_count: 1,
            evidence: FormatEvidence::OperationProbed,
            sampled: false,
            filterable: false,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        };
        assert_eq!(table.record(nothing), Ok(()));
        let read_back = table
            .get(TextureFormat::Rgba8UnormSrgb, 1)
            .expect("the pair was examined");
        assert!(!read_back.sampled);
        assert_eq!(read_back.evidence, FormatEvidence::OperationProbed);
        assert_eq!(table.get(TextureFormat::Bgra8Unorm, 1), None);
    }

    #[test]
    fn only_a_row_that_proves_both_storage_directions_satisfies_the_query() {
        // This query is the per-format half of a device-level storage-image
        // capability, so each direction alone has to leave it unsatisfied: a
        // read-only row describes half the domain the capability names.
        assert!(
            !FormatTable::default().has_storage_read_write(),
            "a table nothing was recorded in proves no storage format"
        );

        for (read, write) in [(true, false), (false, true)] {
            let mut table = FormatTable::default();
            let mut facts = row(TextureFormat::Rgba8Unorm, 1);
            // A storage fact is the one fact the table demands was probed.
            facts.evidence = FormatEvidence::OperationProbed;
            facts.storage_read = read;
            facts.storage_write = write;
            assert_eq!(table.record(facts), Ok(()));
            assert!(
                !table.has_storage_read_write(),
                "read={read} write={write} is one direction, not the domain"
            );
        }

        let mut table = FormatTable::default();
        let mut facts = row(TextureFormat::Depth32Float, 1);
        facts.evidence = FormatEvidence::OperationProbed;
        facts.storage_read = true;
        facts.storage_write = true;
        assert_eq!(table.record(facts), Ok(()));
        assert!(
            table.has_storage_read_write(),
            "one row proving both directions satisfies the query"
        );
    }
}
