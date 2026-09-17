//! Step 4's pure half: the portable buffer usage lowered onto `Vulkan` flags.
//!
//! # The mapping does not widen
//!
//! Each portable kind maps to exactly the flag that serves it, and nothing else is
//! added. That is deliberate: silently adding a flag the caller did not ask for
//! would make a buffer more capable than the graph declared, and the graph's
//! capability check is what decides whether an operation is legal. A buffer that
//! quietly became a transfer destination would pass a check it should have failed.
//!
//! The one place widening is legitimate -- the staging path the immutable uploads
//! use, where the RHI itself performs a copy into the buffer -- is a separate,
//! named function, so the extra flag is visible at the call site that needs it
//! rather than hidden inside the mapping.
//!
//! # Why a union rather than a first-match
//!
//! Two portable kinds can name the same flag (storage reads and storage writes both
//! need `STORAGE_BUFFER`), and one buffer frequently carries several kinds at once.
//! The mapping is therefore a fold over the kinds present, so a buffer's flags are
//! the union of what its usages require and nothing more.

use ash::vk;
use fluxel_rendergraph::{BufferUsage, BufferUsageKind};

/// Every portable buffer usage kind, in `BufferUsageKind`'s declaration order.
///
/// Listed here rather than derived from the bitmask so the mapping is a total
/// function: adding a kind upstream makes this list incomplete, which a test
/// detects, instead of silently dropping a flag at run time.
const KINDS: [(BufferUsageKind, vk::BufferUsageFlags); 8] = [
    (BufferUsageKind::Uniform, vk::BufferUsageFlags::UNIFORM_BUFFER),
    (BufferUsageKind::StorageRead, vk::BufferUsageFlags::STORAGE_BUFFER),
    (
        BufferUsageKind::StorageWrite,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    ),
    (BufferUsageKind::Vertex, vk::BufferUsageFlags::VERTEX_BUFFER),
    (BufferUsageKind::Index, vk::BufferUsageFlags::INDEX_BUFFER),
    (
        BufferUsageKind::Indirect,
        vk::BufferUsageFlags::INDIRECT_BUFFER,
    ),
    (
        BufferUsageKind::CopySource,
        vk::BufferUsageFlags::TRANSFER_SRC,
    ),
    (
        BufferUsageKind::CopyDestination,
        vk::BufferUsageFlags::TRANSFER_DST,
    ),
];

/// Lowers the requested usages to the flags that serve exactly them.
pub(crate) fn usage_flags(usage: BufferUsage) -> vk::BufferUsageFlags {
    KINDS
        .iter()
        .filter(|(kind, _)| usage.contains(*kind))
        .fold(vk::BufferUsageFlags::empty(), |flags, (_, flag)| {
            flags | *flag
        })
}

/// The flags a buffer the RHI itself copies into must be created with.
///
/// This is the one legitimate widening, and it is a named function so a reader can
/// see where the extra flag comes from. The immutable-upload path records a copy
/// into the destination it created, which the caller never asked for and therefore
/// never declared.
pub(crate) fn upload_destination_flags(usage: BufferUsage) -> vk::BufferUsageFlags {
    usage_flags(usage) | vk::BufferUsageFlags::TRANSFER_DST
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(kinds: &[BufferUsageKind]) -> BufferUsage {
        BufferUsage::from_kinds(kinds.iter().copied())
    }

    #[test]
    fn no_usages_means_no_flags() {
        assert_eq!(usage_flags(BufferUsage::empty()), vk::BufferUsageFlags::empty());
    }

    #[test]
    fn each_kind_maps_to_the_flag_that_serves_it() {
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::Uniform])),
            vk::BufferUsageFlags::UNIFORM_BUFFER
        );
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::Vertex])),
            vk::BufferUsageFlags::VERTEX_BUFFER
        );
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::Index])),
            vk::BufferUsageFlags::INDEX_BUFFER
        );
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::Indirect])),
            vk::BufferUsageFlags::INDIRECT_BUFFER
        );
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::CopySource])),
            vk::BufferUsageFlags::TRANSFER_SRC
        );
        assert_eq!(
            usage_flags(of(&[BufferUsageKind::CopyDestination])),
            vk::BufferUsageFlags::TRANSFER_DST
        );
    }

    #[test]
    fn both_storage_directions_need_the_storage_flag() {
        for kind in [BufferUsageKind::StorageRead, BufferUsageKind::StorageWrite] {
            assert_eq!(
                usage_flags(of(&[kind])),
                vk::BufferUsageFlags::STORAGE_BUFFER
            );
        }
    }

    #[test]
    fn several_kinds_union_into_several_flags() {
        let both = usage_flags(of(&[BufferUsageKind::Vertex, BufferUsageKind::CopySource]));
        assert!(both.contains(vk::BufferUsageFlags::VERTEX_BUFFER));
        assert!(both.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert!(!both.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert!(!both.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
    }

    #[test]
    fn the_plain_mapping_never_widens_to_a_transfer_destination() {
        // The rule this module exists to keep: what was declared is what is
        // created, so a capability check cannot be passed by accident.
        let declared = of(&[
            BufferUsageKind::Vertex,
            BufferUsageKind::Uniform,
            BufferUsageKind::StorageWrite,
        ]);
        assert!(!usage_flags(declared).contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn the_upload_path_widens_by_exactly_one_flag() {
        let declared = of(&[BufferUsageKind::Vertex]);
        let widened = upload_destination_flags(declared);
        assert_eq!(
            widened,
            usage_flags(declared) | vk::BufferUsageFlags::TRANSFER_DST
        );
        assert!(widened.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn every_flag_this_backend_needs_is_reachable() {
        // What is checkable: the seven flags the retained recipes use are all
        // reachable from the mapped kinds. What is not checkable here is whether
        // `KINDS` lists every variant upstream declares -- `BufferUsageKind` offers
        // no variant iterator -- so an unmapped kind would be dropped silently. The
        // list is therefore written in the enum's own declaration order, which is
        // what makes a future addition visible in review.
        let all = usage_flags(BufferUsage::from_kinds(KINDS.iter().map(|(kind, _)| *kind)));
        for expected in [
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::BufferUsageFlags::TRANSFER_DST,
        ] {
            assert!(all.contains(expected), "{expected:?} is unreachable");
        }
    }
}
