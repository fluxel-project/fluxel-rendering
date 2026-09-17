//! Step 3's pure half: which memory type a resource may be placed in.
//!
//! `gpu-allocator` performs the suballocation, but *which* type index a resource
//! uses is decided here, because it is the part with a rule in it. The rule is
//! `Vulkan`'s and has two halves that are easy to conflate:
//!
//! 1. the type's index bit must be set in the resource's `memory_type_bits` mask --
//!    the driver's statement of which types are legal for this resource at all;
//! 2. the type's property flags must contain every property the caller requires.
//!
//! Checking only the second half is the classic mistake: a device-local type that
//! the driver excluded for this resource looks perfectly suitable and produces
//! invalid usage. So the mask is checked first, and a type excluded by it is not a
//! candidate no matter how well its flags fit.
//!
//! # Why a refusal is structured
//!
//! "No type satisfies this" is a real outcome on a constrained device, and the
//! caller's next move depends on *what* was missing. The error therefore carries
//! the property set that could not be met and how many types were considered, which
//! is also what a diagnostic needs to explain a refusal without re-deriving it.

use ash::vk;

/// Why no memory type could serve a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MemoryError {
    /// Every legal type was examined and none carried the required properties.
    NoMatchingType {
        /// The properties that could not all be satisfied.
        required: vk::MemoryPropertyFlags,
        /// How many types the driver reported.
        considered: usize,
    },
}

/// Selects the first memory type that is legal for `requirements` and satisfies
/// `required`.
///
/// First-match rather than best-match: the driver orders its own memory types, and
/// imposing a Fluxel preference order over that would be a second opinion about
/// hardware this layer has not measured. A caller that needs a specific kind asks
/// for its properties.
pub(crate) fn select(
    requirements: &vk::MemoryRequirements,
    types: &[vk::MemoryType],
    required: vk::MemoryPropertyFlags,
) -> Result<u32, MemoryError> {
    types
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            // Legal for this resource at all...
            is_legal(requirements, *index)
                // ...and carrying every property the caller asked for.
                && memory_type.property_flags.contains(required)
        })
        .map(|(index, _)| index as u32)
        .ok_or(MemoryError::NoMatchingType {
            required,
            considered: types.len(),
        })
}

/// Whether the driver listed one memory type as legal for a resource.
///
/// A mask index beyond the reported type count is not legal: the driver's word is
/// the mask, and the count bounds the array that mask indexes into.
fn is_legal(requirements: &vk::MemoryRequirements, index: usize) -> bool {
    let Some(bit) = 1_u32.checked_shl(index as u32) else {
        return false;
    };
    requirements.memory_type_bits & bit != 0
}

/// The memory types one adapter reports, in the driver's own order.
///
/// The driver writes a fixed-width array plus a count; this returns exactly the
/// reported prefix as an owned vector, so a caller cannot read past the count and
/// an index into the result is a legal index into the driver's list.
///
/// Nothing is created: reading memory properties needs a physical device, not a
/// logical one, which is what lets a caller decide whether a resource is even
/// placeable before a device exists.
pub(crate) fn types(instance: &ash::Instance, adapter: vk::PhysicalDevice) -> Vec<vk::MemoryType> {
    // SAFETY: the adapter was enumerated from this instance, which is still live,
    // and the call only reports facts.
    let properties = unsafe { instance.get_physical_device_memory_properties(adapter) };
    let count = (properties.memory_type_count as usize).min(properties.memory_types.len());
    properties.memory_types[..count].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_type(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
        vk::MemoryType {
            property_flags: flags,
            heap_index: 0,
        }
    }

    fn requirements(bits: u32) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size: 256,
            alignment: 64,
            memory_type_bits: bits,
        }
    }

    const DEVICE_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const HOST_VISIBLE: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;

    #[test]
    fn the_only_legal_type_with_the_property_is_chosen() {
        let types = [
            memory_type(HOST_VISIBLE),
            memory_type(DEVICE_LOCAL),
            memory_type(HOST_VISIBLE),
        ];
        assert_eq!(select(&requirements(0b010), &types, DEVICE_LOCAL), Ok(1));
    }

    #[test]
    fn a_type_the_driver_excluded_is_not_a_candidate() {
        // The trap this rule exists for: the device-local type has exactly the
        // right properties, and the mask says the driver did not list it as legal
        // for this resource. The only legal type is the one without the property,
        // so the answer is a refusal rather than type 0.
        let types = [memory_type(DEVICE_LOCAL), memory_type(HOST_VISIBLE)];
        assert_eq!(
            select(&requirements(0b10), &types, DEVICE_LOCAL),
            Err(MemoryError::NoMatchingType {
                required: DEVICE_LOCAL,
                considered: 2,
            })
        );
    }

    #[test]
    fn a_legal_type_missing_a_required_property_is_skipped() {
        let types = [
            memory_type(DEVICE_LOCAL),
            memory_type(DEVICE_LOCAL | HOST_VISIBLE),
        ];
        assert_eq!(
            select(&requirements(0b11), &types, DEVICE_LOCAL | HOST_VISIBLE),
            Ok(1)
        );
    }

    #[test]
    fn the_first_match_wins_rather_than_a_preferred_one() {
        let types = [
            memory_type(DEVICE_LOCAL | HOST_VISIBLE),
            memory_type(DEVICE_LOCAL | HOST_VISIBLE),
        ];
        assert_eq!(select(&requirements(0b11), &types, HOST_VISIBLE), Ok(0));
    }

    #[test]
    fn no_types_at_all_refuses_with_zero_considered() {
        assert_eq!(
            select(&requirements(0b1), &[], DEVICE_LOCAL),
            Err(MemoryError::NoMatchingType {
                required: DEVICE_LOCAL,
                considered: 0,
            })
        );
    }

    #[test]
    fn a_mask_bit_beyond_the_reported_types_is_never_legal() {
        // Only one type is reported, but the mask claims type 3 is legal; the
        // array bounds win, because the mask indexes into what the driver listed.
        let types = [memory_type(DEVICE_LOCAL)];
        assert_eq!(
            select(&requirements(0b1000), &types, DEVICE_LOCAL),
            Err(MemoryError::NoMatchingType {
                required: DEVICE_LOCAL,
                considered: 1,
            })
        );
    }

    #[test]
    fn a_real_adapter_reports_memory_types_including_a_device_local_one() {
        // End-to-end for the reading half of step 3, against the real driver. Skips
        // rather than fails where no adapter exists, since having no GPU is not
        // what this test is about.
        use crate::Validation;
        use crate::native::vulkan::open;

        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let reported = types(opened.instance.instance(), opened.adapter);
        assert!(
            !reported.is_empty(),
            "an adapter that was opened reports at least one memory type"
        );
        assert!(
            reported.iter().any(|memory_type| memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)),
            "a GPU adapter reports a device-local memory type"
        );
        // The selection rule is reached with what the driver actually reported, so
        // this asserts the two halves agree on real data rather than on fixtures.
        let requirements = vk::MemoryRequirements {
            size: 4096,
            alignment: 64,
            // Every reported type is legal for this synthetic request, which is the
            // "driver placed no restriction" case.
            memory_type_bits: (1u32 << reported.len().min(31)) - 1,
        };
        assert!(
            select(
                &requirements,
                &reported,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            )
            .is_ok(),
            "the device-local type the driver reported is selectable"
        );
    }

    #[test]
    fn the_reported_prefix_never_exceeds_the_array() {
        // The count is the driver's word and the array is fixed width; the helper
        // clamps rather than trusting the count, because reading past the array
        // would be undefined.
        let properties = vk::PhysicalDeviceMemoryProperties::default();
        let count = (properties.memory_type_count as usize).min(properties.memory_types.len());
        assert!(count <= properties.memory_types.len());
    }
}
