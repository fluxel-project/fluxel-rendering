//! Step 2's first half: which adapters exist, and what each one reports.
//!
//! Nothing is created here. Enumerating physical devices and reading their
//! properties needs no device, which is what lets the adapter be chosen -- and its
//! absence reported -- before any device object exists.
//!
//! # Facts are reported only where they were read
//!
//! Three limits this adapter type carries cannot be read from
//! [`vk::PhysicalDeviceProperties`] alone: multiview view count and the
//! multi-draw count live in extension structures, and the PCI bus identifier and
//! driver name live in others. Those are reported as the values that reject work
//! -- zero, `None`, an empty string -- rather than being guessed or copied from a
//! neighbouring fact. Reading them belongs to the step that needs them, together
//! with the extension query that proves the structure is present.
//!
//! The same rule explains `driver`: the packed `driver_version` word is decoded
//! with the standard `Vulkan` packing, which is what the specification defines for
//! a vendor that does not publish its own scheme. A vendor-specific decoding is a
//! refinement, and claiming one before it exists would misreport the driver.

use std::ffi::CStr;

use ash::vk;

use crate::common::caps::AdapterLimits;
use crate::{Backend, DeviceKind, HardwareInfo};

/// What one physical device reported.
#[derive(Clone, Debug)]
pub(crate) struct AdapterFacts {
    /// Driver identity, in the shape the public facade already reports.
    pub hardware: HardwareInfo,
    /// The numerical facts the capability ledger reads.
    pub limits: AdapterLimits,
}

/// Why an adapter could not be enumerated or selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdapterError {
    /// The loader refused to enumerate physical devices.
    Enumeration(vk::Result),
    /// The requested index is not present.
    Unavailable {
        /// The index that was requested.
        index: usize,
        /// How many adapters were found.
        available: usize,
    },
}

/// Enumerates the physical devices one instance can see.
pub(crate) fn enumerate(instance: &ash::Instance) -> Result<Vec<vk::PhysicalDevice>, AdapterError> {
    // SAFETY: the instance is live and owns the entry point used; the returned
    // handles are borrowed handles into that instance and create nothing.
    unsafe { instance.enumerate_physical_devices() }.map_err(AdapterError::Enumeration)
}

/// Reads one physical device's facts.
pub(crate) fn describe(instance: &ash::Instance, adapter: vk::PhysicalDevice) -> AdapterFacts {
    // SAFETY: the adapter was enumerated from this instance, which is still live.
    let properties = unsafe { instance.get_physical_device_properties(adapter) };
    AdapterFacts {
        hardware: hardware(&properties),
        limits: limits(&properties.limits),
    }
}

/// Selects one adapter by index, reporting the count when it is absent.
///
/// Separate from enumeration so the rule is testable without a driver, and stated
/// once because both the headless path and the surface path select the same way:
/// an index is a bootstrap choice, and an absent index is a structured refusal that
/// names how many adapters exist.
pub(crate) fn select(available: usize, index: usize) -> Result<usize, AdapterError> {
    if index < available {
        Ok(index)
    } else {
        Err(AdapterError::Unavailable { index, available })
    }
}

/// Lowers the driver's identity facts into the public hardware description.
fn hardware(properties: &vk::PhysicalDeviceProperties) -> HardwareInfo {
    HardwareInfo {
        backend: Backend::Vulkan,
        name: device_name(properties),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        kind: device_kind(properties.device_type),
        // Needs `VkPhysicalDevicePCIBusInfoPropertiesEXT`; owed by the step that
        // needs the fact rather than guessed from the device id.
        pci_bus_id: String::new(),
        driver: decode_driver_version(properties.driver_version),
        // Needs `VkPhysicalDeviceDriverProperties`; owed for the same reason.
        driver_info: String::new(),
    }
}

/// Reads the fixed-width device name the driver writes.
fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    // SAFETY: the driver guarantees a NUL-terminated name inside the fixed-width
    // buffer, which is why `from_ptr` is the right reader rather than a slice.
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    name.to_string_lossy().into_owned()
}

/// Maps the driver's classification onto the public one.
///
/// The raw values are the ones the `Vulkan` specification fixes, and they are
/// compared as raw values because the generated type is a newtype over an integer
/// rather than an enum; an unrecognized value is "not classified more precisely",
/// which is what the public type's `Other` means.
fn device_kind(device_type: vk::PhysicalDeviceType) -> DeviceKind {
    match device_type.as_raw() {
        1 => DeviceKind::Integrated,
        2 => DeviceKind::Discrete,
        3 => DeviceKind::Virtual,
        4 => DeviceKind::Cpu,
        _ => DeviceKind::Other,
    }
}

/// Decodes the packed driver version with the specification's own layout.
fn decode_driver_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        version >> 22,
        (version >> 12) & 0x3ff,
        version & 0xfff
    )
}

/// Lowers the driver's numerical facts into the ledger's limit set.
fn limits(limits: &vk::PhysicalDeviceLimits) -> AdapterLimits {
    AdapterLimits {
        max_texture_dimension_2d: limits.max_image_dimension2_d,
        max_texture_dimension_3d: limits.max_image_dimension3_d,
        max_texture_array_layers: limits.max_image_array_layers,
        max_color_attachments: limits.max_color_attachments,
        max_vertex_attributes: limits.max_vertex_input_attributes,
        max_bind_groups: limits.max_bound_descriptor_sets,
        min_uniform_buffer_offset_alignment: narrow(limits.min_uniform_buffer_offset_alignment),
        min_storage_buffer_offset_alignment: narrow(limits.min_storage_buffer_offset_alignment),
        max_uniform_buffer_binding_size: u64::from(limits.max_uniform_buffer_range),
        max_storage_buffer_binding_size: u64::from(limits.max_storage_buffer_range),
        max_compute_workgroups_per_dimension: limits.max_compute_work_group_count,
        max_compute_workgroup_size: limits.max_compute_work_group_size,
        max_compute_invocations_per_workgroup: limits.max_compute_work_group_invocations,
        max_samples: highest_sample_count(limits.framebuffer_color_sample_counts),
        // Extension-structure facts, deliberately unproved here.
        max_multiview_view_count: 0,
        max_multi_draw_indirect_count: None,
    }
}

/// Narrows a device-size alignment onto the ledger's `u32`, saturating.
///
/// An alignment larger than `u32` is not a real driver answer; saturating rather
/// than wrapping keeps the value conservative and never smaller than the truth,
/// which is the direction an alignment check has to fail in.
fn narrow(value: vk::DeviceSize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Returns the largest sample count the driver reported as supported.
///
/// Zero means none was reported, which is the rejecting value: a context that
/// answered nothing satisfies no multisample floor. The candidates are probed in
/// descending order and only through the generated `contains`, because the flag
/// type's raw representation is private to `ash`.
fn highest_sample_count(flags: vk::SampleCountFlags) -> u32 {
    const CANDIDATES: [(u32, vk::SampleCountFlags); 7] = [
        (64, vk::SampleCountFlags::TYPE_64),
        (32, vk::SampleCountFlags::TYPE_32),
        (16, vk::SampleCountFlags::TYPE_16),
        (8, vk::SampleCountFlags::TYPE_8),
        (4, vk::SampleCountFlags::TYPE_4),
        (2, vk::SampleCountFlags::TYPE_2),
        (1, vk::SampleCountFlags::TYPE_1),
    ];
    for (count, flag) in CANDIDATES {
        if flags.contains(flag) {
            return count;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_inside_the_count_selects_that_adapter() {
        assert_eq!(select(3, 0), Ok(0));
        assert_eq!(select(3, 2), Ok(2));
    }

    #[test]
    fn an_absent_index_names_both_numbers() {
        assert_eq!(
            select(3, 3),
            Err(AdapterError::Unavailable {
                index: 3,
                available: 3
            })
        );
    }

    #[test]
    fn no_adapters_at_all_refuses_index_zero() {
        assert_eq!(
            select(0, 0),
            Err(AdapterError::Unavailable {
                index: 0,
                available: 0
            })
        );
    }

    #[test]
    fn the_reported_sample_count_is_the_largest_one_present() {
        assert_eq!(highest_sample_count(vk::SampleCountFlags::TYPE_1), 1);
        assert_eq!(highest_sample_count(vk::SampleCountFlags::TYPE_8), 8);
        assert_eq!(highest_sample_count(vk::SampleCountFlags::TYPE_64), 64);
    }

    #[test]
    fn the_standard_driver_packing_decodes_to_its_three_parts() {
        assert_eq!(decode_driver_version(vk::make_api_version(0, 24, 3, 7)), "24.3.7");
    }

    #[test]
    fn an_unrecognized_device_type_is_not_classified_more_precisely() {
        assert_eq!(device_kind(vk::PhysicalDeviceType::from_raw(99)), DeviceKind::Other);
        assert_eq!(device_kind(vk::PhysicalDeviceType::CPU), DeviceKind::Cpu);
        assert_eq!(
            device_kind(vk::PhysicalDeviceType::DISCRETE_GPU),
            DeviceKind::Discrete
        );
    }
}
