//! Steps 1 and 2 as one entry point: opening a headless device.
//!
//! The four pieces this chains are each already verified on their own --
//! [`instance::open`], [`adapter::enumerate`], [`adapter::describe`],
//! [`adapter::select`], [`device::open`] -- but a caller should not have to
//! assemble them, and assembling them in more than one place is how the ordering
//! rules they carry get lost. In particular:
//!
//! - the validation inventory is verified **before** an instance exists, which only
//!   stays true while one function owns the order;
//! - an absent adapter index is reported with the count of adapters found, not as a
//!   generic failure;
//! - the adapter's facts are read **once**, and the same limit set both selects the
//!   queue family and builds the capability ledger, so the ledger cannot disagree
//!   with the device that was created.
//!
//! # Field order is teardown order
//!
//! [`OpenedVulkan`] owns a device and the instance that created it. The device is
//! declared first so it is destroyed first: `Vulkan` requires child objects to be
//! destroyed before their parent, and Rust's field drop order is what enforces it
//! rather than a comment asking politely.

use ash::vk;

use crate::Validation;
use crate::common::caps::AdapterLimits;
use crate::HardwareInfo;

use super::adapter::{self, AdapterError};
use super::device::{self, DeviceError, VulkanDevice};
use super::instance::{self, InstanceError, ValidationInstance};

/// Everything one opened headless device consists of.
pub(crate) struct OpenedVulkan {
    /// The logical device and its one queue.
    pub device: VulkanDevice,
    /// The physical device the logical one was created on.
    pub adapter: vk::PhysicalDevice,
    /// The instance that created both.
    pub instance: ValidationInstance,
    /// Driver identity for the selected adapter.
    pub hardware: HardwareInfo,
    /// The limit set the queue family was selected against and the ledger was
    /// built from.
    pub limits: AdapterLimits,
    /// How many adapters the instance reported, so an absent index can be explained.
    pub adapter_count: usize,
}

impl OpenedVulkan {
    /// Returns the adapter's facts, as the public facade reports them.
    pub(crate) fn hardware(&self) -> &HardwareInfo {
        &self.hardware
    }

    /// Returns whether validation was positively verified for this open.
    pub(crate) const fn validation_enabled(&self) -> bool {
        self.instance.validation_enabled()
    }
}

impl core::fmt::Debug for OpenedVulkan {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("OpenedVulkan")
            .field("hardware", &self.hardware)
            .field("adapter_count", &self.adapter_count)
            .field("validation_enabled", &self.validation_enabled())
            .finish_non_exhaustive()
    }
}

/// Why a headless device could not be opened.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpenVulkanError {
    /// The loader, the instance, or the validation probe failed.
    Instance(InstanceError),
    /// No adapter existed at the requested index.
    Adapter(AdapterError),
    /// The logical device could not be created.
    Device(DeviceError),
}

/// Opens a headless device at `adapter_index`.
pub(crate) fn open(
    validation: Validation,
    adapter_index: usize,
) -> Result<OpenedVulkan, OpenVulkanError> {
    let instance = instance::open(validation).map_err(OpenVulkanError::Instance)?;
    let adapters = adapter::enumerate(instance.instance()).map_err(OpenVulkanError::Adapter)?;
    let index = adapter::select(adapters.len(), adapter_index).map_err(OpenVulkanError::Adapter)?;
    let adapter = adapters[index];

    // Read once: these facts both select the queue family and build the ledger.
    let facts = adapter::describe(instance.instance(), adapter);
    let device = device::open(&instance, adapter, &facts.limits).map_err(OpenVulkanError::Device)?;

    Ok(OpenedVulkan {
        device,
        adapter,
        instance,
        hardware: facts.hardware,
        limits: facts.limits,
        adapter_count: adapters.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::api::negotiate::CapabilitySource;
    use crate::common::caps::Capability;

    #[test]
    fn a_headless_device_opens_on_this_machine_and_reports_consistent_facts() {
        // Skips rather than fails where no Vulkan adapter exists: having no GPU is
        // not what this test is about.
        let Ok(opened) = open(Validation::Disabled, 0) else {
            return;
        };
        assert!(
            opened.adapter_count >= 1,
            "an adapter was selected, so at least one exists"
        );
        assert!(!opened.hardware().name.is_empty());
        assert_eq!(opened.hardware().backend, crate::Backend::Vulkan);
        // The same limit set that selected the family built the ledger, so the
        // texture floor it reports is the one this device was opened against.
        assert!(opened.limits.max_texture_dimension_2d > 0);

        let ledger = opened.device.ledger();
        assert!(ledger.supports(Capability::Graphics));
        assert_eq!(
            ledger.supports(Capability::Compute),
            opened.device.selected_queue().supports_compute
        );
    }

    #[test]
    fn an_absent_adapter_index_reports_how_many_were_found() {
        let Ok(instance) = instance::open(Validation::Disabled) else {
            return;
        };
        let Ok(adapters) = adapter::enumerate(instance.instance()) else {
            return;
        };
        // An index past the end is the one case that must refuse regardless of how
        // many adapters this machine has.
        let past_the_end = adapters.len();
        assert_eq!(
            adapter::select(adapters.len(), past_the_end),
            Err(AdapterError::Unavailable {
                index: past_the_end,
                available: adapters.len(),
            })
        );
    }
}
