//! Step 14's first piece: the RHI-facing open of this backend.
//!
//! [`super::open`] already chains this backend's own steps into one entry point,
//! and [`super::family`] already makes the common contract's negotiation real
//! against a driver. What was still missing is the vocabulary the RHI itself
//! speaks when it opens a device: the public `Device` reports
//! [`HardwareInfo`](crate::HardwareInfo) *and*
//! [`HardwareCapabilities`](crate::HardwareCapabilities), and refuses with
//! [`OpenError`]. This module supplies both halves and nothing else:
//!
//! - [`capabilities`] lowers the facts the device already owns -- the adapter's
//!   numeric report and the per-format evidence table step 11 recorded -- onto the
//!   RHI's capability facts. It asks the driver nothing and cannot fail.
//! - [`open`] is the entry point in the RHI's error vocabulary: it calls
//!   [`super::open::open`] and lowers any failure through [`open_error`], so a
//!   caller of this backend needs to know only the public error type.
//!
//! # What this deliberately is not
//!
//! It does not swap the route `crate::imp::open` takes. The borrowed `wgpu-hal`
//! path still serves `Backend::Vulkan` for the frozen oracle, because replacing it
//! requires the execution verbs -- the staging upload, the fixed-artifact pipeline
//! and binding construction, the diagnostics capture and the public `Device`
//! wiring -- to speak this backend first. Those are the rest of step 14, and
//! switching the route before they exist would refuse work the oracle executes
//! today.
//!
//! It also does not validate a portable baseline limit set. The borrowed path
//! refuses an adapter that cannot meet the default portable limits with
//! `OpenError::RequiredLimitsUnavailable`, and this backend has no equivalent check
//! yet; inventing one here would be a second, unwritten rule about which numbers
//! the RHI's floor is made of. It arrives with the public `Device` wiring, which is
//! where that floor is defined.

use fluxel_rendergraph::TextureFormat;

use crate::common::caps::AdapterLimits;
use crate::common::formats::FormatTable;
use crate::{Backend, HardwareCapabilities, OpenError, Validation};

use super::adapter::AdapterError;
use super::format_facts::SINGLE_SAMPLE;
use super::instance::InstanceError;
use super::open::{self, OpenVulkanError, OpenedVulkan};

/// Lowers this device's discovery onto the RHI's hardware-capability facts.
///
/// The two inputs are one device's whole fact set for these fields: the adapter's
/// numeric report and the per-format evidence table the device owns. Nothing is
/// re-queried, so a caller comparing this value with the ledger the same device
/// recorded is comparing one discovery with itself.
///
/// # The per-format half is read from the table, and absent means false
///
/// The five `rgba8_unorm_*` facts are the two rows step 11 recorded for
/// `Rgba8Unorm` and its sRGB sibling at [`SINGLE_SAMPLE`]. A format no discovery
/// examined has no row, and the public fact is the rejecting `false` -- the same
/// direction the evidence table takes, where an unexamined pair and a proved
/// negative are both unusable but only one of them is an answer.
///
/// `rgba8_unorm_storage_read_enabled` is the one field whose borrowed-path spelling
/// differs, and the difference is the API rather than the fact. The HAL had to
/// request `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` before its per-format answer
/// meant anything, so it reported `read && requested`. `Vulkan` answers
/// `vkGetPhysicalDeviceFormatProperties` for every format on every device with no
/// feature to enable, so the probed row *is* the enabled fact here. Reporting
/// `read && <some feature>` would invent a gate this API does not have.
///
/// # The numeric half is the adapter's report, not a permission
///
/// Every number is copied unchanged, including the compute workgroup counts on a
/// device whose selected family does not report compute. That is deliberate: this
/// struct is the adapter's raw report (see its own doc), and whether a *domain* may
/// be entered is the capability ledger's answer, recorded at device creation and
/// read by `require`. Gating these numbers on a ledger row here would be a second
/// place that decides the same question.
pub(crate) fn capabilities(
    limits: &AdapterLimits,
    formats: &FormatTable,
) -> HardwareCapabilities {
    let rgba8 = formats.get(TextureFormat::Rgba8Unorm, SINGLE_SAMPLE);
    let rgba8_srgb = formats.get(TextureFormat::Rgba8UnormSrgb, SINGLE_SAMPLE);
    let storage_read = rgba8.is_some_and(|facts| facts.storage_read);
    HardwareCapabilities {
        rgba8_unorm_filterable: rgba8.is_some_and(|facts| facts.filterable),
        rgba8_unorm_srgb_filterable: rgba8_srgb.is_some_and(|facts| facts.filterable),
        rgba8_unorm_storage_read: storage_read,
        rgba8_unorm_storage_write: rgba8.is_some_and(|facts| facts.storage_write),
        // The probed fact is the enabled fact: see the module and function docs.
        rgba8_unorm_storage_read_enabled: storage_read,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_bind_groups: limits.max_bind_groups,
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_compute_workgroup_size: limits.max_compute_workgroup_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
    }
}

/// Opens this backend in the RHI's own error vocabulary.
///
/// This is the one call the public `Device` will make once the execution layer
/// speaks this backend: [`super::open::open`] does the work and [`open_error`]
/// answers the caller in public terms.
pub(crate) fn open(
    validation: Validation,
    adapter_index: usize,
) -> Result<OpenedVulkan, OpenError> {
    open::open(validation, adapter_index).map_err(open_error)
}

/// Lowers one open failure onto the public [`OpenError`].
///
/// Two of the native sentences already have an exact public counterpart and are
/// mapped rather than flattened, because a caller acts on them differently:
///
/// - a validation facility that could not be verified is
///   [`OpenError::ValidationUnavailable`], which is the fail-closed refusal the
///   plan's preserved-semantics table fixes;
/// - an absent adapter index is [`OpenError::AdapterUnavailable`], which carries
///   both numbers so the caller can tell "no such index" from "no adapter".
///
/// Everything else keeps its own diagnostic in [`OpenError::NativeUnavailable`].
/// The reason is the native error's `Debug` rendering, prefixed with the layer that
/// produced it, because the inner types are structured values -- a `vk::Result`, a
/// missing-layer name, a missing device extension -- and flattening them into a
/// prose sentence here would throw away the one thing a diagnostic needs. No
/// variant is invented and no native error is folded into a neighbouring one.
pub(crate) fn open_error(error: OpenVulkanError) -> OpenError {
    match error {
        OpenVulkanError::Instance(InstanceError::ValidationMissing(_)) => {
            OpenError::ValidationUnavailable {
                backend: Backend::Vulkan,
            }
        }
        OpenVulkanError::Adapter(AdapterError::Unavailable { index, available }) => {
            OpenError::AdapterUnavailable {
                backend: Backend::Vulkan,
                adapter_index: index,
                available_adapters: available,
            }
        }
        OpenVulkanError::Instance(reason) => OpenError::NativeUnavailable {
            backend: Backend::Vulkan,
            reason: format!("instance: {reason:?}"),
        },
        OpenVulkanError::Adapter(reason) => OpenError::NativeUnavailable {
            backend: Backend::Vulkan,
            reason: format!("adapter: {reason:?}"),
        },
        OpenVulkanError::Device(reason) => OpenError::NativeUnavailable {
            backend: Backend::Vulkan,
            reason: format!("device: {reason:?}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::formats::{FormatCapabilities, FormatEvidence};

    /// One evidence row with every fact off unless a test turns it on.
    fn row(format: TextureFormat) -> FormatCapabilities {
        FormatCapabilities {
            format,
            sample_count: SINGLE_SAMPLE,
            evidence: FormatEvidence::OperationProbed,
            sampled: false,
            filterable: false,
            renderable: false,
            blendable: false,
            storage_read: false,
            storage_write: false,
            copy_source: false,
            copy_destination: false,
        }
    }

    /// A table holding exactly the rows a test recorded.
    fn table(rows: &[FormatCapabilities]) -> FormatTable {
        let mut table = FormatTable::default();
        for entry in rows {
            table.record(*entry).expect("the fixture rows do not disagree");
        }
        table
    }

    /// Limits whose every number is a distinguishable non-default value.
    fn limits() -> AdapterLimits {
        AdapterLimits {
            max_texture_dimension_2d: 16_384,
            max_bind_groups: 8,
            min_uniform_buffer_offset_alignment: 64,
            min_storage_buffer_offset_alignment: 32,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_compute_workgroups_per_dimension: [65_535, 65_535, 65_535],
            max_compute_workgroup_size: [1024, 1024, 64],
            max_compute_invocations_per_workgroup: 1024,
            ..AdapterLimits::unavailable()
        }
    }

    #[test]
    fn a_device_that_proved_no_format_reports_the_rejecting_rgba8_facts() {
        // The fail-closed direction: an unexamined table and unavailable limits
        // produce every boolean false and every number zero rather than a default.
        let capabilities = self::capabilities(&AdapterLimits::unavailable(), &FormatTable::default());

        assert!(!capabilities.rgba8_unorm_filterable);
        assert!(!capabilities.rgba8_unorm_srgb_filterable);
        assert!(!capabilities.rgba8_unorm_storage_read);
        assert!(!capabilities.rgba8_unorm_storage_write);
        assert!(!capabilities.rgba8_unorm_storage_read_enabled);
        assert_eq!(capabilities.max_texture_dimension_2d, 0);
        assert_eq!(capabilities.max_bind_groups, 0);
        assert_eq!(capabilities.max_storage_buffer_binding_size, 0);
        assert_eq!(
            capabilities.max_compute_workgroups_per_dimension,
            [0; 3]
        );
    }

    #[test]
    fn the_rgba8_facts_come_from_the_two_rows_the_table_proved() {
        let mut unorm = row(TextureFormat::Rgba8Unorm);
        unorm.filterable = true;
        unorm.storage_read = true;
        // A write fact the table would accept: its evidence is the probe above.
        unorm.storage_write = true;
        let srgb = row(TextureFormat::Rgba8UnormSrgb);

        let capabilities = self::capabilities(&limits(), &table(&[unorm, srgb]));

        assert!(capabilities.rgba8_unorm_filterable);
        assert!(capabilities.rgba8_unorm_storage_read);
        assert!(capabilities.rgba8_unorm_storage_write);
        assert!(
            !capabilities.rgba8_unorm_srgb_filterable,
            "the sRGB row was recorded and proved no filtering: a proved negative, not an absent row"
        );
    }

    #[test]
    fn the_srgb_fact_is_read_from_its_own_row_and_never_inherited() {
        // The retained linear-clamp recipe's legality depends on sRGB filterability
        // being a separate adapter fact; the unorm row must not supply it.
        let mut unorm = row(TextureFormat::Rgba8Unorm);
        unorm.filterable = true;
        let capabilities = self::capabilities(&limits(), &table(&[unorm]));
        assert!(capabilities.rgba8_unorm_filterable);
        assert!(
            !capabilities.rgba8_unorm_srgb_filterable,
            "a different format's fact is absent, not copied"
        );
    }

    #[test]
    fn the_storage_read_enabled_fact_is_the_probed_fact() {
        // The HAL needed a feature request before its per-format answer counted;
        // this API does not, so the two fields agree by construction. Pinning it
        // means a later edit cannot reintroduce a gate this layer has no switch for.
        let plain = row(TextureFormat::Rgba8Unorm);
        let capabilities = self::capabilities(&limits(), &table(&[plain]));
        assert!(!capabilities.rgba8_unorm_storage_read);
        assert!(!capabilities.rgba8_unorm_storage_read_enabled);

        let mut storing = row(TextureFormat::Rgba8Unorm);
        storing.storage_read = true;
        let capabilities = self::capabilities(&limits(), &table(&[storing]));
        assert!(capabilities.rgba8_unorm_storage_read);
        assert!(capabilities.rgba8_unorm_storage_read_enabled);
    }

    #[test]
    fn the_numeric_facts_are_the_adapter_report_unchanged() {
        let capabilities = self::capabilities(&limits(), &FormatTable::default());

        assert_eq!(capabilities.max_texture_dimension_2d, 16_384);
        assert_eq!(capabilities.max_bind_groups, 8);
        assert_eq!(capabilities.min_uniform_buffer_offset_alignment, 64);
        assert_eq!(capabilities.min_storage_buffer_offset_alignment, 32);
        assert_eq!(
            capabilities.max_storage_buffer_binding_size,
            128 * 1024 * 1024
        );
        assert_eq!(
            capabilities.max_compute_workgroups_per_dimension,
            [65_535, 65_535, 65_535]
        );
        assert_eq!(capabilities.max_compute_workgroup_size, [1024, 1024, 64]);
        assert_eq!(capabilities.max_compute_invocations_per_workgroup, 1024);
    }

    #[test]
    fn a_missing_validation_facility_is_the_public_validation_refusal() {
        use super::super::validation::MissingValidation;

        // The preserved semantics: `Required` that cannot be verified is refused
        // *before* an instance exists, and the public sentence names the facility.
        assert_eq!(
            open_error(OpenVulkanError::Instance(InstanceError::ValidationMissing(
                MissingValidation::Layer
            ))),
            OpenError::ValidationUnavailable {
                backend: Backend::Vulkan
            }
        );
    }

    #[test]
    fn an_absent_adapter_index_is_the_public_adapter_refusal() {
        assert_eq!(
            open_error(OpenVulkanError::Adapter(AdapterError::Unavailable {
                index: 3,
                available: 2,
            })),
            OpenError::AdapterUnavailable {
                backend: Backend::Vulkan,
                adapter_index: 3,
                available_adapters: 2,
            }
        );
    }

    #[test]
    fn every_other_failure_keeps_a_diagnostic_and_the_backend_name() {
        // Neither an instance that could not be created nor a device that could not
        // be created is flattened into a neighbouring public variant, because the
        // caller has no other way to learn which layer refused.
        let instance = open_error(OpenVulkanError::Instance(InstanceError::Creation(
            ash::vk::Result::ERROR_INITIALIZATION_FAILED,
        )));
        match instance {
            OpenError::NativeUnavailable { backend, reason } => {
                assert_eq!(backend, Backend::Vulkan);
                assert!(
                    reason.contains("ERROR_INITIALIZATION_FAILED"),
                    "the driver's own result survives into the diagnostic: {reason}"
                );
            }
            other => panic!("a creation failure is not {other:?}"),
        }

        let device = open_error(OpenVulkanError::Device(
            super::super::device::DeviceError::NoGraphicsQueue,
        ));
        match device {
            OpenError::NativeUnavailable { backend, reason } => {
                assert_eq!(backend, Backend::Vulkan);
                assert!(
                    reason.contains("NoGraphicsQueue"),
                    "the device layer's own sentence survives: {reason}"
                );
            }
            other => panic!("a queue refusal is not {other:?}"),
        }
    }

    #[test]
    fn a_real_device_opens_through_the_rhi_facing_entry_with_both_fact_sets() {
        // A machine with no Vulkan loader or adapter returns before asserting,
        // because having no GPU is not this test's subject. The facts asserted
        // below are the ones `Vulkan` makes mandatory for R8G8B8A8_UNORM with
        // optimal tiling, so a failure is a real disagreement rather than an
        // optional capability this board happens to lack.
        let Ok(opened) = open(Validation::Disabled, 0) else {
            return;
        };

        assert_eq!(opened.hardware().backend, Backend::Vulkan);
        assert!(!opened.hardware().name.is_empty());

        let capabilities = opened.capabilities();
        assert_eq!(
            capabilities.max_texture_dimension_2d,
            opened.limits.max_texture_dimension_2d,
            "the numeric half is the same limit set the device was opened against"
        );
        assert!(
            capabilities.rgba8_unorm_filterable,
            "R8G8B8A8_UNORM is mandatory with SAMPLED_IMAGE_FILTER_LINEAR under optimal tiling"
        );
        assert_eq!(
            capabilities.rgba8_unorm_storage_read,
            opened
                .device
                .formats()
                .get(TextureFormat::Rgba8Unorm, SINGLE_SAMPLE)
                .is_some_and(|facts| facts.storage_read),
            "the public storage fact is the device's own table answer, not a re-query"
        );
        assert!(
            !opened.validation_enabled(),
            "Validation::Disabled enables and verifies nothing"
        );
    }
}
