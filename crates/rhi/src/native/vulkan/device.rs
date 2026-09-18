//! Step 2's second half: the logical device and its one queue.
//!
//! # One queue, chosen by rule
//!
//! The retained execution model submits everything on logical queue 0, so exactly
//! one queue family is selected and exactly one queue is created from it. Which
//! family that is cannot be assumed: a driver may report a transfer-only family
//! first, and picking by index would then create a device that cannot rasterize.
//! [`select_queue_family`] states the rule and is pure, so the interesting layouts
//! -- a decoy family in front, no graphics family at all -- are testable without a
//! driver.
//!
//! Whether the chosen family also supports compute is **reported, not required**.
//! Requiring it would refuse a device that can rasterize, and the decision that
//! needs compute is the capability row, which this selection feeds: a graphics
//! family without compute leaves the compute row unproved, and the requirement
//! check refuses the graph that needed it. Selecting here and deciding there keeps
//! one rule in one place.
//!
//! # What is deliberately not enabled
//!
//! No device features, and no device extension on the headless path. Every feature
//! a step needs is enabled by that step, and enabling one without recording the
//! fact it establishes would put an unproved claim into the ledger.
//! `VK_KHR_swapchain` belongs to step 10, so it is enabled only by
//! [`open_with_swapchain`] -- and only after the physical device's own extension
//! inventory was read and positively contained it. A device opened for headless work
//! enables nothing, which is why the two entry points are separate rather than one
//! function with a flag: [`SwapchainDevice`] is the witness that the extension is
//! there, and a headless [`VulkanDevice`] cannot reach a swapchain call at all.

use std::ffi::{CStr, c_char};

use ash::vk;

use crate::common::base::stamp::DeviceStamp;
use crate::common::caps::{
    AdapterLimits, Capability, CapabilityEvidence, CapabilityFact, CapabilityLedger, OperationProbe,
};

use super::instance::{SurfaceInstance, ValidationInstance};
use super::inventory::{EnumerationError, enumerate_device_extensions};

/// The device extension the swapchain path enables.
///
/// Non-NUL-terminated for the same reason [`super::validation::REQUIRED_LAYER`] is:
/// this is the value the inventory comparison reads, while the loader is handed the
/// `CStr` below.
pub(crate) const SWAPCHAIN: &str = "VK_KHR_swapchain";

/// The same name, NUL-terminated for the loader.
///
/// A `c"..."` literal rather than `SWAPCHAIN.as_ptr()`, because `str::as_ptr` does
/// **not** hand the loader a NUL-terminated name. A test asserts the two still
/// spell the same name, which is the one drift the type system cannot express here.
const SWAPCHAIN_C: &CStr = c"VK_KHR_swapchain";

/// Which device extension a physical device did not report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingDeviceExtension {
    /// `VK_KHR_swapchain` was not reported.
    Swapchain,
}

/// Verifies that a physical device's extension inventory can serve a swapchain.
///
/// The comparison is exact, for the same reason [`super::validation::verify_required`]'s
/// is: extension names are case-sensitive, so a near miss is a different extension
/// and enabling it would make the device creation fail rather than report which
/// facility is missing.
pub(crate) fn verify_device_extensions(
    extensions: &[String],
) -> Result<(), MissingDeviceExtension> {
    if extensions.iter().any(|name| name == SWAPCHAIN) {
        Ok(())
    } else {
        Err(MissingDeviceExtension::Swapchain)
    }
}

/// The queue family and queue the device was created with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SelectedQueue {
    /// Index of the selected queue family.
    pub family: u32,
    /// Whether that family also supports compute dispatches.
    pub supports_compute: bool,
}

/// Why the logical device could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeviceError {
    /// The loader refused to report queue families.
    QueueFamilies(vk::Result),
    /// No reported family can rasterize, so no usable device exists.
    NoGraphicsQueue,
    /// The swapchain path was asked for and the extension names could not be read.
    ExtensionEnumeration(EnumerationError),
    /// The physical device does not report `VK_KHR_swapchain`.
    MissingExtension(MissingDeviceExtension),
    /// The driver refused to create the device.
    Creation(vk::Result),
}

/// Selects the one queue family the retained execution model uses.
///
/// The first family whose flags contain `GRAPHICS` is chosen, and its `COMPUTE`
/// bit is reported alongside. `queue_count` is not consulted: the family is the
/// unit the rule is about, and a family reporting zero queues is not something a
/// conformant driver does -- if it did, device creation would fail and report it.
pub(crate) fn select_queue_family(
    families: &[vk::QueueFamilyProperties],
) -> Result<SelectedQueue, DeviceError> {
    families
        .iter()
        .enumerate()
        .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(index, family)| SelectedQueue {
            family: index as u32,
            supports_compute: family.queue_flags.contains(vk::QueueFlags::COMPUTE),
        })
        .ok_or(DeviceError::NoGraphicsQueue)
}

/// A logical device and the single queue created from it.
///
/// Field order is teardown order: the device is destroyed before the instance that
/// created it, which is the caller's field order rather than this type's.
pub(crate) struct VulkanDevice {
    device: ash::Device,
    queue: vk::Queue,
    selected: SelectedQueue,
    ledger: CapabilityLedger,
    stamp: DeviceStamp,
}

impl VulkanDevice {
    /// Returns the device's function table.
    pub(crate) fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Returns the one queue every submission uses.
    pub(crate) fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Returns the family and queue index the device was created with.
    pub(crate) const fn selected_queue(&self) -> SelectedQueue {
        self.selected
    }

    /// Returns this device generation's stamp.
    ///
    /// One device is one generation: this backend has no recovery path yet, so the
    /// first generation is the only one. Every resource id is stamped with this
    /// value, which is what lets a stale id from a replaced device be refused
    /// without consulting any table.
    pub(crate) const fn stamp(&self) -> DeviceStamp {
        self.stamp
    }
}

/// The first backend to implement the common layer's capability query.
///
/// The ledger is captured at device creation and stored, not recomputed on demand:
/// capability is a fact about one opened device, and a query that re-read it could
/// answer about hardware that has since been replaced.
impl crate::common::api::negotiate::CapabilitySource for VulkanDevice {
    fn ledger(&self) -> &CapabilityLedger {
        &self.ledger
    }
}

/// Records what creating this device proved, and nothing else.
///
/// Two rows are established here:
///
/// - **Graphics** by the fact that a device was created on a family whose flags
///   contain graphics. That is the evidence an explicit API offers; it is not a
///   command probe, which is why `Capability::Graphics` does not require one.
/// - **Compute** only where that same family also reports compute. A graphics
///   family without it leaves the row unexamined rather than negative, so the
///   requirement check reports "no route on this device" instead of pretending a
///   failed probe.
///
/// Every other row is absent, because nothing has queried for it yet. Absence is
/// the rejecting value, so no unproved domain can be entered by accident -- and
/// each row is added by the step that actually proves it.
pub(crate) fn ledger(selected: SelectedQueue, limits: &AdapterLimits) -> CapabilityLedger {
    let mut ledger = CapabilityLedger::default();
    ledger.record(
        Capability::Graphics,
        CapabilityFact {
            evidence: Some(CapabilityEvidence::Core),
            limits_satisfied: limits.max_texture_dimension_2d != 0,
            // This backend's proof for graphics is structural: a device exists on
            // a family whose flags contain graphics. No command was needed, which
            // `NotRequired` states -- as distinct from `NotRun`, which would mean
            // the proof is still owed.
            operation_probe: OperationProbe::NotRequired,
        },
    );
    if selected.supports_compute {
        ledger.record(
            Capability::Compute,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                limits_satisfied: limits.supports_compute(),
                // Structural, like graphics: the queue family reported compute and
                // a device was created on it.
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    ledger
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the device; destroying it also
        // destroys the queue, which is not a separately owned object. Nothing else
        // holds a device-level object because nothing else has been created yet.
        unsafe { self.device.destroy_device(None) };
    }
}

impl core::fmt::Debug for VulkanDevice {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("VulkanDevice")
            .field("selected_queue", &self.selected)
            .finish_non_exhaustive()
    }
}

/// A logical device created with `VK_KHR_swapchain` verified and enabled.
///
/// A distinct type rather than a flag on [`VulkanDevice`], because
/// `vkCreateSwapchainKHR` exists only when the extension was enabled and `ash`
/// substitutes a panicking stub for a function the loader did not resolve. So
/// "create a swapchain on a device opened for headless work" must not be
/// expressible: only [`open_with_swapchain`] produces this value, and
/// `super::swapchain::create` accepts only it. A headless device has no path to a
/// swapchain call at all, which is the same witness rule
/// [`super::instance::SurfaceInstance`] states for the surface entry points.
#[derive(Debug)]
pub(crate) struct SwapchainDevice(VulkanDevice);

impl SwapchainDevice {
    /// Returns the logical device the extension was enabled on.
    pub(crate) fn device(&self) -> &VulkanDevice {
        &self.0
    }
}

/// Creates the logical device and its one queue on `adapter`.
///
/// `limits` is the adapter's already-read limit set: the caller has it from step
/// 2's first half, and re-reading the properties here would be a second query for
/// one fact.
pub(crate) fn open(
    instance: &ValidationInstance,
    adapter: vk::PhysicalDevice,
    limits: &AdapterLimits,
) -> Result<VulkanDevice, DeviceError> {
    create(instance.instance(), adapter, limits, &[])
}

/// Creates the logical device with `VK_KHR_swapchain` verified and enabled.
///
/// The order is the behavior, exactly as it is for the validation probe: the
/// physical device's own extension inventory is read and checked **before**
/// `create_device` is reached, so a device that cannot present is refused without
/// having created anything. The instance is a [`SurfaceInstance`] because this path
/// only makes sense beside a surface, and that witness already proves the surface
/// extensions were enabled.
pub(crate) fn open_with_swapchain(
    instance: &SurfaceInstance,
    adapter: vk::PhysicalDevice,
    limits: &AdapterLimits,
) -> Result<SwapchainDevice, DeviceError> {
    let instance = instance.instance().instance();
    let extensions = enumerate_device_extensions(instance, adapter)
        .map_err(DeviceError::ExtensionEnumeration)?;
    verify_device_extensions(&extensions).map_err(DeviceError::MissingExtension)?;
    create(instance, adapter, limits, &[SWAPCHAIN_C]).map(SwapchainDevice)
}

/// The shared order: read the families, select one, create the device and its queue.
fn create(
    instance: &ash::Instance,
    adapter: vk::PhysicalDevice,
    limits: &AdapterLimits,
    enabled_extensions: &[&CStr],
) -> Result<VulkanDevice, DeviceError> {
    // SAFETY: the adapter belongs to this instance, which is still live, and the
    // call only reports facts.
    let families = unsafe { instance.get_physical_device_queue_family_properties(adapter) };
    let selected = select_queue_family(&families)?;

    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected.family)
        .queue_priorities(&priorities);
    // The queue-info slice must outlive the create call, so it is a binding
    // rather than an inline array literal; the same is true of the enabled
    // extension name pointers.
    let queue_infos = [queue_info];
    let extension_pointers: Vec<*const c_char> = enabled_extensions
        .iter()
        .map(|name| name.as_ptr())
        .collect();
    let mut create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
    if !extension_pointers.is_empty() {
        create_info = create_info.enabled_extension_names(&extension_pointers);
    }

    // SAFETY: `create_info` and everything it points at -- the queue info and the
    // extension name array -- are locals that outlive the call; the caller has
    // verified every enabled extension name against the physical device's own
    // inventory; and no allocation callbacks are supplied, which asks for the
    // driver's default. The returned value is the loaded device itself, not a bare
    // handle: `ash` resolves the device-level entry points as part of creation.
    let device = unsafe { instance.create_device(adapter, &create_info, None) }
        .map_err(DeviceError::Creation)?;

    // SAFETY: the queue was created by the device creation above -- exactly one
    // queue in the selected family -- and the device is still live.
    let queue = unsafe { device.get_device_queue(selected.family, 0) };

    Ok(VulkanDevice {
        device,
        queue,
        selected,
        ledger: ledger(selected, limits),
        // The first generation of a freshly identified device. Identity comes from
        // the crate's monotonic counter, which is unique among live devices; the
        // generation advances only when a device is replaced, which this backend
        // does not yet do.
        stamp: DeviceStamp::initial(fluxel_rendergraph::DeviceIdentity::new(crate::next_identity())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
            ..Default::default()
        }
    }

    #[test]
    fn a_graphics_family_behind_a_transfer_family_is_still_selected() {
        // The decoy is the point: selecting by index would create a device that
        // cannot rasterize.
        let families = [
            family(vk::QueueFlags::TRANSFER),
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
        ];
        assert_eq!(
            select_queue_family(&families),
            Ok(SelectedQueue {
                family: 1,
                supports_compute: true
            })
        );
    }

    #[test]
    fn a_graphics_family_without_compute_is_selected_and_reports_no_compute() {
        let families = [family(vk::QueueFlags::GRAPHICS)];
        assert_eq!(
            select_queue_family(&families),
            Ok(SelectedQueue {
                family: 0,
                supports_compute: false
            })
        );
    }

    #[test]
    fn no_graphics_family_refuses_rather_than_selecting_a_transfer_queue() {
        let families = [
            family(vk::QueueFlags::TRANSFER),
            family(vk::QueueFlags::COMPUTE),
        ];
        assert_eq!(
            select_queue_family(&families),
            Err(DeviceError::NoGraphicsQueue)
        );
    }

    #[test]
    fn an_empty_family_list_refuses() {
        assert_eq!(select_queue_family(&[]), Err(DeviceError::NoGraphicsQueue));
    }

    #[test]
    fn the_first_graphics_family_wins() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER),
        ];
        let selected = select_queue_family(&families).expect("a graphics family exists");
        assert_eq!(selected.family, 0);
        assert!(selected.supports_compute);
    }

    #[test]
    fn the_enabled_extension_name_still_spells_the_name_the_probe_compares() {
        // The guard for the one drift this module cannot express in the type
        // system: the loader is handed a NUL-terminated `CStr`, while the probe
        // compares an owned `String`.
        assert_eq!(SWAPCHAIN_C.to_str().expect("ASCII name"), SWAPCHAIN);
    }

    #[test]
    fn a_device_without_the_swapchain_extension_is_refused_by_name() {
        let without = vec!["VK_KHR_portability_subset".to_owned()];
        assert_eq!(
            verify_device_extensions(&without),
            Err(MissingDeviceExtension::Swapchain)
        );
        let with = vec!["VK_KHR_get_physical_device_properties2".to_owned(), SWAPCHAIN.to_owned()];
        assert_eq!(verify_device_extensions(&with), Ok(()));
    }

    #[test]
    fn a_near_miss_on_the_extension_name_does_not_enable_it() {
        // Case and a suffix are different extensions, and enabling one the
        // physical device did not report is exactly the creation failure this
        // check exists to turn into a named refusal.
        let near_miss = vec!["vk_khr_swapchain".to_owned()];
        assert_eq!(
            verify_device_extensions(&near_miss),
            Err(MissingDeviceExtension::Swapchain)
        );
        let prefixed = vec!["VK_KHR_swapchain_extra".to_owned()];
        assert_eq!(
            verify_device_extensions(&prefixed),
            Err(MissingDeviceExtension::Swapchain)
        );
    }

    #[test]
    fn a_real_device_opens_on_the_first_adapter_this_machine_reports() {
        // End-to-end smoke test for step 2: instance, adapter enumeration, queue
        // family selection and device creation against the real driver. It asserts
        // only that the path is total -- a machine with no Vulkan adapter returns
        // before asserting, because having no GPU is not this test's subject.
        use crate::Validation;
        use crate::native::vulkan::{adapter, instance};

        let Ok(instance) = instance::open(Validation::Disabled) else {
            return;
        };
        let Ok(adapters) = adapter::enumerate(instance.instance()) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let index = adapter::select(adapters.len(), 0).expect("index zero exists");
        let facts = adapter::describe(instance.instance(), adapters[index]);
        assert!(!facts.hardware.name.is_empty(), "a driver reports a device name");
        assert!(
            facts.limits.max_texture_dimension_2d > 0,
            "a usable adapter reports a texture extent"
        );

        let device = open(&instance, adapters[index], &facts.limits).expect("a graphics family exists");
        let selected = device.selected_queue();
        assert!(
            selected.family < u32::MAX,
            "the selected family is a real index"
        );

        // The first real backend implementing the common layer's capability query.
        //
        // This asserts the *ledger*, not `require`: `require::<_, Graphics>(&device)`
        // does not compile yet, because `VulkanDevice` has no `Provides<Graphics>`
        // until the family's vocabulary exists. That refusal is the design working
        // -- the type system answers before the ledger is consulted -- and it is why
        // the value half is asserted here on its own.
        use crate::common::api::negotiate::CapabilitySource;
        assert!(device.ledger().supports(Capability::Graphics));
        assert_eq!(
            device.ledger().supports(Capability::Compute),
            selected.supports_compute,
            "compute disagreement: queue={selected:?} limits={:?}",
            facts.limits
        );
        assert!(
            !device.ledger().supports(Capability::StorageBuffer),
            "nothing has proved storage buffers on this device yet"
        );
    }

    #[test]
    fn a_real_device_opens_with_the_swapchain_extension_enabled() {
        // The step 10 device half against the real driver. The test is not vacuous:
        // when the physical device's own inventory reports the extension, opening
        // must succeed -- a creation failure after that check would mean this module
        // enabled a name it had verified but the driver did not accept.
        use crate::Validation;
        use crate::native::vulkan::{adapter, instance, inventory};

        let Ok(surface_instance) = instance::open_with_surface(Validation::Disabled) else {
            return;
        };
        let instance = surface_instance.instance().instance();
        let Ok(adapters) = adapter::enumerate(instance) else {
            return;
        };
        if adapters.is_empty() {
            return;
        }
        let index = adapter::select(adapters.len(), 0).expect("index zero exists");
        let Ok(extensions) = inventory::enumerate_device_extensions(instance, adapters[index])
        else {
            return;
        };
        if !extensions.iter().any(|name| name == SWAPCHAIN) {
            // A physical device without the extension is not this test's subject;
            // its refusal is covered by the pure test above.
            return;
        }
        let facts = adapter::describe(instance, adapters[index]);
        let device = open_with_swapchain(&surface_instance, adapters[index], &facts.limits)
            .expect("the extension was reported, so the device enables it");
        assert!(
            device.device().selected_queue().family < u32::MAX,
            "the selected family is a real index"
        );
    }
}
