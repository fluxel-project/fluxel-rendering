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
//! # Which features are enabled, and why so few
//!
//! One step of eleven needs a device feature so far, and this is where it is
//! enabled: the shader-store pair [`super::features`] requests, which is what proves
//! the two storage rows. Every other feature a step needs will be enabled by that
//! step, because enabling one without recording the fact it establishes would put an
//! unproved claim into the ledger -- and enabling one the adapter never reported is
//! not a soft failure but a `vkCreateDevice` error, so the requested set is always
//! the adapter's own report narrowed to what this backend has been taught.
//!
//! # The device owns the format table its storage-image row reads
//!
//! The storage-image row needs a per-format fact beside the feature pair, so
//! [`create`] queries every format this backend maps through
//! [`super::format_facts::record_mapped`] and stores the resulting table beside the
//! ledger. Reading it here rather than leaving it to a caller is what lets the row
//! be recorded at all: the ledger is captured at creation and never recomputed, so a
//! fact discovered later could not reach it. It also keeps one discovery -- the
//! format evidence the capability lowering folds is the same table the ledger read.
//!
//! # Which device extensions are enabled
//!
//! `VK_KHR_swapchain` belongs to step 10, so it is enabled only by
//! [`open_with_swapchain`] -- and only after the physical device's own extension
//! inventory was read and positively contained it. That is why the two entry points
//! are separate rather than one function with a flag: [`SwapchainDevice`] is the
//! witness that the extension is there, and a headless [`VulkanDevice`] cannot reach
//! a swapchain call at all.
//!
//! The one extension **both** paths enable is `VK_KHR_maintenance1`, and it is not a
//! convenience: step 12 lowers the dynamic viewport with the borrowed path's Y flip
//! (`y + height`, a negative height), and a negative viewport height is a validation
//! error on a `Vulkan` 1.0 device unless that extension is enabled. The instance
//! this backend requests is 1.0, so the extension is the only route to the geometry
//! the frozen oracle renders. Every other extension still arrives with the step that
//! proves a row with it.
//!
//! Both are verified against the physical device's report before `vkCreateDevice`,
//! for the reason the validation probe already states: enabling an extension the
//! adapter never reported is a creation failure with no diagnosis, while a refusal
//! names the missing facility.

use std::ffi::{CStr, c_char};

use ash::vk;

use crate::common::base::stamp::DeviceStamp;
use crate::common::caps::{
    AdapterLimits, Capability, CapabilityEvidence, CapabilityFact, CapabilityLedger, OperationProbe,
};
use crate::common::formats::FormatTable;

use super::allocator::GpuAllocator;
use super::command::{CommandPool, RecordError};
use super::features::{self, StoreFeatures};
use super::format_facts::{self, FormatError};
use super::instance::{SurfaceInstance, ValidationInstance};
use super::inventory::{EnumerationError, enumerate_device_extensions};
use super::resource::ResourceTable;

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

/// The device extension that makes a negative viewport height legal.
///
/// Non-NUL-terminated for the same reason [`SWAPCHAIN`] is: this is the value the
/// inventory comparison reads, while the loader is handed the `CStr` below.
pub(crate) const MAINTENANCE1: &str = "VK_KHR_maintenance1";

/// The same name, NUL-terminated for the loader.
///
/// A `c"..."` literal rather than `MAINTENANCE1.as_ptr()`, because `str::as_ptr`
/// does **not** hand the loader a NUL-terminated name. A test asserts the two still
/// spell the same name, which is the one drift the type system cannot express here.
const MAINTENANCE1_C: &CStr = c"VK_KHR_maintenance1";

/// Which device extension a physical device did not report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingDeviceExtension {
    /// `VK_KHR_swapchain` was not reported.
    Swapchain,
    /// `VK_KHR_maintenance1` was not reported.
    Maintenance1,
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

/// Verifies that a physical device's extension inventory carries
/// `VK_KHR_maintenance1`.
///
/// The same exact, case-sensitive comparison as the swapchain check, and for the
/// same reason: the extension is what makes the negative viewport height step 12
/// records legal, so a near miss is a different extension and enabling it would make
/// device creation fail rather than name the missing facility.
pub(crate) fn verify_maintenance1(
    extensions: &[String],
) -> Result<(), MissingDeviceExtension> {
    if extensions.iter().any(|name| name == MAINTENANCE1) {
        Ok(())
    } else {
        Err(MissingDeviceExtension::Maintenance1)
    }
}

/// The queue family and queue the device was created with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SelectedQueue {
    /// Index of the selected queue family.
    pub family: u32,
    /// Whether that family also supports compute dispatches.
    pub supports_compute: bool,
    /// The family's own timestamp-valid-bit report.
    ///
    /// The specification fixes this as either zero or a value in `36..=64`, so a
    /// non-zero report is exactly "this family can write timestamps" and a zero one
    /// is exactly "it cannot". It is kept as the driver's number rather than as a
    /// boolean so the fact stays the report it is, and it is carried beside
    /// [`Self::supports_compute`] because it is a fact about the same family rather
    /// than a second query.
    pub timestamp_valid_bits: u32,
}

impl SelectedQueue {
    /// Whether this family's timestamp-valid-bit report proves the query row.
    ///
    /// `> 0` rather than `>= 36`: the specification defines no value between one and
    /// thirty-five, so the two spellings agree on every conformant driver while the
    /// `> 0` form also refuses a driver that reported a non-zero value it never
    /// defined. The borrowed `Vulkan` path being replaced spells the same rule as
    /// `timestamp_valid_bits >= 36`.
    pub(crate) const fn supports_timestamps(&self) -> bool {
        self.timestamp_valid_bits != 0
    }
}

/// Why the logical device could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeviceError {
    /// The loader refused to report queue families.
    QueueFamilies(vk::Result),
    /// No reported family can rasterize, so no usable device exists.
    NoGraphicsQueue,
    /// The extension names could not be read from the physical device.
    ///
    /// Both entry points read the inventory -- the swapchain path to verify
    /// `VK_KHR_swapchain`, the headless path to verify `VK_KHR_maintenance1` -- so
    /// an enumeration failure is not specific to one of them.
    ExtensionEnumeration(EnumerationError),
    /// The physical device does not report `VK_KHR_swapchain`.
    MissingExtension(MissingDeviceExtension),
    /// The driver refused to create the device.
    Creation(vk::Result),
    /// A format this backend maps could not be queried or recorded.
    ///
    /// The per-format evidence the storage-image row is read from is discovered
    /// before `vkCreateDevice` is reached, so a driver that cannot answer for a
    /// mapped format refuses the open with nothing created.
    Format(FormatError),
    /// `gpu-allocator` refused to build the suballocator for this device.
    ///
    /// The inner reason is `gpu_allocator`'s `AllocationError`, which is neither
    /// `Copy` nor `PartialEq` and carries a `String`, so it is not carried here --
    /// the same flattening `ResourceError::Memory` already performs at the
    /// allocation boundary, and for the same reason: this error is compared in
    /// tests, and widening it to a foreign non-`Copy` type would make every use
    /// site pay for one diagnostic.
    Allocator,
    /// The one command pool could not be created on the selected queue family.
    CommandPool(RecordError),
}

/// Selects the one queue family the retained execution model uses.
///
/// The first family whose flags contain `GRAPHICS` is chosen, and its `COMPUTE`
/// bit and `timestamp_valid_bits` report are carried alongside. Those two are
/// **reported, not required**: the selection rule is only "this family rasterizes",
/// and a fact this family happens to report is what the ledger records rather than
/// something the choice depends on. `queue_count` is not consulted: the family is
/// the unit the rule is about, and a family reporting zero queues is not something
/// a conformant driver does -- if it did, device creation would fail and report it.
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
            timestamp_valid_bits: family.timestamp_valid_bits,
        })
        .ok_or(DeviceError::NoGraphicsQueue)
}

/// The logical device handle, with destruction owned by the field that holds it.
///
/// Splitting the handle out of [`VulkanDevice`] is what lets field order express the
/// dependency `Vulkan` requires. A type with a manual `Drop` runs that body *before*
/// its fields are dropped, so a `destroy_device` written there would run before the
/// resource table and the command pool that must be released first. As a field, the
/// destruction is the last thing that happens, after every sibling.
struct OwnedDevice(ash::Device);

impl OwnedDevice {
    /// Returns the function table and handle every call site uses.
    fn get(&self) -> &ash::Device {
        &self.0
    }
}

impl Drop for OwnedDevice {
    fn drop(&mut self) {
        // SAFETY: this is the only owner of the device. Every child object of this
        // device is a field of the containing `VulkanDevice` declared before this
        // one, so it has already been released; the other holders of `ash::Device`
        // are clones, which copy a handle and a function table rather than making a
        // second ownership claim. Destroying the device also destroys the queue,
        // which is not a separately owned object.
        unsafe { self.0.destroy_device(None) };
    }
}

/// A logical device and the one queue created from it.
///
/// # The device owns the table and the pool, and the field order says so
///
/// A device owns its resources and the one recording that consumes them:
/// [`ResourceTable`] holds the handles, the image views and the suballocations, and
/// [`CommandPool`] holds the family its command buffers are submittable to. Both are
/// built here rather than by a caller because the capability layer reaches them
/// through the device -- `Provides::provide` takes only `&self`, so a family handle
/// can resolve a [`BufferId`](crate::common::base::resource::BufferId) no other way
/// -- and because a second owner of either would be a second answer about one
/// device.
///
/// Field order is teardown order, and this is the one place it is load bearing: the
/// table releases every allocation and destroys every handle, the pool destroys the
/// command pool, and only then is the device itself destroyed. `OwnedDevice` is what
/// makes that ordering hold; see its own docs.
pub(crate) struct VulkanDevice {
    /// Every resource this device owns, over one suballocator.
    table: ResourceTable,
    /// The one command pool, created on `selected.family`.
    pool: CommandPool,
    /// The device handle, destroyed last.
    device: OwnedDevice,
    queue: vk::Queue,
    selected: SelectedQueue,
    stores: StoreFeatures,
    formats: FormatTable,
    ledger: CapabilityLedger,
    stamp: DeviceStamp,
}

impl VulkanDevice {
    /// Returns the device's function table.
    pub(crate) fn device(&self) -> &ash::Device {
        self.device.get()
    }

    /// Returns the table that owns every resource of this device generation.
    pub(crate) const fn table(&self) -> &ResourceTable {
        &self.table
    }

    /// Returns the table for mutation, which is how a resource is created or
    /// destroyed.
    pub(crate) fn table_mut(&mut self) -> &mut ResourceTable {
        &mut self.table
    }

    /// Returns the one command pool of this device generation.
    ///
    /// The pool is the device's, so a family handle created through
    /// `crate::common::api::negotiate::require` begins its recording from here
    /// without the caller having to own a pool at all.
    pub(crate) const fn pool(&self) -> &CommandPool {
        &self.pool
    }

    /// Returns the one queue every submission uses.
    pub(crate) fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Returns the family and queue index the device was created with.
    pub(crate) const fn selected_queue(&self) -> SelectedQueue {
        self.selected
    }

    /// Returns the shader-store features this device was created with.
    ///
    /// The *enabled* facts, not a second reading of the adapter's report: this is
    /// the value the storage-buffer row was recorded from, so a caller comparing the
    /// two is comparing the ledger with its own input rather than with what the
    /// driver would say now.
    pub(crate) const fn stores(&self) -> StoreFeatures {
        self.stores
    }

    /// Returns the per-format evidence this device's discovery recorded.
    ///
    /// This is the same table the ledger's storage-image row was read from and the
    /// same one the capability lowering folds, so a caller has one discovery answer
    /// rather than a second query to keep in step with it.
    pub(crate) const fn formats(&self) -> &FormatTable {
        &self.formats
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
/// Every row is proved by a fact that was already read rather than by a command:
///
/// - **Graphics** by the fact that a device was created on a family whose flags
///   contain graphics. That is the evidence an explicit API offers; it is not a
///   command probe, which is why `Capability::Graphics` does not require one.
/// - **Compute** only where that same family also reports compute. A graphics
///   family without it leaves the row unexamined rather than negative, so the
///   requirement check reports "no route on this device" instead of pretending a
///   failed probe.
/// - **Copy** from the API version itself: `Vulkan` 1.0 guarantees
///   `vkCmdCopyBuffer` and `vkCmdCopyImage` on a graphics family, step 8 records
///   both routes, and no device feature gates either. The same structural proof as
///   graphics, and the same `NotRequired` probe outcome.
/// - **IndirectDispatch** only beside a proved compute row. `vkCmdDispatchIndirect`
///   is core `Vulkan` 1.0 and the borrowed path being replaced calls it
///   unconditionally, so its proof is structural too -- but an indirect dispatch is
///   still a dispatch, so it keeps the compute row's own numeric floor.
/// - **TimestampQuery** only where the family's `timestamp_valid_bits` report is
///   non-zero. That fact was read with the same `queue_family_properties` call the
///   selection already made, so the row costs no query of its own.
/// - **OcclusionQuery** from the same structural fact graphics has: a core
///   `Vulkan` 1.0 occlusion query on a graphics queue. The row names the imprecise
///   answer, so the `occlusionQueryPrecise` feature this device leaves disabled is
///   not part of its proof.
/// - **ElapsedQuery** wherever the timestamp row is proved, because an elapsed
///   interval here is two timestamps and their difference rather than a separate
///   query type. It is not recorded where the family reported no usable timestamps,
///   which is the one fact that would make a claimed duration unmeasurable.
/// - **StorageBuffer** only where the device was created with the shader-store pair
///   [`StoreFeatures::proves_storage_buffers`] names. The row describes a buffer a
///   shader may read *and* write, and `Vulkan` gates the write half per stage, so a
///   device created with one feature of the pair serves one stage and not the other
///   -- which is a refusal, not half a row.
/// - **StorageImage** only where that same pair was enabled **and** the format table
///   proved a format with both storage directions. The pair is the stage half of the
///   proof; the resource half is the per-format answer, because whether a format may
///   be used through a storage image at all is what the driver reports per format.
///   That fact is this row's numeric floor, so a device created with the pair on an
///   adapter whose formats none support storage is *examined and refused* rather
///   than left unexamined -- a difference the ledger keeps for diagnostics.
/// - **BaseVertex** and **FirstInstance** from the core `Vulkan` 1.0 draw
///   parameters: an indexed draw's `vertexOffset` and a draw's `firstInstance` are
///   part of `vkCmdDraw` / `vkCmdDrawIndexed` rather than a device feature, and step
///   12 records both commands on the graphics family this device was created on.
///   The `drawIndirectFirstInstance` feature gates only the *indirect* form, which
///   the unproved `IndirectDraw` row would name, so neither row claims it. Like copy
///   and occlusion, neither has a numeric floor of its own.
///
/// Every other row is absent, because nothing has proved it. Absence is the
/// rejecting value, so no unproved domain can be entered by accident -- and each
/// row is added by the step that actually proves it. The rows that are owed and
/// deliberately absent here are named in the module's step 11 entry.
pub(crate) fn ledger(
    selected: SelectedQueue,
    limits: &AdapterLimits,
    stores: StoreFeatures,
    formats: &FormatTable,
) -> CapabilityLedger {
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
    // Copy has no numeric floor of its own -- the API version is the whole proof --
    // so its floor is trivially satisfied rather than borrowed from a neighbouring
    // limit, which would tie the row to a fact it does not depend on.
    ledger.record(
        Capability::Copy,
        CapabilityFact {
            evidence: Some(CapabilityEvidence::Core),
            limits_satisfied: true,
            operation_probe: OperationProbe::NotRequired,
        },
    );
    // The two draw-parameter rows are the same structural proof copy has: the base
    // offset and the first instance are core `Vulkan` 1.0 parameters of the draw
    // commands step 12 records, not device features. They are recorded as two rows
    // because they are two families (plan section 20.1) -- one device may prove one
    // without the other -- and neither borrows a numeric floor it does not read.
    for row in [Capability::BaseVertex, Capability::FirstInstance] {
        ledger.record(
            row,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                limits_satisfied: true,
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    // Occlusion is a core `Vulkan` 1.0 query type on a graphics queue, and the row
    // describes the *imprecise* answer -- "did any sample pass" -- which is the
    // answer the core type gives. The exact sample count is the
    // `occlusionQueryPrecise` feature this device does not enable, so claiming the
    // row without it would claim the feature rather than the query. Its proof is the
    // same structural fact graphics has: a device exists on a family whose flags
    // contain graphics. Like copy, it has no numeric floor of its own.
    ledger.record(
        Capability::OcclusionQuery,
        CapabilityFact {
            evidence: Some(CapabilityEvidence::Core),
            limits_satisfied: true,
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
        ledger.record(
            Capability::IndirectDispatch,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                // The compute row's floor, not a second one: an indirect dispatch
                // is a dispatch whose count arrives from a buffer, so the numbers a
                // direct dispatch needs are the numbers this needs too.
                limits_satisfied: limits.supports_compute(),
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    if selected.supports_timestamps() {
        ledger.record(
            Capability::TimestampQuery,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                limits_satisfied: true,
                operation_probe: OperationProbe::NotRequired,
            },
        );
        // An elapsed interval on `Vulkan` is two timestamp writes and their
        // difference, so the family's own valid-bit report is this row's route as
        // well -- there is no separate elapsed query type to ask about. The two rows
        // are still recorded separately because the ledger models a dedicated
        // elapsed facility (the GL family's `TIME_ELAPSED`) that this API reaches
        // through its timestamp domain; the repetition is honest and is what W4 will
        // look at once a second backend proves the same pair. Like the timestamp row
        // it has no numeric floor of its own.
        ledger.record(
            Capability::ElapsedQuery,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                limits_satisfied: true,
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    if stores.proves_storage_buffers() {
        ledger.record(
            Capability::StorageBuffer,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                // The binding size the driver reported is the row's numeric floor: a
                // device whose maximum storage-buffer range is zero can bind none,
                // and the ledger must not record a reachable domain on numbers that
                // reject it.
                limits_satisfied: limits.max_storage_buffer_binding_size != 0,
                // Structural: the device was created with both write features
                // enabled, and a feature enabled at creation is not something a
                // conformant driver refuses per command. A run-time probe would be a
                // second proof of a fact the create-info already states.
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    // The same stage-gated pair gates the image row; what differs is its resource
    // half. `Vulkan`'s store features are about storage *operations* rather than
    // about buffers, so the image row reuses the pair and takes its numeric floor
    // from the per-format fact the format table proved.
    if stores.proves_storage_buffers() {
        ledger.record(
            Capability::StorageImage,
            CapabilityFact {
                evidence: Some(CapabilityEvidence::Core),
                // The resource floor, not a size limit: a device whose formats none
                // support storage has no image to bind through this domain, and the
                // ledger must not record a reachable domain on a fact that rejects
                // it. A table that proved nothing leaves the floor unsatisfied for
                // the same reason a zero binding size does.
                limits_satisfied: formats.has_storage_read_write(),
                // Structural, like the buffer row: an enabled feature and a format
                // the driver reported are not facts a conformant driver contradicts
                // per command, so a run-time probe would be a second proof.
                operation_probe: OperationProbe::NotRequired,
            },
        );
    }
    ledger
}

impl core::fmt::Debug for VulkanDevice {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("VulkanDevice")
            .field("selected_queue", &self.selected)
            .field("stores", &self.stores)
            .field("table", &self.table)
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
/// one fact. `VK_KHR_maintenance1` is read from the adapter's own inventory and
/// verified **before** [`create`] is reached, so an adapter that cannot make the
/// draw path's negative viewport height legal is refused with nothing created.
pub(crate) fn open(
    instance: &ValidationInstance,
    adapter: vk::PhysicalDevice,
    limits: &AdapterLimits,
) -> Result<VulkanDevice, DeviceError> {
    let instance = instance.instance();
    let extensions = enumerate_device_extensions(instance, adapter)
        .map_err(DeviceError::ExtensionEnumeration)?;
    verify_maintenance1(&extensions).map_err(DeviceError::MissingExtension)?;
    create(instance, adapter, limits, &[MAINTENANCE1_C])
}

/// Creates the logical device with `VK_KHR_swapchain` verified and enabled.
///
/// The order is the behavior, exactly as it is for the validation probe: the
/// physical device's own extension inventory is read and checked **before**
/// `create_device` is reached, so a device that cannot present is refused without
/// having created anything. `VK_KHR_maintenance1` is verified against the same
/// inventory and enabled beside it, because the draw path's viewport needs it on
/// every device this backend opens. The instance is a [`SurfaceInstance`] because
/// this path only makes sense beside a surface, and that witness already proves the
/// surface extensions were enabled.
pub(crate) fn open_with_swapchain(
    instance: &SurfaceInstance,
    adapter: vk::PhysicalDevice,
    limits: &AdapterLimits,
) -> Result<SwapchainDevice, DeviceError> {
    let instance = instance.instance().instance();
    let extensions = enumerate_device_extensions(instance, adapter)
        .map_err(DeviceError::ExtensionEnumeration)?;
    verify_device_extensions(&extensions).map_err(DeviceError::MissingExtension)?;
    verify_maintenance1(&extensions).map_err(DeviceError::MissingExtension)?;
    create(
        instance,
        adapter,
        limits,
        &[SWAPCHAIN_C, MAINTENANCE1_C],
    )
    .map(SwapchainDevice)
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

    // SAFETY: the adapter belongs to this instance, which is still live; the call
    // creates nothing and reports the adapter's own feature set.
    let reported_features = unsafe { instance.get_physical_device_features(adapter) };
    // The requested set is the adapter's report narrowed to the features this
    // backend has been taught. It is bound here rather than inline because the
    // create-info borrows it, and the facts the ledger reads are read off the
    // *requested* value so the row cannot be recorded from a second reading of the
    // adapter.
    let requested_features = features::request(&reported_features);
    let stores = features::store(&requested_features);

    // The per-format evidence the storage-image row's floor is read from. It is read
    // before `vkCreateDevice` is reached -- the query needs only the physical device
    // -- so a driver that cannot answer for a mapped format refuses the open with
    // nothing created, the same fail-closed order the validation probe uses.
    let mut formats = FormatTable::default();
    format_facts::record_mapped(instance, adapter, &mut formats).map_err(DeviceError::Format)?;

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
    let mut create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_features(&requested_features);
    if !extension_pointers.is_empty() {
        create_info = create_info.enabled_extension_names(&extension_pointers);
    }

    // SAFETY: `create_info` and everything it points at -- the queue info, the
    // extension name array and the feature set -- are locals that outlive the call;
    // the caller has verified every enabled extension name against the physical
    // device's own inventory; the feature set contains only features the adapter
    // reported, so the driver cannot refuse it with `FEATURE_NOT_PRESENT`; and no
    // allocation callbacks are supplied, which asks for the driver's default. The
    // returned value is the loaded device itself, not a bare handle: `ash` resolves
    // the device-level entry points as part of creation.
    let device = unsafe { instance.create_device(adapter, &create_info, None) }
        .map_err(DeviceError::Creation)?;

    // SAFETY: the queue was created by the device creation above -- exactly one
    // queue in the selected family -- and the device is still live.
    let queue = unsafe { device.get_device_queue(selected.family, 0) };

    // The first generation of a freshly identified device. Identity comes from the
    // crate's monotonic counter, which is unique among live devices; the generation
    // advances only when a device is replaced, which this backend does not yet do.
    let stamp =
        DeviceStamp::initial(fluxel_rendergraph::DeviceIdentity::new(crate::next_identity()));

    // The suballocator and the table it serves are built here, from the raw handles
    // rather than from a `VulkanDevice` borrow: this is the scope in which the device
    // value does not exist yet, and the table has to be *in* it. Both failures below
    // happen after `vkCreateDevice`, so each destroys the device it was handed --
    // there is no owner to inherit it.
    let allocator = match GpuAllocator::from_handles(instance, &device, adapter) {
        Ok(allocator) => allocator,
        Err(_) => {
            // SAFETY: the device was created just above and has no child object yet,
            // so destroying it is the whole teardown of this refused open.
            unsafe { device.destroy_device(None) };
            return Err(DeviceError::Allocator);
        }
    };
    let table = ResourceTable::new(&device, stamp, allocator);
    // The one command pool of this device generation, on the family the rule above
    // selected: a command buffer is submittable only to a queue of its pool's family,
    // so the pool is where that fact is stated.
    let pool = match CommandPool::new(&device, selected.family) {
        Ok(pool) => pool,
        Err(error) => {
            // The table is already live, so it is released before the device rather
            // than by field order, which an early return does not reach.
            drop(table);
            // SAFETY: the table was the only child object and is gone, so the device
            // is destroyed once, here.
            unsafe { device.destroy_device(None) };
            return Err(DeviceError::CommandPool(error));
        }
    };

    Ok(VulkanDevice {
        table,
        pool,
        device: OwnedDevice(device),
        queue,
        selected,
        stores,
        ledger: ledger(selected, limits, stores, &formats),
        formats,
        stamp,
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

    /// The same, with the family's timestamp-valid-bit report stated.
    fn family_reporting(
        flags: vk::QueueFlags,
        timestamp_valid_bits: u32,
    ) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            timestamp_valid_bits,
            ..family(flags)
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
                supports_compute: true,
                timestamp_valid_bits: 0,
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
                supports_compute: false,
                timestamp_valid_bits: 0,
            })
        );
    }

    #[test]
    fn the_family_timestamp_report_is_carried_rather_than_consulted() {
        // The selection rule is "this family rasterizes"; the timestamp report is a
        // fact the ledger records, and a family with a zero report is still selected.
        let families = [family_reporting(vk::QueueFlags::GRAPHICS, 64)];
        let selected = select_queue_family(&families).expect("a graphics family exists");
        assert_eq!(selected.timestamp_valid_bits, 64);
        assert!(selected.supports_timestamps());

        let silent = [family_reporting(vk::QueueFlags::GRAPHICS, 0)];
        let selected = select_queue_family(&silent).expect("a graphics family exists");
        assert!(!selected.supports_timestamps());
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
        assert_eq!(
            MAINTENANCE1_C.to_str().expect("ASCII name"),
            MAINTENANCE1
        );
    }

    #[test]
    fn a_device_without_maintenance1_is_refused_by_name() {
        // The extension is what makes step 12's negative viewport height legal, so
        // an adapter that cannot report it cannot serve the draw path at all. A near
        // miss is a different extension, exactly as for the swapchain name.
        let without = vec![SWAPCHAIN.to_owned()];
        assert_eq!(
            verify_maintenance1(&without),
            Err(MissingDeviceExtension::Maintenance1)
        );
        let with = vec![MAINTENANCE1.to_owned()];
        assert_eq!(verify_maintenance1(&with), Ok(()));
        for near_miss in ["vk_khr_maintenance1", "VK_KHR_maintenance1_extra"] {
            assert_eq!(
                verify_maintenance1(&[near_miss.to_owned()]),
                Err(MissingDeviceExtension::Maintenance1),
                "{near_miss} is a different extension"
            );
        }
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

    /// The selected-queue value the ledger tests build from.
    fn selected(supports_compute: bool, timestamp_valid_bits: u32) -> SelectedQueue {
        SelectedQueue {
            family: 0,
            supports_compute,
            timestamp_valid_bits,
        }
    }

    /// Adapter limits whose compute and storage floors are satisfied.
    fn computing_limits() -> AdapterLimits {
        AdapterLimits {
            max_texture_dimension_2d: 8192,
            max_compute_workgroups_per_dimension: [65535, 65535, 65535],
            max_compute_workgroup_size: [1024, 1024, 64],
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            ..AdapterLimits::unavailable()
        }
    }

    /// The feature facts of an adapter that reported no shader stores.
    fn no_stores() -> StoreFeatures {
        StoreFeatures {
            fragment: false,
            vertex: false,
        }
    }

    /// The feature facts of a device created with both shader-store features.
    fn all_stores() -> StoreFeatures {
        StoreFeatures {
            fragment: true,
            vertex: true,
        }
    }

    /// A format table nothing was recorded in: no format proves storage.
    fn no_formats() -> FormatTable {
        FormatTable::default()
    }

    /// A format table holding one format with exactly the storage facts stated.
    ///
    /// Built through the real per-format lowering so the fixture is a driver answer
    /// this backend could actually receive rather than a hand-written row.
    fn formats_with(storage_read: bool, storage_write: bool) -> FormatTable {
        let mut facts = format_facts::capabilities(
            fluxel_rendergraph::TextureFormat::Rgba8Unorm,
            &vk::FormatProperties {
                optimal_tiling_features: vk::FormatFeatureFlags::STORAGE_IMAGE,
                ..Default::default()
            },
        );
        facts.storage_read = storage_read;
        facts.storage_write = storage_write;
        let mut table = FormatTable::default();
        table
            .record(facts)
            .expect("a probed storage fact is recordable");
        table
    }

    /// A format table holding one format a shader may both read and write.
    fn stored_formats() -> FormatTable {
        formats_with(true, true)
    }

    #[test]
    fn a_created_device_proves_copy_without_a_command() {
        // Copy is the API version's own guarantee on a graphics family, so the
        // proof is structural and no probe was owed.
        let ledger = ledger(selected(false, 0), &computing_limits(), no_stores(), &no_formats());
        assert!(ledger.supports(Capability::Copy));
        assert_eq!(
            ledger.fact(Capability::Copy).map(|fact| fact.operation_probe),
            Some(OperationProbe::NotRequired),
        );
        assert_eq!(
            ledger.fact(Capability::Copy).map(|fact| fact.limits_satisfied),
            Some(true),
            "copy has no numeric floor, so nothing can leave it unsatisfied"
        );
    }

    #[test]
    fn the_draw_parameter_rows_are_proved_by_the_core_draw_commands() {
        // Both are core `Vulkan` 1.0 draw parameters rather than features, so they
        // are proved in a ledger whose every reported number is absent -- the
        // discriminating shape for a floor that is trivially satisfied instead of
        // borrowed from a neighbouring limit.
        let bare = ledger(
            selected(false, 0),
            &AdapterLimits::unavailable(),
            no_stores(),
            &no_formats(),
        );
        for row in [Capability::BaseVertex, Capability::FirstInstance] {
            assert!(bare.supports(row), "{row:?} is a core draw parameter");
            assert_eq!(
                bare.fact(row).map(|fact| fact.operation_probe),
                Some(OperationProbe::NotRequired),
                "{row:?} needs no probe: the parameter is part of the draw command"
            );
            assert_eq!(
                bare.fact(row).map(|fact| fact.limits_satisfied),
                Some(true),
                "{row:?} has no numeric floor, so nothing can leave it unsatisfied"
            );
        }
    }

    #[test]
    fn indirect_dispatch_arrives_only_beside_a_proved_compute_row() {
        // Without compute the row is not merely disabled, it was never examined --
        // which is the difference the ledger keeps for diagnostics.
        let without = ledger(selected(false, 0), &computing_limits(), no_stores(), &no_formats());
        assert!(!without.supports(Capability::IndirectDispatch));
        assert_eq!(without.fact(Capability::IndirectDispatch), None);

        let with = ledger(selected(true, 0), &computing_limits(), no_stores(), &no_formats());
        assert!(with.supports(Capability::IndirectDispatch));
    }

    #[test]
    fn an_unsatisfied_compute_floor_leaves_indirect_dispatch_disabled_too() {
        // An indirect dispatch is a dispatch: it keeps the compute row's floor
        // rather than a floor of its own, so the two cannot disagree.
        let ledger = ledger(selected(true, 0), &AdapterLimits::unavailable(), no_stores(), &no_formats());
        assert!(!ledger.supports(Capability::Compute));
        assert!(!ledger.supports(Capability::IndirectDispatch));
    }

    #[test]
    fn a_timestamp_row_arrives_only_from_a_non_zero_valid_bit_report() {
        let silent = ledger(selected(false, 0), &computing_limits(), no_stores(), &no_formats());
        assert!(!silent.supports(Capability::TimestampQuery));
        assert_eq!(
            silent.fact(Capability::TimestampQuery),
            None,
            "a family that reported nothing was never examined for timestamps"
        );

        let reporting = ledger(selected(false, 64), &computing_limits(), no_stores(), &no_formats());
        assert!(reporting.supports(Capability::TimestampQuery));
    }

    #[test]
    fn the_query_rows_arrive_from_the_graphics_and_timestamp_facts() {
        // Occlusion is core on a graphics family, and the row has no numeric floor
        // of its own -- so it is proved even in a ledger whose every reported number
        // is absent, which is the discriminating shape for a floor that is trivially
        // satisfied rather than borrowed from a neighbouring fact.
        let bare = ledger(
            selected(false, 0),
            &AdapterLimits::unavailable(),
            no_stores(),
            &no_formats(),
        );
        assert!(bare.supports(Capability::OcclusionQuery));
        assert_eq!(
            bare.fact(Capability::OcclusionQuery)
                .map(|fact| fact.limits_satisfied),
            Some(true),
            "occlusion has no numeric floor, so nothing can leave it unsatisfied"
        );

        // Elapsed is the timestamp facility used twice, so it follows that row
        // exactly: unexamined where the family reported no usable timestamps, and
        // proved where it did.
        assert!(!bare.supports(Capability::ElapsedQuery));
        assert_eq!(
            bare.fact(Capability::ElapsedQuery),
            None,
            "a family that reported no timestamps was never examined for elapsed time"
        );

        let reporting = ledger(selected(false, 64), &computing_limits(), no_stores(), &no_formats());
        assert!(reporting.supports(Capability::ElapsedQuery));
        assert_eq!(
            reporting
                .fact(Capability::ElapsedQuery)
                .map(|fact| fact.operation_probe),
            Some(OperationProbe::NotRequired),
            "the family's own valid-bit report is the proof, so no command was owed"
        );
    }

    #[test]
    fn the_storage_buffer_row_arrives_only_with_both_store_features() {
        // The row describes a buffer a shader may read and write, and `Vulkan` gates
        // the write half per stage, so each half alone is a partial proof that the
        // domain cannot report.
        let neither = ledger(selected(true, 0), &computing_limits(), no_stores(), &no_formats());
        assert!(!neither.supports(Capability::StorageBuffer));
        assert_eq!(
            neither.fact(Capability::StorageBuffer),
            None,
            "an adapter that reported no store feature was never examined for storage"
        );

        for partial in [
            StoreFeatures {
                fragment: true,
                vertex: false,
            },
            StoreFeatures {
                fragment: false,
                vertex: true,
            },
        ] {
            let ledger = ledger(selected(true, 0), &computing_limits(), partial, &no_formats());
            assert!(
                !ledger.supports(Capability::StorageBuffer),
                "{partial:?} serves one stage's write, not the domain"
            );
            assert_eq!(ledger.fact(Capability::StorageBuffer), None);
        }

        let both = ledger(selected(true, 0), &computing_limits(), all_stores(), &no_formats());
        assert!(both.supports(Capability::StorageBuffer));
        assert_eq!(
            both.fact(Capability::StorageBuffer)
                .map(|fact| fact.operation_probe),
            Some(OperationProbe::NotRequired),
            "the create-info's own feature set is the proof, so no command was owed"
        );
    }

    #[test]
    fn a_zero_storage_binding_size_leaves_the_row_disabled() {
        // The pair is enabled, so the row is examined; the driver's own binding-size
        // report is the floor, and a zero report is a device that can bind none.
        let limits = AdapterLimits {
            max_storage_buffer_binding_size: 0,
            ..computing_limits()
        };
        let ledger = ledger(selected(true, 0), &limits, all_stores(), &no_formats());
        assert!(!ledger.supports(Capability::StorageBuffer));
        assert_eq!(
            ledger.fact(Capability::StorageBuffer).map(|fact| fact.limits_satisfied),
            Some(false),
            "examined and refused is a different sentence from never asked"
        );
    }

    #[test]
    fn the_storage_image_row_needs_the_store_pair_and_a_storage_format() {
        // The row's two halves fail in different ledger fields, and the difference is
        // the sentence a caller gets: without the stage pair nothing examined the row,
        // while with it the row is examined and refused by its resource floor.
        let without_pair = ledger(
            selected(true, 0),
            &computing_limits(),
            no_stores(),
            &stored_formats(),
        );
        assert!(!without_pair.supports(Capability::StorageImage));
        assert_eq!(
            without_pair.fact(Capability::StorageImage),
            None,
            "the pair is the route, so without it nothing proved this backend has one"
        );

        // With the pair enabled, every shape that lacks the resource half is
        // examined and refused rather than left unexamined.
        for storage in [
            (false, false),
            // One direction is half the domain, and nothing may imply the other.
            (true, false),
            (false, true),
        ] {
            let ledger = ledger(
                selected(true, 0),
                &computing_limits(),
                all_stores(),
                &formats_with(storage.0, storage.1),
            );
            assert!(
                !ledger.supports(Capability::StorageImage),
                "read={} write={} is not the read-and-write domain",
                storage.0,
                storage.1
            );
            assert_eq!(
                ledger
                    .fact(Capability::StorageImage)
                    .map(|fact| fact.limits_satisfied),
                Some(false),
                "the pair exists, so the row was examined and its floor refused it"
            );
        }

        for partial in [
            StoreFeatures {
                fragment: true,
                vertex: false,
            },
            StoreFeatures {
                fragment: false,
                vertex: true,
            },
        ] {
            let ledger = ledger(
                selected(true, 0),
                &computing_limits(),
                partial,
                &stored_formats(),
            );
            assert!(
                !ledger.supports(Capability::StorageImage),
                "{partial:?} serves one stage's store, not the domain"
            );
            assert_eq!(
                ledger.fact(Capability::StorageImage),
                None,
                "and a partial pair is not a route, so the row stays unexamined"
            );
        }

        let both = ledger(
            selected(true, 0),
            &computing_limits(),
            all_stores(),
            &stored_formats(),
        );
        assert!(both.supports(Capability::StorageImage));
        assert_eq!(
            both.fact(Capability::StorageImage)
                .map(|fact| fact.operation_probe),
            Some(OperationProbe::NotRequired),
            "an enabled feature and a driver-reported format are the whole proof"
        );
    }

    #[test]
    fn the_feature_gated_and_preserved_rows_stay_unproved() {
        // These are absent for different reasons, and each is stated: the step that
        // can prove one narrows the family's parameter space or enables the feature.
        // The fixture records everything this device already reads -- the compute
        // family, a non-zero timestamp report and both store features -- so the rows
        // that *are* proved (the command rows, the query rows and the storage rows)
        // are absent from this list as a fact about the rows rather than about the
        // fixture.
        let ledger = ledger(selected(true, 64), &computing_limits(), all_stores(), &no_formats());
        for row in [
            // The `IndirectDrawApi` family takes a draw count, and a count above one
            // is the `MultiDrawIndirect` capability -- so claiming this row would
            // claim a capability the device did not enable a feature for.
            Capability::IndirectDraw,
            // Needs the `multiDrawIndirect` feature.
            Capability::MultiDrawIndirect,
            // Needs the `samplerAnisotropy` feature.
            Capability::AnisotropicFiltering,
            // Preserved closed by plan section 4 while its attach path is unproved.
            Capability::Multiview,
            // One queue, so neither a second compute queue nor a transfer queue was
            // ever created.
            Capability::AsyncCompute,
            Capability::TransferQueue,
        ] {
            assert!(!ledger.supports(row), "{row:?} must stay unproved");
            assert_eq!(ledger.fact(row), None, "{row:?} was never examined");
        }
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

        // `VK_KHR_maintenance1` is now part of what a usable device is: step 12's
        // viewport flip needs it on a 1.0 device. An adapter that cannot report it is
        // a fact about the machine rather than a failure of this module -- the frozen
        // oracle's borrowed path hides such an adapter too -- so the test returns
        // where the extension is absent rather than failing.
        let extensions = enumerate_device_extensions(instance.instance(), adapters[index])
            .expect("an adapter's extension inventory");
        if verify_maintenance1(&extensions).is_err() {
            return;
        }

        let device = open(&instance, adapters[index], &facts.limits).expect("a graphics family exists");
        let selected = device.selected_queue();
        assert!(
            selected.family < u32::MAX,
            "the selected family is a real index"
        );

        // The first real backend implementing the common layer's capability query,
        // and now the first to *negotiate* a family through it. `require` consults
        // this ledger before the backend is asked for a handle, so the negotiation
        // succeeding is the same fact the assertions below state about the value.
        use crate::common::api::family::Graphics;
        use crate::common::api::negotiate::{CapabilitySource, require};
        assert!(device.ledger().supports(Capability::Graphics));
        assert!(
            require::<_, Graphics>(&device).is_ok(),
            "the graphics row is proved, so a handle exists"
        );
        assert_eq!(
            device.ledger().supports(Capability::Compute),
            selected.supports_compute,
            "compute disagreement: queue={selected:?} limits={:?}",
            facts.limits
        );
        assert!(
            device.ledger().supports(Capability::Copy),
            "Vulkan 1.0 guarantees copies on the graphics family this device was created on"
        );
        assert_eq!(
            device.ledger().supports(Capability::IndirectDispatch),
            selected.supports_compute,
            "an indirect dispatch arrives with the compute row it is a form of"
        );
        assert_eq!(
            device.ledger().supports(Capability::TimestampQuery),
            selected.supports_timestamps(),
            "timestamp disagreement: queue={selected:?}"
        );
        assert!(
            device.ledger().supports(Capability::OcclusionQuery),
            "occlusion is a core query type on the graphics family this device was created on"
        );
        assert_eq!(
            device.ledger().supports(Capability::ElapsedQuery),
            selected.supports_timestamps(),
            "an elapsed interval is the timestamp facility used twice"
        );
        assert!(
            device.ledger().supports(Capability::BaseVertex),
            "vkCmdDrawIndexed's vertexOffset is core 1.0 on the graphics family"
        );
        assert!(
            device.ledger().supports(Capability::FirstInstance),
            "vkCmdDraw's firstInstance is core 1.0 on the graphics family"
        );
        assert!(
            !device.ledger().supports(Capability::IndirectDraw),
            "the family's count names MultiDrawIndirect, which this device enables no feature for"
        );

        // The storage-buffer row against the real driver, in both directions. The
        // adapter's report is read again and put through the same pure request the
        // creation path used, so this asserts that the device was created with
        // exactly what the rule decided -- not merely that the ledger agrees with
        // itself. A feature the adapter did not report stays disabled, and a device
        // created with both features proves the row.
        // SAFETY: the adapter belongs to the instance, which is still live, and the
        // call creates nothing and reports only the adapter's own facts.
        let reported = unsafe { instance.instance().get_physical_device_features(adapters[index]) };
        let expected = features::store(&features::request(&reported));
        assert_eq!(
            device.stores(),
            expected,
            "the device's enabled feature set is the pure rule applied to the adapter's report"
        );
        assert_eq!(
            device.ledger().supports(Capability::StorageBuffer),
            expected.proves_storage_buffers(),
            "the row is exactly the store pair this device was created with: {expected:?}"
        );

        // The storage-image row is the same pair plus a per-format fact, so the
        // assertion reads the device's *own* format table rather than querying again:
        // one discovery answer, and the row is exactly the two halves this device was
        // opened with.
        assert_eq!(
            device.formats().iter().count(),
            crate::native::vulkan::format::MAPPED.len(),
            "the device queried every format this backend maps"
        );
        assert_eq!(
            device.ledger().supports(Capability::StorageImage),
            expected.proves_storage_buffers() && device.formats().has_storage_read_write(),
            "the image row is the stage pair and the per-format resource fact together"
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
        if verify_maintenance1(&extensions).is_err() {
            // The same for a device that cannot serve the draw path's viewport.
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
