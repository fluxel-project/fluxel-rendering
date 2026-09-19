//! Step 1's instance creation, and the fail-closed order it preserves.
//!
//! The order is the behavior: when `Validation::Required` is asked for, the
//! inventory is verified **before** an instance exists, so a machine without a
//! usable validation facility is refused without creating anything. A refusal that
//! had to create an instance first would be a refusal that already had a side
//! effect, which is the property the rest of this crate refuses to give up.
//!
//! ```text
//! Required -> load entry -> enumerate -> verify -> create with layer + extension
//!                             |
//!                             `- missing -> refuse, nothing created
//! Disabled -> load entry -> create with neither enabled, and verify nothing
//! ```
//!
//! `Validation::Disabled` is never evidence that validation is active: it enables
//! nothing, checks nothing, and it is not an error for such an instance to have no
//! validation at all.
//!
//! # Why the enabled names are `CStr` and the compared names are `&str`
//!
//! `enabled_layer_names` takes an array of `*const c_char`, and
//! `str::as_ptr` does **not** give a NUL-terminated pointer -- building the array
//! from `REQUIRED_LAYER.as_ptr()` would hand the driver a name that runs past its
//! end. The enabled names are therefore `c"..."` literals, which are
//! NUL-terminated, and a test asserts that each one still spells the name the
//! validation probe compares against, so the two cannot drift silently.
//!
//! # What this deliberately does not do yet
//!
//! It requests API version 1.0 and enables no extension beyond the ones the caller
//! asked for: the validation extension on the `Required` path, and the surface
//! extensions on the [`open_with_surface`] path. Raising the device/instance feature
//! set belongs to the steps that need them, which is also when the facts they enable
//! must be recorded in the capability ledger rather than assumed.

use std::ffi::{CStr, c_char};

use ash::vk;

use crate::Validation;

use super::inventory::{Enumeration, EnumerationError, enumerate, load_entry};
use super::surface::{
    MissingSurfaceExtension, SURFACE_EXTENSIONS, verify_surface_extensions,
};
use super::validation::{MissingValidation, verify_required};

/// The validation layer name, NUL-terminated for the loader.
const REQUIRED_LAYER_C: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The validation-features extension name, NUL-terminated for the loader.
const REQUIRED_FEATURE_EXTENSION_C: &CStr = c"VK_EXT_validation_features";

/// An instance usable for headless execution, with the loader that created it.
///
/// Field order is teardown order: the instance is destroyed before the loader that
/// created it, and `Drop` is what makes that happen rather than a convention.
pub(crate) struct ValidationInstance {
    instance: ash::Instance,
    entry: ash::Entry,
    validation_enabled: bool,
}

impl ValidationInstance {
    /// Returns the loader this instance came from.
    pub(crate) fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// Returns the instance handle.
    pub(crate) fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Whether validation was positively verified and enabled here.
    ///
    /// This is recorded rather than inferred from the request, because a caller
    /// asking whether validation is active must be able to tell "not requested"
    /// from "requested and verified".
    pub(crate) const fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }
}

impl Drop for ValidationInstance {
    fn drop(&mut self) {
        // SAFETY: this is the only owner, so the instance is destroyed exactly
        // once; no child object outlives the call because nothing else holds one,
        // and the loader is still alive as the next field.
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl core::fmt::Debug for ValidationInstance {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ValidationInstance")
            .field("validation_enabled", &self.validation_enabled)
            .finish_non_exhaustive()
    }
}

/// An instance created with the surface instance extensions verified and enabled.
///
/// A distinct type rather than a flag on [`ValidationInstance`], because the
/// surface entry points (`vkCreateWin32SurfaceKHR` among them) exist only when the
/// extension was enabled: `ash` substitutes a panicking stub for a function the
/// loader did not resolve, so "create a surface on an instance that never enabled
/// `VK_KHR_win32_surface`" must not be expressible. Only [`open_with_surface`]
/// produces this value.
pub(crate) struct SurfaceInstance(ValidationInstance);

impl SurfaceInstance {
    /// Returns the underlying instance.
    pub(crate) fn instance(&self) -> &ValidationInstance {
        &self.0
    }

    /// Returns whether validation was positively verified for this open.
    ///
    /// Delegated so a caller holding the surface witness does not have to reach
    /// through it to ask the same question.
    pub(crate) const fn validation_enabled(&self) -> bool {
        self.0.validation_enabled()
    }
}

impl core::fmt::Debug for SurfaceInstance {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SurfaceInstance")
            .field("validation_enabled", &self.validation_enabled())
            .finish_non_exhaustive()
    }
}

/// Why an instance could not be opened.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InstanceError {
    /// The loader or its instance-level names could not be read.
    Enumeration(EnumerationError),
    /// The surface path was asked for and the loader offers no surface facility.
    SurfaceExtension(MissingSurfaceExtension),
    /// `Validation::Required` was asked for and could not be verified.
    ValidationMissing(MissingValidation),
    /// The loader refused to create the instance.
    Creation(vk::Result),
}

/// Loads a loader, reads its inventory, verifies `validation`, and creates the
/// instance.
pub(crate) fn open(validation: Validation) -> Result<ValidationInstance, InstanceError> {
    open_with(validation, false)
}

/// Loads a loader, reads its inventory, verifies the surface extensions, and
/// creates an instance a `VkSurfaceKHR` can be created from.
///
/// This is the instance half of the surface path: the extensions the surface needs
/// are verified against the inventory **before** creation, exactly as
/// `Validation::Required` is, so a loader that cannot present is refused without
/// having created anything.
pub(crate) fn open_with_surface(validation: Validation) -> Result<SurfaceInstance, InstanceError> {
    open_with(validation, true).map(SurfaceInstance)
}

/// The shared order: enumerate, verify what was asked for, create.
fn open_with(validation: Validation, surface: bool) -> Result<ValidationInstance, InstanceError> {
    // SAFETY: loading a Vulkan loader is the platform contract of this module;
    // failure is reported as a value rather than assumed away.
    let entry = unsafe { load_entry() }.map_err(InstanceError::Enumeration)?;
    let inventory = enumerate(&entry).map_err(InstanceError::Enumeration)?;
    create(entry, validation, &inventory, surface)
}

/// Creates the instance for one already-read inventory.
///
/// Split from [`open`] so the ordering rule is visible in one function: every
/// verification happens before `create_instance` is reached, and the `?` is what
/// guarantees no instance exists on a refusal path. The surface extensions are the
/// caller's explicit request for this path, so they are checked first.
fn create(
    entry: ash::Entry,
    validation: Validation,
    inventory: &Enumeration,
    surface: bool,
) -> Result<ValidationInstance, InstanceError> {
    if surface {
        verify_surface_extensions(&inventory.inventory())
            .map_err(InstanceError::SurfaceExtension)?;
    }

    let verify = matches!(validation, Validation::Required);
    if verify {
        verify_required(&inventory.inventory()).map_err(InstanceError::ValidationMissing)?;
    }

    let application_name = c"fluxel-rhi";
    let application_info = vk::ApplicationInfo::default()
        .application_name(application_name)
        .api_version(vk::API_VERSION_1_0);

    // Both arrays and the pNext chain must outlive `create_instance`, so they are
    // locals the builder borrows for the duration of the call.
    let layer_names = [REQUIRED_LAYER_C.as_ptr()];
    let mut extension_names: Vec<*const c_char> = Vec::new();
    if surface {
        extension_names.extend(SURFACE_EXTENSIONS.iter().map(|name| name.as_ptr()));
    }
    if verify {
        extension_names.push(REQUIRED_FEATURE_EXTENSION_C.as_ptr());
    }
    let mut validation_features = vk::ValidationFeaturesEXT::default()
        .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION]);

    let mut create_info = vk::InstanceCreateInfo::default().application_info(&application_info);
    if !extension_names.is_empty() {
        create_info = create_info.enabled_extension_names(&extension_names);
    }
    if verify {
        create_info = create_info
            .enabled_layer_names(&layer_names)
            .push_next(&mut validation_features);
    }

    // SAFETY: `create_info` and everything it points at -- the application info,
    // both name arrays and the validation-features chain -- are locals that outlive
    // this call. No allocation callbacks are supplied, which asks the driver for
    // its default allocator.
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.map_err(InstanceError::Creation)?;

    Ok(ValidationInstance {
        instance,
        entry,
        validation_enabled: verify,
    })
}

#[cfg(test)]
mod tests {
    use crate::native::vulkan::validation::{REQUIRED_FEATURE_EXTENSION, REQUIRED_LAYER};
    use super::*;

    #[test]
    fn the_enabled_names_still_spell_the_names_the_probe_compares() {
        // The guard for the one drift this module cannot express in the type
        // system: the loader is handed a NUL-terminated `CStr`, while the probe
        // compares owned `String`s.
        assert_eq!(
            REQUIRED_LAYER_C.to_str().expect("ASCII name"),
            REQUIRED_LAYER
        );
        assert_eq!(
            REQUIRED_FEATURE_EXTENSION_C.to_str().expect("ASCII name"),
            REQUIRED_FEATURE_EXTENSION
        );
    }

    #[test]
    fn a_required_open_refuses_before_creating_anything() {
        // An inventory that can never satisfy `Required` must be refused with the
        // validation reason, never with a creation failure -- which is what proves
        // nothing was created on that path.
        let empty = Enumeration::default();
        // SAFETY: a loader is needed to reach the code under test; when none is
        // installed, having no loader is the honest answer and the test returns.
        let Ok(entry) = (unsafe { load_entry() }) else {
            return;
        };
        let error = create(entry, Validation::Required, &empty, false)
            .expect_err("an empty inventory cannot satisfy Required");
        assert_eq!(
            error,
            InstanceError::ValidationMissing(MissingValidation::Layer)
        );
    }

    #[test]
    fn a_surface_open_refuses_before_validation_is_even_asked() {
        // The surface extensions are verified first because they are this path's
        // explicit request, and both verifications precede creation. An empty
        // inventory therefore names the surface facility rather than validation.
        let empty = Enumeration::default();
        // SAFETY: as above.
        let Ok(entry) = (unsafe { load_entry() }) else {
            return;
        };
        let error = create(entry, Validation::Required, &empty, true)
            .expect_err("an empty inventory offers no surface extension");
        assert_eq!(
            error,
            InstanceError::SurfaceExtension(MissingSurfaceExtension::Surface)
        );
    }

    #[test]
    fn a_disabled_open_never_fails_on_validation() {
        let empty = Enumeration::default();
        // SAFETY: as above.
        let Ok(entry) = (unsafe { load_entry() }) else {
            return;
        };
        match create(entry, Validation::Disabled, &empty, false) {
            Ok(instance) => assert!(!instance.validation_enabled()),
            // A driver without Vulkan, or an exhausted loader, is not this test's
            // subject; the subject is that validation is never the reason.
            Err(InstanceError::Creation(_)) => {}
            Err(other) => panic!("a disabled open must not fail on validation: {other:?}"),
        }
    }

    #[test]
    fn a_surface_capable_instance_opens_with_the_verified_extensions() {
        // A real loader and a real instance on this machine. A loader that cannot
        // present on this target is a fact about the machine rather than this
        // module, and its refusal names the extension; creation failing *after* the
        // extensions were verified is the one outcome that would mean this module
        // enabled something it did not check.
        match open_with_surface(Validation::Disabled) {
            Ok(instance) => assert!(!instance.validation_enabled()),
            Err(InstanceError::SurfaceExtension(_)) | Err(InstanceError::Enumeration(_)) => {}
            Err(InstanceError::Creation(result)) => {
                panic!("surface extensions were verified yet creation failed: {result:?}")
            }
            Err(other) => panic!("unexpected surface-open failure: {other:?}"),
        }
    }
}
