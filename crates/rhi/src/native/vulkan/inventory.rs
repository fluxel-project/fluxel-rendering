//! Enumerating the instance-level names the validation probe reads.
//!
//! This is the FFI half of step 1. It creates nothing: both loader calls only
//! enumerate, so this can run before any instance exists, which is what lets
//! `Validation::Required` be refused before an instance is created rather than
//! after.
//!
//! The result is owned strings, and [`Enumeration::inventory`] lends them to
//! [`verify_required`] as the exact-name view that decision consumes. Keeping the
//! two apart is what makes the decision testable without a driver.
//!
//! # Why a malformed name is a failure and not a skip
//!
//! The loader reports names in fixed-width NUL-padded buffers. A name that is not
//! NUL-terminated or not UTF-8 is a fact this module cannot interpret, and
//! dropping it would be the permissive direction: the missing name might be
//! exactly the validation layer, and the probe would then report a clean
//! inventory. So an unreadable name fails the enumeration, and `Required`
//! therefore cannot be satisfied by an inventory that was not read whole.

use std::ffi::CStr;

use ash::vk;

use super::validation::InstanceInventory;

/// What one enumeration pass reported, as owned names.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Enumeration {
    /// Layer names the loader reported.
    pub layers: Vec<String>,
    /// Instance extension names the loader reported.
    pub instance_extensions: Vec<String>,
}

impl Enumeration {
    /// Lends the names to the validation decision.
    pub(crate) fn inventory(&self) -> InstanceInventory<'_> {
        InstanceInventory {
            layers: &self.layers,
            instance_extensions: &self.instance_extensions,
        }
    }
}

/// Why the instance-level names could not be read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EnumerationError {
    /// The loader failed to load, so nothing could be enumerated.
    Loader,
    /// The loader refused to enumerate layers.
    Layers(vk::Result),
    /// The loader refused to enumerate instance extensions.
    Extensions(vk::Result),
    /// A reported name was not NUL-terminated or not UTF-8.
    MalformedName,
}

/// Loads the Vulkan loader through `ash`'s dynamic entry point.
///
/// # Safety
///
/// The caller receives an owned entry point. `ash`'s dynamic loading is unsafe
/// because the loaded library must be a Vulkan loader; no further precondition is
/// placed on the caller here, and the returned value is bound to this module so no
/// loader function pointer escapes `native::vulkan`.
pub(crate) unsafe fn load_entry() -> Result<ash::Entry, EnumerationError> {
    // SAFETY: the caller's contract is that a Vulkan loader may be loaded; failure
    // is reported rather than assumed.
    unsafe { ash::Entry::load() }.map_err(|_| EnumerationError::Loader)
}

/// Reads the instance-level layer and extension names.
pub(crate) fn enumerate(entry: &ash::Entry) -> Result<Enumeration, EnumerationError> {
    // SAFETY: both calls only enumerate instance-level names; neither creates an
    // object. `entry` outlives both calls, and the returned vectors own their
    // contents, so nothing borrows the loader afterwards.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .map_err(EnumerationError::Layers)?;
    // SAFETY: as above, with no layer name requested, which is the instance-level
    // extension list.
    let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(EnumerationError::Extensions)?;
    Ok(Enumeration {
        layers: collect_names(
            layers
                .iter()
                .map(vk::LayerProperties::layer_name_as_c_str),
        )?,
        instance_extensions: collect_names(
            extensions
                .iter()
                .map(vk::ExtensionProperties::extension_name_as_c_str),
        )?,
    })
}

/// Converts the loader's NUL-padded name buffers into owned strings.
fn collect_names<'a>(
    values: impl Iterator<Item = Result<&'a CStr, impl core::fmt::Debug>>,
) -> Result<Vec<String>, EnumerationError> {
    values
        .map(|value| {
            value
                .map_err(|_| EnumerationError::MalformedName)
                .and_then(|name| {
                    name.to_str()
                        .map(str::to_owned)
                        .map_err(|_| EnumerationError::MalformedName)
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::vulkan::validation::{MissingValidation, verify_required};

    fn enumeration(layers: &[&str], extensions: &[&str]) -> Enumeration {
        Enumeration {
            layers: layers.iter().map(|name| (*name).to_owned()).collect(),
            instance_extensions: extensions.iter().map(|name| (*name).to_owned()).collect(),
        }
    }

    #[test]
    fn an_empty_enumeration_is_an_empty_inventory() {
        let enumeration = Enumeration::default();
        let inventory = enumeration.inventory();
        assert!(inventory.layers.is_empty());
        assert!(inventory.instance_extensions.is_empty());
        assert_eq!(
            verify_required(&inventory),
            Err(MissingValidation::Layer)
        );
    }

    #[test]
    fn the_borrowed_view_reads_the_names_the_enumeration_owns() {
        let enumeration = enumeration(&["VK_LAYER_KHRONOS_validation"], &[]);
        assert_eq!(
            verify_required(&enumeration.inventory()),
            Err(MissingValidation::FeatureExtension)
        );
    }

    #[test]
    fn a_full_enumeration_satisfies_the_requirement() {
        let enumeration = enumeration(
            &["VK_LAYER_KHRONOS_validation"],
            &["VK_EXT_validation_features"],
        );
        assert_eq!(verify_required(&enumeration.inventory()), Ok(()));
    }

    #[test]
    fn a_name_that_cannot_be_read_fails_the_whole_enumeration() {
        // Standing in for a name buffer that is not NUL-terminated or not UTF-8:
        // the failure has to propagate, because a skipped name could have been the
        // validation layer and the probe would then look clean.
        let broken: Vec<Result<&CStr, core::fmt::Error>> = vec![Err(core::fmt::Error)];
        assert_eq!(collect_names(broken.into_iter()), Err(EnumerationError::MalformedName));
    }

    #[test]
    fn readable_names_are_collected_in_order() {
        let names = [c"VK_KHR_surface", c"VK_EXT_validation_features"];
        let collected = collect_names(
            names
                .iter()
                .map(|name| Ok::<&CStr, core::fmt::Error>(*name)),
        )
        .expect("both names are readable");
        assert_eq!(
            collected,
            vec!["VK_KHR_surface".to_owned(), "VK_EXT_validation_features".to_owned()]
        );
    }
}
