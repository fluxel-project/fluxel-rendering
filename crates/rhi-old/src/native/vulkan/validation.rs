//! The fail-closed probe `Validation::Required` depends on.
//!
//! `Validation::Required` is only meaningful if a native validation facility was
//! *positively observed*. `Validation::Disabled` permits ordinary probing but is
//! never evidence that validation was active, and a request for `Required` that
//! cannot be verified is refused before any device is returned.
//!
//! Vulkan needs two things, and the current implementation gates on both being
//! available before it creates an instance
//! (`imp/device_open.rs::open_vulkan`, `vulkan_validation_is_available`):
//!
//! - the `VK_LAYER_KHRONOS_validation` layer, which supplies the checks;
//! - the `VK_EXT_validation_features` instance extension, which the HAL uses for
//!   synchronization validation.
//!
//! # Why the decision is separated from the enumeration
//!
//! Enumerating layers and instance extensions is FFI: it returns OS strings and
//! needs a live entry point. Deciding whether what was enumerated satisfies the
//! requirement is not, and separating them means the rule can be tested against
//! exact inventories instead of against whatever this machine happens to have
//! installed. That matters because the interesting cases are the near misses --
//! a similarly spelled layer, a missing extension -- and those are not
//! reproducible on a developer's machine.
//!
//! # Why the reason is internal
//!
//! This returns *which* piece is missing, because a probe that only says "not
//! available" cannot be diagnosed. `OpenError::ValidationUnavailable` is a public
//! type that carries only the backend, and widening it is a separate decision
//! about the public surface; the caller therefore maps this reason onto the
//! existing error and keeps the observable behavior unchanged.

/// The layer that supplies Vulkan validation.
pub(crate) const REQUIRED_LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// The instance extension used for synchronization validation.
pub(crate) const REQUIRED_FEATURE_EXTENSION: &str = "VK_EXT_validation_features";

/// Exact names one instance reported, as owned strings.
///
/// The FFI half converts the driver's fixed-width name buffers into these; nothing
/// else about the probe needs a live instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InstanceInventory<'a> {
    /// Layer names, in the order the loader reported them.
    pub layers: &'a [String],
    /// Instance extension names, in the order the loader reported them.
    pub instance_extensions: &'a [String],
}

/// Which part of the validation facility was not observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingValidation {
    /// The validation layer was not reported.
    Layer,
    /// The layer is present and the features extension is not.
    FeatureExtension,
}

/// Verifies that an instance's inventory can satisfy `Validation::Required`.
///
/// The comparison is exact. Vulkan names are case-sensitive, so a layer whose name
/// differs in any character is a different layer and does not satisfy the
/// requirement; accepting a case-insensitive match would be exactly the
/// permissiveness that makes a "verified" validation flag untrustworthy.
///
/// The layer is checked first, because it is the facility itself: reporting a
/// missing extension when no layer was found would name the smaller of two
/// problems.
pub(crate) fn verify_required(
    inventory: &InstanceInventory<'_>,
) -> Result<(), MissingValidation> {
    if !inventory.layers.iter().any(|name| name == REQUIRED_LAYER) {
        return Err(MissingValidation::Layer);
    }
    if !inventory
        .instance_extensions
        .iter()
        .any(|name| name == REQUIRED_FEATURE_EXTENSION)
    {
        return Err(MissingValidation::FeatureExtension);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn inventory<'a>(layers: &'a [String], extensions: &'a [String]) -> InstanceInventory<'a> {
        InstanceInventory {
            layers,
            instance_extensions: extensions,
        }
    }

    #[test]
    fn both_pieces_present_satisfies_the_requirement() {
        let layers = names(&[REQUIRED_LAYER]);
        let extensions = names(&[REQUIRED_FEATURE_EXTENSION]);
        assert_eq!(verify_required(&inventory(&layers, &extensions)), Ok(()));
    }

    #[test]
    fn other_layers_and_extensions_alongside_them_are_fine() {
        let layers = names(&["VK_LAYER_KHRONOS_synchronization2", REQUIRED_LAYER]);
        let extensions = names(&["VK_KHR_surface", REQUIRED_FEATURE_EXTENSION]);
        assert_eq!(verify_required(&inventory(&layers, &extensions)), Ok(()));
    }

    #[test]
    fn an_empty_inventory_reports_the_layer_first() {
        let none: Vec<String> = Vec::new();
        assert_eq!(
            verify_required(&inventory(&none, &none)),
            Err(MissingValidation::Layer)
        );
    }

    #[test]
    fn a_missing_feature_extension_is_reported_once_the_layer_is_present() {
        let layers = names(&[REQUIRED_LAYER]);
        let extensions = names(&["VK_KHR_surface"]);
        assert_eq!(
            verify_required(&inventory(&layers, &extensions)),
            Err(MissingValidation::FeatureExtension)
        );
    }

    #[test]
    fn a_near_miss_on_case_does_not_satisfy_the_requirement() {
        // Vulkan names are case-sensitive. A probe that matched loosely would
        // report validation as verified when a different layer was loaded.
        let layers = names(&["vk_layer_khronos_validation"]);
        let extensions = names(&[REQUIRED_FEATURE_EXTENSION]);
        assert_eq!(
            verify_required(&inventory(&layers, &extensions)),
            Err(MissingValidation::Layer)
        );
    }

    #[test]
    fn a_near_miss_on_the_extension_is_refused_too() {
        let layers = names(&[REQUIRED_LAYER]);
        let extensions = names(&["VK_EXT_validation_feature"]);
        assert_eq!(
            verify_required(&inventory(&layers, &extensions)),
            Err(MissingValidation::FeatureExtension)
        );
    }

    #[test]
    fn a_prefix_of_the_layer_name_is_not_the_layer() {
        let layers = names(&["VK_LAYER_KHRONOS_validation_core"]);
        let extensions = names(&[REQUIRED_FEATURE_EXTENSION]);
        assert_eq!(
            verify_required(&inventory(&layers, &extensions)),
            Err(MissingValidation::Layer)
        );
    }
}
