//! Step 11's shader-store half: the device features the storage-buffer row needs,
//! read from the adapter and enabled only where the adapter reported them.
//!
//! # One row, two stage-gated facts
//!
//! [`Capability::StorageBuffer`] describes a buffer a shader may read **and**
//! write, and `Vulkan` 1.0 gates the write half by stage: reading a storage buffer
//! is core, while writing one from a fragment shader needs
//! `fragmentStoresAndAtomics` and writing one from a vertex shader needs
//! `vertexPipelineStoresAndAtomics`. Neither is enabled by default -- a device
//! enables exactly the features its create-info names -- so a backend that never
//! asked would record the row on a device where a fragment-shader store is an
//! invalid shader. That is the claim the borrowed path being replaced refuses to
//! make, and it is the one this module exists to keep honest.
//!
//! # What is requested is what was reported, and nothing else
//!
//! [`request`] answers from the adapter's own report and never from a preference: a
//! feature the adapter did not report stays disabled, because enabling one that
//! `VkPhysicalDeviceFeatures` does not contain makes `vkCreateDevice` fail outright
//! rather than leave the row unproved. Nothing else is requested either -- the
//! returned value is this backend's whole feature list, so a feature no step has
//! been taught cannot arrive by inheritance from a default.
//!
//! [`store`] then describes the set that was *requested*, so the ledger reads the
//! features the device was actually created with rather than a second reading of
//! the adapter's report. The two can only differ if a driver reported a feature and
//! then refused the device that enabled it, which is a creation failure and not a
//! silently unproved row.
//!
//! [`Capability::StorageBuffer`]: crate::common::caps::Capability::StorageBuffer

use ash::vk;

/// The shader-store features this backend asks for, and the row they prove.
///
/// A value rather than the raw `VkPhysicalDeviceFeatures` for the same reason
/// [`super::device::SelectedQueue`] is: the ledger reads facts about the device it
/// was created with, and the interesting rule here is stated over two booleans
/// rather than over an `ash` struct that carries no `PartialEq` to test with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StoreFeatures {
    /// `fragmentStoresAndAtomics`: a fragment shader may write a storage buffer.
    pub fragment: bool,
    /// `vertexPipelineStoresAndAtomics`: a vertex shader may write one.
    pub vertex: bool,
}

impl StoreFeatures {
    /// Whether the pair proves [`Capability::StorageBuffer`].
    ///
    /// Both halves, because the row is one claim about one domain and the write side
    /// is gated per stage: a device with only one of them can serve a fragment store
    /// but not a vertex one, so reporting the row from either half alone would
    /// authorize a shader the device refuses. The strict direction is also the
    /// fail-closed one -- a row left unproved refuses a graph, while a row proved
    /// too early records a pipeline the driver will reject.
    ///
    /// [`Capability::StorageBuffer`]: crate::common::caps::Capability::StorageBuffer
    pub(crate) const fn proves_storage_buffers(&self) -> bool {
        self.fragment && self.vertex
    }
}

/// The device create-info's feature set, from the adapter's own report.
///
/// Every other feature of `VkPhysicalDeviceFeatures` is left at the value that
/// enables nothing, which is what this backend requested before this module existed.
pub(crate) fn request(reported: &vk::PhysicalDeviceFeatures) -> vk::PhysicalDeviceFeatures {
    // `VkBool32` is a `u32`, and a driver answers `VK_TRUE` or `VK_FALSE`; any
    // non-zero value is a report of support rather than a value this layer defines.
    vk::PhysicalDeviceFeatures::default()
        .fragment_stores_and_atomics(reported.fragment_stores_and_atomics != 0)
        .vertex_pipeline_stores_and_atomics(reported.vertex_pipeline_stores_and_atomics != 0)
}

/// Which store features a requested set establishes.
pub(crate) const fn store(requested: &vk::PhysicalDeviceFeatures) -> StoreFeatures {
    StoreFeatures {
        fragment: requested.fragment_stores_and_atomics != 0,
        vertex: requested.vertex_pipeline_stores_and_atomics != 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An adapter reporting both store features and several this backend does not
    /// request.
    fn permissive() -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .sampler_anisotropy(true)
            .multi_draw_indirect(true)
            .shader_storage_image_write_without_format(true)
            .independent_blend(true)
    }

    #[test]
    fn a_reported_store_pair_is_requested_and_proves_the_row() {
        let requested = request(&permissive());
        assert_ne!(requested.fragment_stores_and_atomics, 0);
        assert_ne!(requested.vertex_pipeline_stores_and_atomics, 0);
        assert!(store(&requested).proves_storage_buffers());
    }

    #[test]
    fn a_store_feature_the_adapter_did_not_report_is_not_requested() {
        // The fail-closed direction: enabling an unreported feature does not leave a
        // row unproved, it makes the device creation fail. So the request has to be
        // narrower than "what would be nice to have".
        let reported = vk::PhysicalDeviceFeatures::default()
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(false);
        let requested = request(&reported);
        assert_ne!(requested.fragment_stores_and_atomics, 0);
        assert_eq!(
            requested.vertex_pipeline_stores_and_atomics, 0,
            "the adapter did not report the vertex half"
        );
        assert!(
            !store(&requested).proves_storage_buffers(),
            "one half does not serve a row that describes both"
        );
    }

    #[test]
    fn no_other_feature_arrives_by_inheritance() {
        // `request` returns the whole feature list the create-info is handed, so a
        // feature that was reported and that no step establishes a row with must
        // still come out disabled -- otherwise the row's evidence would be a device
        // capability nothing recorded.
        let requested = request(&permissive());
        assert_eq!(requested.sampler_anisotropy, 0);
        assert_eq!(requested.multi_draw_indirect, 0);
        assert_eq!(requested.shader_storage_image_write_without_format, 0);
        assert_eq!(requested.independent_blend, 0);
    }

    #[test]
    fn an_adapter_that_reported_nothing_requests_nothing() {
        let requested = request(&vk::PhysicalDeviceFeatures::default());
        assert_eq!(requested.fragment_stores_and_atomics, 0);
        assert_eq!(requested.vertex_pipeline_stores_and_atomics, 0);
        assert_eq!(
            store(&requested),
            StoreFeatures {
                fragment: false,
                vertex: false,
            }
        );
    }

    #[test]
    fn the_row_needs_both_halves() {
        // The discriminating pair for the one rule this type states: each half alone
        // is a different stage's write, and the row is the domain.
        assert!(
            !StoreFeatures {
                fragment: true,
                vertex: false,
            }
            .proves_storage_buffers()
        );
        assert!(
            !StoreFeatures {
                fragment: false,
                vertex: true,
            }
            .proves_storage_buffers()
        );
        assert!(
            StoreFeatures {
                fragment: true,
                vertex: true,
            }
            .proves_storage_buffers()
        );
    }
}
