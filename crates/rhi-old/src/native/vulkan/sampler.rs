//! Step 4's sampler half: the portable sampler descriptor lowered onto `Vulkan`.
//!
//! # The absent comparison must reach the driver as an absent comparison
//!
//! [`SamplerDescriptor::compare`] is an `Option` for a reason that has already cost
//! this project a vendored dependency: the borrowed DX12 path lowers a *null*
//! comparison to `D3D12_COMPARISON_FUNC_ALWAYS` instead of `..._NONE`, which makes
//! an ordinary filtering sampler look like a shadow sampler and trips D3D12
//! validation error #1361. `Vulkan` spells the same distinction differently -- it
//! has no `COMPARE_OP_NONE`, and instead switches the comparison off with
//! `compare_enable` -- so the accident is available here in a different form:
//! enabling the comparison while filling `compare_op` from a default.
//!
//! The rule this module keeps is therefore stated as one direction only:
//! `compare_enable` is derived from `compare.is_some()` and from nothing else, and
//! `compare_op` is written only in the `Some` branch. A descriptor with no
//! comparison leaves `compare_op` at the `Vulkan` default, where the disabled flag
//! makes it an ignored value rather than a claim.
//!
//! # Nothing is invented for a field the descriptor does not state
//!
//! Every remaining field of `VkSamplerCreateInfo` is pinned to the value that
//! claims nothing:
//!
//! - `mip_lod_bias` is zero, because the portable descriptor carries no bias;
//! - anisotropy is disabled with `max_anisotropy` at 1.0. Anisotropic filtering is
//!   its own capability row and this device has not proved it, so enabling it here
//!   would be a capability claim rather than a lowering;
//! - `border_color` keeps the `Vulkan` default. `ClampToBorder` is not reachable
//!   from the portable address modes, so the field is never read;
//! - `unnormalized_coordinates` is false: the portable vocabulary has one
//!   coordinate convention and it is the normalized one.
//!
//! # One refusal
//!
//! `Vulkan` requires `minLod <= maxLod`. An inverted range is a caller's mistake,
//! and it is refused here rather than passed to the driver, for the same reason
//! [`super::buffer::create_info`] refuses a zero size: a validation error is a worse
//! answer than a reason the caller can act on.

use ash::vk;

use crate::common::sampler::{AddressMode, CompareFunction, FilterMode, SamplerDescriptor};

/// The `Vulkan` address mode for one portable mode.
///
/// Note the spelling: the portable `MirrorRepeat` is `Vulkan`'s
/// `MIRRORED_REPEAT`, not a second `MIRROR_REPEAT`. It is a separate constant and
/// the name differs, so a copy of the portable spelling would not compile rather
/// than silently selecting another mode.
pub(crate) const fn address_mode(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

/// The `Vulkan` texel filter for one portable filter mode.
pub(crate) const fn filter(mode: FilterMode) -> vk::Filter {
    match mode {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

/// The `Vulkan` mipmap filter for one portable filter mode.
///
/// A separate function from [`filter`] because `Vulkan` has a separate enum for it:
/// `VkSamplerMipmapMode` states how two *mip levels* are combined, which is not the
/// same question as how two texels within one level are.
pub(crate) const fn mipmap_mode(mode: FilterMode) -> vk::SamplerMipmapMode {
    match mode {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// The `Vulkan` comparison for one portable comparison function.
///
/// Every portable variant maps, and the mapping is a bijection: the two enums state
/// the same eight orderings. `Always` is a real comparison here, exactly as the
/// portable type documents, which is why it may not be reached from an *absent*
/// comparison (see the module docs).
pub(crate) const fn compare_function(function: CompareFunction) -> vk::CompareOp {
    match function {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

/// The create-info for a sampler described by `desc`.
///
/// Returns `None` for a description that is internally inconsistent -- today only an
/// inverted LOD range -- so the caller refuses it before the driver is reached. The
/// comparison is switched on exactly when the descriptor carries one; see the module
/// docs for why that is the one rule this file exists to keep.
pub(crate) fn create_info(desc: &SamplerDescriptor) -> Option<vk::SamplerCreateInfo<'static>> {
    if desc.lod_min_clamp > desc.lod_max_clamp {
        return None;
    }

    let mut info = vk::SamplerCreateInfo::default()
        .mag_filter(filter(desc.mag_filter))
        .min_filter(filter(desc.min_filter))
        .mipmap_mode(mipmap_mode(desc.mipmap_filter))
        .address_mode_u(address_mode(desc.address_u))
        .address_mode_v(address_mode(desc.address_v))
        .address_mode_w(address_mode(desc.address_w))
        // The portable descriptor states no bias, so the value that biases nothing.
        .mip_lod_bias(0.0)
        // Anisotropic filtering is a capability row this device has not proved.
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .min_lod(desc.lod_min_clamp)
        .max_lod(desc.lod_max_clamp)
        // The portable vocabulary has one coordinate convention: normalized.
        .unnormalized_coordinates(false);

    if let Some(compare) = desc.compare {
        // The only place `compare_enable` is written, so an absent comparison cannot
        // become an enabled one by a default or a later edit.
        info = info
            .compare_enable(true)
            .compare_op(compare_function(compare));
    }
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_comparison_reaches_the_driver_as_an_absent_one() {
        // The `Vulkan` form of the mistake the vendored DX12 patch fixes: a filtering
        // sampler must not arrive enabled as a comparison sampler.
        for desc in [
            SamplerDescriptor::nearest_clamp(),
            SamplerDescriptor::linear_clamp(),
        ] {
            let info = create_info(&desc).expect("a consistent description");
            assert_eq!(info.compare_enable, vk::FALSE);
            assert_ne!(info.compare_op, vk::CompareOp::ALWAYS);
            // The default is what an ignored field keeps, so the value is stated
            // rather than left to whatever ash's default happens to be.
            assert_eq!(info.compare_op, vk::CompareOp::NEVER);
        }
    }

    #[test]
    fn a_present_comparison_is_enabled_and_carries_its_ordering() {
        let desc = SamplerDescriptor {
            compare: Some(CompareFunction::LessEqual),
            ..SamplerDescriptor::nearest_clamp()
        };
        let info = create_info(&desc).expect("a consistent description");
        assert_eq!(info.compare_enable, vk::TRUE);
        assert_eq!(info.compare_op, vk::CompareOp::LESS_OR_EQUAL);
    }

    #[test]
    fn always_is_a_comparison_here_too() {
        // `Always` is reachable only through `Some`, and it is a real comparison.
        let desc = SamplerDescriptor {
            compare: Some(CompareFunction::Always),
            ..SamplerDescriptor::nearest_clamp()
        };
        let info = create_info(&desc).expect("a consistent description");
        assert_eq!(info.compare_enable, vk::TRUE);
        assert_eq!(info.compare_op, vk::CompareOp::ALWAYS);
    }

    /// Every portable comparison, listed once.
    const ALL_COMPARISONS: [CompareFunction; 8] = [
        CompareFunction::Never,
        CompareFunction::Less,
        CompareFunction::Equal,
        CompareFunction::LessEqual,
        CompareFunction::Greater,
        CompareFunction::NotEqual,
        CompareFunction::GreaterEqual,
        CompareFunction::Always,
    ];

    #[test]
    fn the_eight_comparisons_map_to_eight_distinct_vulkan_orders() {
        // A bijection, so a copy-paste in the match shows up as a shared value.
        for (index, function) in ALL_COMPARISONS.iter().enumerate() {
            for (other_index, other) in ALL_COMPARISONS.iter().enumerate() {
                if index != other_index {
                    assert_ne!(
                        compare_function(*function),
                        compare_function(*other),
                        "{function:?} and {other:?} share a Vulkan comparison"
                    );
                }
            }
        }
        assert_eq!(compare_function(CompareFunction::Never), vk::CompareOp::NEVER);
        assert_eq!(compare_function(CompareFunction::Always), vk::CompareOp::ALWAYS);
    }

    #[test]
    fn each_address_mode_maps_to_its_own_vulkan_mode() {
        assert_eq!(
            address_mode(AddressMode::ClampToEdge),
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        );
        assert_eq!(address_mode(AddressMode::Repeat), vk::SamplerAddressMode::REPEAT);
        assert_eq!(
            address_mode(AddressMode::MirrorRepeat),
            vk::SamplerAddressMode::MIRRORED_REPEAT
        );
        assert_ne!(
            address_mode(AddressMode::Repeat),
            address_mode(AddressMode::MirrorRepeat)
        );
    }

    #[test]
    fn each_filter_maps_to_its_own_vulkan_filter_in_both_places() {
        assert_eq!(filter(FilterMode::Nearest), vk::Filter::NEAREST);
        assert_eq!(filter(FilterMode::Linear), vk::Filter::LINEAR);
        assert_eq!(mipmap_mode(FilterMode::Nearest), vk::SamplerMipmapMode::NEAREST);
        assert_eq!(mipmap_mode(FilterMode::Linear), vk::SamplerMipmapMode::LINEAR);
        assert_ne!(filter(FilterMode::Nearest), filter(FilterMode::Linear));
        assert_ne!(
            mipmap_mode(FilterMode::Nearest),
            mipmap_mode(FilterMode::Linear)
        );
    }

    #[test]
    fn the_three_coordinates_carry_their_own_address_modes() {
        // The three may differ, so a lowering that wrote one mode into all three
        // would otherwise go unnoticed.
        let desc = SamplerDescriptor {
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::MirrorRepeat,
            ..SamplerDescriptor::nearest_clamp()
        };
        let info = create_info(&desc).expect("a consistent description");
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.address_mode_v, vk::SamplerAddressMode::REPEAT);
        assert_eq!(info.address_mode_w, vk::SamplerAddressMode::MIRRORED_REPEAT);
    }

    #[test]
    fn the_linear_recipe_carries_its_filters_its_clamps_and_claims_nothing_else() {
        let info =
            create_info(&SamplerDescriptor::linear_clamp()).expect("a consistent description");
        assert_eq!(info.mag_filter, vk::Filter::LINEAR);
        assert_eq!(info.min_filter, vk::Filter::LINEAR);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.min_lod, 0.0);
        assert_eq!(info.max_lod, 32.0);
        // The fields the portable descriptor does not state keep the value that
        // claims nothing, including the unproved anisotropic-filtering row.
        assert_eq!(info.mip_lod_bias, 0.0);
        assert_eq!(info.anisotropy_enable, vk::FALSE);
        assert_eq!(info.max_anisotropy, 1.0);
        assert_eq!(info.unnormalized_coordinates, vk::FALSE);
        assert_eq!(info.border_color, vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
    }

    #[test]
    fn the_lod_clamps_reach_the_create_info() {
        let desc = SamplerDescriptor {
            lod_min_clamp: 1.5,
            lod_max_clamp: 7.0,
            ..SamplerDescriptor::nearest_clamp()
        };
        let info = create_info(&desc).expect("a consistent description");
        assert_eq!(info.min_lod, 1.5);
        assert_eq!(info.max_lod, 7.0);
    }

    #[test]
    fn an_inverted_lod_range_is_refused_rather_than_sent_to_the_driver() {
        let desc = SamplerDescriptor {
            lod_min_clamp: 4.0,
            lod_max_clamp: 2.0,
            ..SamplerDescriptor::nearest_clamp()
        };
        assert!(create_info(&desc).is_none());
    }

    #[test]
    fn a_degenerate_lod_range_is_legal() {
        // A single mip level is a range, not an inversion: min == max is what a
        // sampler that may select exactly one level states.
        let desc = SamplerDescriptor {
            lod_min_clamp: 2.0,
            lod_max_clamp: 2.0,
            ..SamplerDescriptor::nearest_clamp()
        };
        assert!(create_info(&desc).is_some());
    }

    #[test]
    fn a_real_sampler_is_created_and_destroyed_on_this_machine() {
        // Step 4's sampler half against the real driver: a sampler needs no memory
        // and no view, so the handle is the whole resource. Skips where no adapter
        // exists.
        use crate::Validation;
        use crate::native::vulkan::open;

        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let info = create_info(&SamplerDescriptor::linear_clamp()).expect("a consistent description");
        // SAFETY: the device is live and owns the create/destroy entry points; the
        // handle is destroyed exactly once below and never stored.
        let handle = unsafe { opened.device.device().create_sampler(&info, None) }
            .expect("a valid sampler description");
        // SAFETY: the handle came from this device and is destroyed here once.
        unsafe { opened.device.device().destroy_sampler(handle, None) };
    }

    #[test]
    fn a_real_comparison_sampler_is_created_on_this_machine() {
        // The enabled-comparison branch reaches a real driver too, because a
        // driver that refuses it would be the one fact a pure test cannot see.
        use crate::Validation;
        use crate::native::vulkan::open;

        let Ok(opened) = open::open(Validation::Disabled, 0) else {
            return;
        };
        let desc = SamplerDescriptor {
            compare: Some(CompareFunction::GreaterEqual),
            ..SamplerDescriptor::nearest_clamp()
        };
        let info = create_info(&desc).expect("a consistent description");
        // SAFETY: live device, valid description, handle destroyed below.
        let handle = unsafe { opened.device.device().create_sampler(&info, None) }
            .expect("a valid comparison sampler description");
        // SAFETY: the handle came from this device and is destroyed here once.
        unsafe { opened.device.device().destroy_sampler(handle, None) };
    }
}
