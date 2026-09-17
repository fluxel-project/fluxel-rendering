//! The sampler vocabulary a backend creates samplers from.
//!
//! # Why `compare` is an `Option` and not a value with a sensible default
//!
//! A sampler either performs a comparison per sample (a shadow sampler) or does
//! not. Those are different samplers, and the difference has already cost this
//! project a vendored dependency: `wgpu-hal` 30 lowers a *null* comparison to
//! `D3D12_COMPARISON_FUNC_ALWAYS` instead of `D3D12_COMPARISON_FUNC_NONE`, which
//! triggers D3D12 validation error #1361 for an ordinary filtering sampler. The
//! fix is carried as a patch in `crates/wgpu-hal` today, and lead 3F owns it
//! explicitly instead of inheriting it (plan section 4, and W3).
//!
//! So this type states the absence as an absence. There is no `Always` default,
//! no `compare_or_default`, and no way to read a comparison function out of a
//! sampler that has none: [`CompareFunction::Always`] is a *real* comparison that
//! always passes, and it is not the same value as `None`.
//!
//! # The closed set
//!
//! Every field is needed by a retained artifact or by the backend that must
//! distinguish "no comparison" from a comparison. Nothing here exists for a
//! hypothetical recipe.

/// How out-of-range texture coordinates are resolved.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AddressMode {
    /// Coordinates outside the texture clamp to the edge texel.
    ClampToEdge,
    /// Coordinates outside the texture wrap around.
    Repeat,
    /// Coordinates outside the texture mirror and wrap.
    MirrorRepeat,
}

/// How a texel is chosen when a sample falls between texels or mip levels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum FilterMode {
    /// Selects the nearest texel or mip level.
    Nearest,
    /// Interpolates between neighbouring texels or mip levels.
    Linear,
}

/// A comparison a shadow sampler performs per sample.
///
/// The full ordering is stated because a comparison sampler is the one case where
/// every variant is meaningful; only the *presence* of the comparison is optional,
/// and that optionality lives on [`SamplerDescriptor::compare`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompareFunction {
    /// The comparison never passes.
    Never,
    /// Passes when the reference is less than the sampled value.
    Less,
    /// Passes when they are equal.
    Equal,
    /// Passes when the reference is less than or equal.
    LessEqual,
    /// Passes when the reference is greater.
    Greater,
    /// Passes when they differ.
    NotEqual,
    /// Passes when the reference is greater than or equal.
    GreaterEqual,
    /// Passes always -- a real comparison, and deliberately not the absent one.
    Always,
}

/// A complete sampler description.
///
/// The LOD clamps are plain `f32` and this type therefore implements `PartialEq`
/// rather than `Eq`. A backend that keys a cache on a descriptor must wrap or
/// canonicalize them first, the way the GL family already does with its finite
/// float wrapper; hashing the raw bits of a `NaN` would produce a key that never
/// compares equal to itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SamplerDescriptor {
    /// Address mode for the first coordinate.
    pub address_u: AddressMode,
    /// Address mode for the second coordinate.
    pub address_v: AddressMode,
    /// Address mode for the third coordinate.
    pub address_w: AddressMode,
    /// Filter used when magnifying.
    pub mag_filter: FilterMode,
    /// Filter used when minifying between texels.
    pub min_filter: FilterMode,
    /// Filter used between mip levels.
    pub mipmap_filter: FilterMode,
    /// Lowest mip level a sample may select.
    pub lod_min_clamp: f32,
    /// Highest mip level a sample may select.
    pub lod_max_clamp: f32,
    /// The per-sample comparison, or `None` for an ordinary filtering sampler.
    pub compare: Option<CompareFunction>,
}

impl SamplerDescriptor {
    /// A descriptor with no comparison and no filtering.
    ///
    /// The default is stated rather than derived, because the one field whose
    /// default would be dangerous is `compare`: `#[derive(Default)]` would give
    /// `None` for an `Option` correctly, but stating it here is what makes the
    /// absence visible at the definition site instead of inferred from a derive.
    pub(crate) const fn nearest_clamp() -> Self {
        Self {
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
        }
    }

    /// The descriptor the retained linear-clamp recipes create.
    pub(crate) const fn linear_clamp() -> Self {
        Self {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Self::nearest_clamp()
        }
    }

    /// Whether this sampler performs a per-sample comparison.
    pub(crate) const fn is_comparison(&self) -> bool {
        self.compare.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_sampler_has_no_comparison() {
        // The behavior the vendored DX12 patch exists to obtain: absence must
        // reach the backend as absence, not as `ALWAYS`.
        let sampler = SamplerDescriptor::nearest_clamp();
        assert_eq!(sampler.compare, None);
        assert!(!sampler.is_comparison());
    }

    #[test]
    fn always_is_a_comparison_and_not_the_absence_of_one() {
        let comparing = SamplerDescriptor {
            compare: Some(CompareFunction::Always),
            ..SamplerDescriptor::nearest_clamp()
        };
        assert!(comparing.is_comparison());
        assert_ne!(comparing.compare, None);
        assert_ne!(comparing, SamplerDescriptor::nearest_clamp());
    }

    #[test]
    fn the_linear_clamp_recipes_differ_from_nearest_only_in_filters() {
        let linear = SamplerDescriptor::linear_clamp();
        let nearest = SamplerDescriptor::nearest_clamp();
        assert_eq!(linear.address_u, AddressMode::ClampToEdge);
        assert_eq!(linear.address_v, AddressMode::ClampToEdge);
        assert_eq!(linear.address_w, AddressMode::ClampToEdge);
        assert_eq!(linear.mag_filter, FilterMode::Linear);
        assert_eq!(linear.min_filter, FilterMode::Linear);
        assert_eq!(linear.compare, None);
        // Wrapping and mip interpolation are what the recipes do not ask for.
        assert_eq!(linear.mipmap_filter, FilterMode::Nearest);
        assert_eq!(
            SamplerDescriptor {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                ..nearest
            },
            linear
        );
    }

    #[test]
    fn the_three_address_modes_are_distinct() {
        assert_ne!(AddressMode::ClampToEdge, AddressMode::Repeat);
        assert_ne!(AddressMode::Repeat, AddressMode::MirrorRepeat);
    }
}
