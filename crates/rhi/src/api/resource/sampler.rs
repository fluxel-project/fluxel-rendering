//! Samplers (specification sections 16.1 and 16.2).
//!
//! A sampler is created once and may be bound many times. This module owns the
//! addressing modes, filters, comparison function, and descriptor that a caller
//! states, the opaque object that results, and the portable rules the descriptor
//! must satisfy.
//!
//! # What this module does not own
//!
//! - The *binding* contract a sampler takes part in — section 16.1's
//!   `SamplerKind` table, which distinguishes a comparison sampler from a
//!   non-filtering one from a filtering one — is the binding chapter's
//!   (section 20). `SamplerKind` describes what a shader interface requires of a
//!   sampler, which is a fact about the binding, not about the descriptor: the
//!   same `SamplerDescriptor` may be bound as a filtering sampler in one place
//!   and fail to satisfy a non-filtering interface in another, and that verdict
//!   belongs where the interface is known.
//! - Whether a texture may be sampled through this sampler is decided by the
//!   binding of the pair, not here.
//! - Backend-specific sampler objects and texture-state lowering are explicitly
//!   implementation details (section 16.1), so nothing in this module exposes
//!   one.
//!
//! # The one capability this descriptor can need
//!
//! Anisotropy is not a base capability: on WebGL2 it comes from an extension
//! (section 16.1), so a descriptor asking for `max_anisotropy > 1` is asking for
//! an *optional feature* to have been enabled, and for a value at or below the
//! device's maximum. Both halves are checked by
//! `validate_sampler_anisotropy`, which takes the two device facts as
//! parameters because the types that would carry them
//! (`OptionalFeature`, `DeviceLimits`) belong to module 01 and are not declared
//! yet.

use core::fmt;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};

/// How texture coordinates outside the `[0, 1]` range are resolved.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    /// Coordinates outside the range take the value of the nearest edge texel.
    ClampToEdge,
    /// The texture is tiled.
    Repeat,
    /// The texture is tiled, mirrored on every other tile.
    MirrorRepeat,
}

/// How texels are combined when more than one contributes to a sample.
///
/// This variant set is deliberately **not** `#[non_exhaustive]`, although every
/// other enum in this chapter is. The specification shows it without the
/// attribute (section 16.1), and it is transcribed as shown rather than
/// "corrected" — see the note in this crate's `0.16` series audit. A caller
/// matching it exhaustively therefore gets a compile error if a mode is added
/// later, which is the trade the specification made here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// The nearest texel wins.
    Nearest,
    /// A weighted combination of texels.
    Linear,
}

/// A comparison used when a sampler compares instead of filtering.
///
/// Needed for shadow-map style sampling, where the sampler returns the result of
/// a comparison rather than a filtered value. Section 16.1's `SamplerKind` table
/// calls a sampler with a comparison function a comparison sampler and requires
/// that such a sampler is not also expected to filter.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareFunction {
    /// The comparison never passes.
    Never,
    /// Passes when the sampled value is less than the reference.
    Less,
    /// Passes when the sampled value equals the reference.
    Equal,
    /// Passes when the sampled value is less than or equal to the reference.
    LessEqual,
    /// Passes when the sampled value is greater than the reference.
    Greater,
    /// Passes when the sampled value differs from the reference.
    NotEqual,
    /// Passes when the sampled value is greater than or equal to the reference.
    GreaterEqual,
    /// The comparison always passes.
    Always,
}

/// Everything a caller states about a sampler before it exists.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SamplerDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// Addressing mode for the U axis.
    pub address_u: AddressMode,
    /// Addressing mode for the V axis.
    pub address_v: AddressMode,
    /// Addressing mode for the W axis.
    pub address_w: AddressMode,

    /// Filter used when magnifying.
    pub mag_filter: FilterMode,
    /// Filter used when minifying.
    pub min_filter: FilterMode,
    /// Filter used between mip levels.
    pub mip_filter: FilterMode,

    /// Lower end of the level-of-detail clamp.
    pub lod_min: f32,
    /// Upper end of the level-of-detail clamp.
    pub lod_max: f32,

    /// Present when this sampler compares instead of filtering.
    pub compare: Option<CompareFunction>,

    /// 1 = anisotropy disabled.
    pub max_anisotropy: u16,
}

// Section 16.1 declares `new()` and no `Default` impl, so the clippy suggestion
// to add one is declined rather than satisfied: a `Default` impl is public API,
// and inventing one here would be designing an interface the specification did
// not freeze. The defaults below are the portable baseline the rest of the
// specification follows.
//
// `expect` rather than `allow`: both silence the lint, but `allow` keeps
// silencing it after the day the suggestion stops applying — at which point the
// suppression is dead weight nobody can see. `expect` fails loudly that day.
#[expect(
    clippy::new_without_default,
    reason = "section 16.1 declares new() and no Default, so adding one would be public API the specification did not freeze"
)]
impl SamplerDescriptor {
    /// A sampler with the portable defaults.
    ///
    /// Section 16.1's "Default:" block names only `max_anisotropy = 1`. The rest
    /// — clamp-to-edge on every axis, nearest for every filter, an LOD range of
    /// `0.0 ..= 32.0`, and no comparison — are the values that make "a descriptor
    /// built by a constructor" mean "a sampler that only does what the caller
    /// asked for": no filtering, no anisotropy, no comparison, and no LOD
    /// clamping beyond the standard full mip chain.
    pub fn new() -> Self {
        Self {
            label: Label::default(),
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod_min: 0.0,
            lod_max: 32.0,
            compare: None,
            max_anisotropy: 1,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Sets all three addressing modes at once.
    ///
    /// One builder for the three axes rather than three builders: the axes are
    /// almost always set together, and a per-axis setter would make the common
    /// case — all three the same — the longest one to write.
    pub fn with_address_modes(mut self, u: AddressMode, v: AddressMode, w: AddressMode) -> Self {
        self.address_u = u;
        self.address_v = v;
        self.address_w = w;
        self
    }

    /// Sets the magnification, minification, and mip filters at once.
    pub fn with_filters(mut self, mag: FilterMode, min: FilterMode, mip: FilterMode) -> Self {
        self.mag_filter = mag;
        self.min_filter = min;
        self.mip_filter = mip;
        self
    }

    /// Sets the level-of-detail clamp.
    ///
    /// Not validated here beyond what `validate_sampler_descriptor` checks:
    /// finiteness and ordering are decided there so that a builder chain cannot
    /// leave the descriptor in a state a caller believes is valid.
    pub fn with_lod_clamp(mut self, min: f32, max: f32) -> Self {
        self.lod_min = min;
        self.lod_max = max;
        self
    }

    /// Makes this a comparison sampler.
    pub fn with_compare(mut self, compare: CompareFunction) -> Self {
        self.compare = Some(compare);
        self
    }

    /// Sets the anisotropy level. 1 disables anisotropy.
    pub fn with_max_anisotropy(mut self, value: u16) -> Self {
        self.max_anisotropy = value;
        self
    }
}

/// A created sampler.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. Section 18.6 makes sampler backing part
/// of completion-safe retirement like every other resource.
#[derive(Clone)]
pub struct Sampler {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: SamplerDescriptor,
}

impl Sampler {
    /// Assembles a created sampler.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`crate::api::platform::Device::create_sampler`] may produce one. The
    /// verb itself waits on `api::platform`, which is not declared yet.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_sampler calls this once api::platform is declared"
        )
    )]
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, descriptor: SamplerDescriptor) -> Self {
        Self {
            id,
            device,
            descriptor,
        }
    }

    /// This sampler's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this sampler.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The descriptor this sampler was created from.
    pub fn descriptor(&self) -> &SamplerDescriptor {
        &self.descriptor
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (adjudication A16): a sampler is
/// `#[derive(Clone)]` and is named by bindings elsewhere, but section 7.1
/// describes an object by its identity rather than its contents. The backend
/// port will add a native field that has no reason to be `Debug`, and printing a
/// native handle into a log would leak it. `finish_non_exhaustive()` is what
/// makes it honest that the descriptor is not shown — a caller who needs it calls
/// [`Sampler::descriptor`].
impl fmt::Debug for Sampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sampler")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// Checks the parts of a sampler descriptor that no device can change.
///
/// Section 16.1's list:
///
/// ```text
/// lod_min / lod_max finite
/// lod_min <= lod_max
/// max_anisotropy >= 1
/// ```
///
/// "Finite" is not pedantry: `NaN` propagates through every comparison it
/// appears in, so a `NaN` bound would pass a naive `lod_min <= lod_max` check
/// (that comparison is false, but so is its negation) and reach a backend as an
/// undefined clamp. Infinity is refused for the same reason — a native LOD clamp
/// is a finite float on every backend this crate targets.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_sampler validates through this once api::platform is declared"
    )
)]
pub(crate) fn validate_sampler_descriptor(desc: &SamplerDescriptor) -> RhiResult<()> {
    if !desc.lod_min.is_finite() || !desc.lod_max.is_finite() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "sampler LOD clamp must be finite, not {} .. {}",
                desc.lod_min, desc.lod_max
            ),
        ));
    }
    if desc.lod_min > desc.lod_max {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "sampler LOD clamp {} .. {} is inverted",
                desc.lod_min, desc.lod_max
            ),
        ));
    }
    if desc.max_anisotropy < 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "sampler max_anisotropy must be at least 1; 1 disables anisotropy",
        ));
    }
    Ok(())
}

/// Checks an anisotropy request against the two device facts it depends on.
///
/// Section 16.1:
///
/// ```text
/// if max_anisotropy > 1:
///     OptionalFeature::SamplerAnisotropy must be enabled
///     value <= DeviceLimits::MaxSamplerAnisotropy
/// ```
///
/// The two device facts arrive as an `Option<u16>`: `None` means the optional
/// feature is not enabled, and `Some(limit)` means it is enabled with that
/// ceiling. They are parameters rather than reads because both are owned by the
/// capability module (`OptionalFeature`, `DeviceLimits`), which section 7 places
/// in module 01 — passing the two values keeps this rule portable and testable,
/// while a struct-literal of a type that does not exist yet could not compile.
///
/// A disabled feature is [`RhiErrorKind::Unsupported`], because the device
/// cannot express the request; a value over the ceiling is
/// [`RhiErrorKind::InvalidUsage`], because the device can honour anisotropy and
/// the number is out of range. The split matters to a caller deciding whether to
/// retry with a smaller value or to stop asking.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_sampler calls this once the capability module supplies the \
                  optional-feature and limit facts"
    )
)]
pub(crate) fn validate_sampler_anisotropy(
    desc: &SamplerDescriptor,
    anisotropy: Option<u16>,
) -> RhiResult<()> {
    if desc.max_anisotropy <= 1 {
        return Ok(());
    }
    let limit = anisotropy.ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "anisotropic filtering requires the SamplerAnisotropy optional feature, which \
             this device did not enable",
        )
    })?;
    if desc.max_anisotropy > limit {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "max_anisotropy {} exceeds the device maximum {limit}",
                desc.max_anisotropy
            ),
        ));
    }
    Ok(())
}
