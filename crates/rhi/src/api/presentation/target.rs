//! Presentation surface facts (specification section 42).
//!
//! What a *target* reports about itself, and the vocabulary it reports in: the
//! present modes, the two-dimensional extent, who owns the drawable size, and the
//! snapshot a device answers with. It does not own a lease over a target
//! ([`crate::api::presentation::ConfiguredPresentation`] does), and it holds no
//! verb that reaches a platform surface — the one exception is
//! `Device::presentation_capabilities`, which belongs next to the snapshot it
//! returns and is written here for that reason.
//!
//! Invariant: the only source of surface facts is *one device plus one target*.
//! Section 42 requires that pairing because the same target can be preflighted by
//! several devices and get different answers; presentation is never a
//! device-global boolean.

use crate::api::error::{RhiError, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::platform::Device;

/// A present mode a caller may ask for.
///
/// Section 42.2 freezes four names and one rule: [`Self::Automatic`] is the only
/// mode every presentation backend must support, and it does **not** promise a
/// mapping onto any particular native mode. A browser backend may report only
/// that one, which is legal.
///
/// The other three are requests in the ordinary sense: a caller may ask, the
/// target may not offer it, and the configuration is refused rather than quietly
/// substituted. That last point is why there is no `display-policy`-shaped
/// variant: a host or compositor that presents on its own schedule must be
/// reported as `Automatic`, because pressing it into `Fifo` would claim a pacing
/// guarantee the RHI does not have.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMode {
    /// The backend's own default, and the only universally supported mode.
    Automatic,
    /// Queue for presentation; presentation is locked to the display's refresh.
    Fifo,
    /// Queue for presentation, but a newer image replaces an undisplayed one.
    Mailbox,
    /// Present as soon as the image is ready, without waiting for the display.
    Immediate,
}

/// A width and height in texels.
///
/// Public fields, for the same reason [`crate::api::resource::texture::Extent3d`]
/// has them: an extent is arithmetic rather than identity, and each concrete
/// constraint on a value belongs to the thing being sized, not to the number.
/// Which extents are *legal* is decided by [`PresentationExtentControl`] and by
/// the configuration that consumes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent2d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

/// Who decides the drawable extent of a target.
///
/// Section 42.3 splits the two cases that a single `resize()`-shaped API would
/// have blurred. The concrete mapping onto a backend is private; what matters to
/// a caller is whether an exact extent may be requested at all, which is what
/// [`crate::api::presentation::PresentationExtent::Exact`] consults.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationExtentControl {
    /// The host owns the drawable size: a browser canvas, an adopted GL context,
    /// and several surface/window paths.
    ///
    /// `current` is what the target reports right now, and `None` means the
    /// target could not say — a suspended, hidden, or zero-sized surface is the
    /// usual reason. Section 42.5 already makes the whole snapshot a
    /// query-time answer, so `None` is a fact about this query rather than a
    /// promise about later ones.
    HostManaged {
        /// The extent the target reports at query time, when it can report one.
        current: Option<Extent2d>,
    },

    /// The RHI may request an exact drawable extent, within the given bounds.
    Configurable {
        /// The smallest extent the target accepts.
        min: Extent2d,
        /// The largest extent the target accepts.
        max: Extent2d,
    },
}

/// What one device reports about one presentation target.
///
/// Section 42.4 freezes this snapshot and deliberately stops at these three
/// answers. Two things are *not* here, and their absence is the contract:
///
/// ```text
/// an ordinary drawable TextureView   not exposed, and not to be sneaked in
/// image count (min/max/desired)      not exposed, because no unified model exists
/// ```
///
/// A GL/WebGL2 default framebuffer is not a texture, so a "surface texture" would
/// pollute resource identity, lifetime, and inventory for the sake of a few
/// backends; section 42.4 requires a separate `PresentationTexture` extension if a
/// real consumer ever needs to sample or copy an acquired drawable. Likewise the
/// buffering counts are absent because WebGPU, GL, Metal, and native swapchains do
/// not share an ownership model for them — the portable rule that replaces them is
/// the single outstanding frame of section 43.4.
///
/// Section 42.5 makes every one of these answers a snapshot: resize, display move,
/// host context recreation, compositor change, and surface loss all make it stale,
/// so a successful query is not a permanent guarantee that `configure` succeeds.
/// `crate::api::presentation::validate_presentation_configuration` therefore
/// re-checks a configuration against the facts at configuration time.
#[derive(Clone, Debug)]
pub struct PresentationTargetCapabilities {
    formats: Vec<TextureFormat>,
    present_modes: Vec<PresentMode>,
    extent_control: PresentationExtentControl,
}

impl PresentationTargetCapabilities {
    /// Assembles one device's answer about one target.
    ///
    /// Crate-private: these are surface facts, so only the code that queried the
    /// surface may state them. A caller that could build this snapshot could claim
    /// a format the target cannot present, and the refusal would then land on the
    /// first frame instead of at configuration time.
    pub(crate) fn new(
        formats: Vec<TextureFormat>,
        present_modes: Vec<PresentMode>,
        extent_control: PresentationExtentControl,
    ) -> Self {
        Self {
            formats,
            present_modes,
            extent_control,
        }
    }

    /// The formats this target can be configured with.
    ///
    /// An empty slice is a real answer — this target offers nothing the RHI can
    /// present in — and not a placeholder for "unknown".
    pub fn formats(&self) -> &[TextureFormat] {
        &self.formats
    }

    /// The present modes this target offers.
    ///
    /// [`PresentMode::Automatic`] is always accepted by configuration whether or
    /// not it appears here, because section 42.2 makes it the one mode every
    /// backend must support.
    pub fn present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// Who owns this target's drawable extent, and within what bounds.
    pub fn extent_control(&self) -> PresentationExtentControl {
        self.extent_control
    }
}

impl Device {
    /// What this device reports about `target`.
    ///
    /// The only way to obtain a [`PresentationTargetCapabilities`], and the reason
    /// section 42 pairs a device with a target rather than asking either alone:
    /// the same surface can answer differently through different providers, so a
    /// snapshot taken from one device must not be used to configure another.
    ///
    /// Panics until the presentation backend exists. There is nothing to validate
    /// first: a target carries no device identity of its own by design (section
    /// 42.1 keeps it a host object outside the device execution domain), so this
    /// call cannot be a cross-device mistake.
    pub fn presentation_capabilities(
        &self,
        target: &crate::api::presentation::PresentationTarget,
    ) -> RhiResult<PresentationTargetCapabilities> {
        self.native()
            .presentation()
            .ok_or_else(|| {
                RhiError::new(
                    crate::api::error::RhiErrorKind::Unsupported,
                    "this backend does not implement presentation",
                )
                .at("Device::presentation_capabilities")
            })?
            .capabilities(target.id())
    }
}
