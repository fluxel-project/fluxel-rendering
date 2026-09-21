//! Platform-neutral GL-family profile facts.

/// A GL-family context profile normalized without retaining platform handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GlFamilyProfile {
    /// A desktop OpenGL context with its reported core version.
    Desktop { major: u8, minor: u8 },
    /// An OpenGL ES context with its reported core version.
    Embedded { major: u8, minor: u8 },
    /// The WebGL 2 core profile.
    WebGl2,
}

/// A core-version requirement for one GL family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GlVersion {
    /// Major version component.
    pub major: u8,
    /// Minor version component.
    pub minor: u8,
}

impl GlVersion {
    /// Creates a core-version requirement.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Returns whether `actual` meets this requirement.
    pub const fn is_met_by(self, actual: Self) -> bool {
        actual.major > self.major || (actual.major == self.major && actual.minor >= self.minor)
    }
}

impl GlFamilyProfile {
    pub(crate) const DESKTOP_MINIMUM: GlVersion = GlVersion::new(4, 0);
    pub(crate) const EMBEDDED_MINIMUM: GlVersion = GlVersion::new(3, 0);

    /// Whether this context meets Fluxel's GL-family baseline.
    pub(crate) const fn accepts(self) -> bool {
        match self {
            Self::Desktop { major, minor } => {
                Self::DESKTOP_MINIMUM.is_met_by(GlVersion::new(major, minor))
            }
            Self::Embedded { major, minor } => {
                Self::EMBEDDED_MINIMUM.is_met_by(GlVersion::new(major, minor))
            }
            Self::WebGl2 => true,
        }
    }

    /// Whether a feature has a native core route in this exact profile.
    pub(crate) const fn has_core(self, desktop: GlVersion, embedded: GlVersion) -> bool {
        match self {
            Self::Desktop { major, minor } => desktop.is_met_by(GlVersion::new(major, minor)),
            Self::Embedded { major, minor } => embedded.is_met_by(GlVersion::new(major, minor)),
            Self::WebGl2 => false,
        }
    }

    /// Returns the native core version, if this is not a browser profile.
    pub const fn version(self) -> Option<GlVersion> {
        match self {
            Self::Desktop { major, minor } | Self::Embedded { major, minor } => {
                Some(GlVersion::new(major, minor))
            }
            Self::WebGl2 => None,
        }
    }

    /// Returns whether this profile meets the requirement for its own family.
    pub const fn meets(self, desktop: Option<GlVersion>, embedded: Option<GlVersion>) -> bool {
        match self {
            Self::Desktop { major, minor } => match desktop {
                Some(required) => required.is_met_by(GlVersion::new(major, minor)),
                None => false,
            },
            Self::Embedded { major, minor } => match embedded {
                Some(required) => required.is_met_by(GlVersion::new(major, minor)),
                None => false,
            },
            Self::WebGl2 => false,
        }
    }
}
