//! The tooling SPI version (rhi-design section 53.1).
//!
//! # What it is
//!
//! One value that names the version of the *interface* in this module: the
//! shape of the observer callback, the subscription linearization rules, and
//! the semantic-event, object-definition, and portable-command vocabulary.
//!
//! # What it deliberately does not own
//!
//! It is **not** a capture artifact schema version. An artifact is a file with
//! its own magic, chunk layout, manifest, and migration story, and none of that
//! belongs to RHI (section 58.3). A tool that writes artifacts records both
//! numbers separately: the artifact schema version it wrote, and the
//! [`TOOLING_SPI_VERSION`] whose semantics it read. A capture produced from SPI
//! 1.0 stays readable after the SPI reaches 2.0; the tool is what decides
//! whether it can still interpret the old semantics.

/// The version of the tooling SPI.
///
/// The major half must change when a change is incompatible with an existing
/// observer: a removed or reinterpreted event variant, a changed definition
/// field meaning, or a changed linearization guarantee. Adding an observer-visible
/// variant to a `#[non_exhaustive]` enum is a minor change, because an existing
/// observer is already required to tolerate variants it does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToolingSpiVersion {
    /// The incompatible-change half.
    pub major: u16,
    /// The compatible-addition half.
    pub minor: u16,
}

/// The tooling SPI version this build implements.
///
/// A tooling consumer compares this with the version it was written against
/// before it interprets anything, because the SPI is not the crate's semver
/// surface: it may move inside a compatible crate release.
pub const TOOLING_SPI_VERSION: ToolingSpiVersion = ToolingSpiVersion { major: 1, minor: 0 };
