//! Basic identity types (specification sections 3, 3.1, and 3.4).
//!
//! Every identifier here is an **opaque token**: a caller may compare, hash, and
//! print it, but cannot construct a valid one. That is not a style preference.
//! Section 3.1 makes the whole portable error model rest on identity comparison
//! — "target `DeviceIdentity` != object `DeviceIdentity` -> `WrongDevice`", "the
//! identity is the same but the Device is lost -> `DeviceLost`" — and a token a
//! caller could mint would let a caller forge a `WrongDevice` answer into an
//! accepted one. The fields are therefore private and the public surface is the
//! accessors below.
//!
//! The constructors are `pub(crate)` rather than absent. The platform layer has
//! to mint identities, and a backend has to mint object IDs; hiding the
//! constructor from *callers* is the requirement, not hiding it from the crate.
//!
//! `Label` is deliberately not a token. It is diagnostic text, it is
//! caller-owned, and section 19.8 excludes labels from every canonical hash, so
//! a public constructor costs nothing.

use core::fmt;

/// Opaque identity of one RHI instance within this process.
///
/// ```compile_fail
/// use fluxel_rhi::api::DeviceInstanceId;
/// // A caller cannot mint an instance identity.
/// let _ = DeviceInstanceId::new(1);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceInstanceId(u64);

impl DeviceInstanceId {
    /// Mints an instance identity.
    ///
    /// Crate-private: only the platform layer that opened the instance may mint
    /// one. The value is process-local and never derived from a native handle.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the platform layer that opens an instance is not written"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the process-local value for diagnostics and logging.
    ///
    /// The value is not stable across processes and must not be used as a
    /// serialized identity; see [`ObjectId`] for the same rule.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Opaque generation within a Fluxel logical Device execution domain.
///
/// A caller may compare, hash, and print this token, but cannot construct a
/// valid generation. It is part of public identity, not a recovery counter that
/// callers or backends may increment transparently: section 3.1 removes
/// generation++ recovery from P0 entirely, and section 65.1 records the decision
/// that a lost device is terminal and re-requested rather than revived.
///
/// ```compile_fail
/// use fluxel_rhi::api::DeviceGeneration;
/// let _ = DeviceGeneration::new(2);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceGeneration(u64);

impl DeviceGeneration {
    /// Mints a generation.
    ///
    /// Crate-private because the *only* legal mint is "one new generation domain
    /// for a device that was just requested". Nothing in the crate may mint a
    /// generation in order to revive an existing identity — section 3.1 lists
    /// that under "P0 None".
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the process-local value for diagnostics and logging.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A Fluxel logical Device execution domain.
///
/// P0 provides no transparent device recovery: device loss is terminal and
/// re-requesting a device obtains a new identity and a new generation domain.
/// Two `Device`s may therefore share a [`DeviceInstanceId`] — `Device::clone()`
/// yields the same identity — while a standalone `request_device()` always
/// yields a new one.
///
/// Every operation that accepts a resource and a target device compares this
/// value first, in O(1), before any backend is touched (section 3.1).
///
/// ```compile_fail
/// use fluxel_rhi::api::DeviceIdentity;
/// let _ = DeviceIdentity::new(1, 1);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceIdentity {
    instance: DeviceInstanceId,
    generation: DeviceGeneration,
}

impl DeviceIdentity {
    /// Composes the identity of one logical device execution domain.
    pub(crate) fn new(instance: DeviceInstanceId, generation: DeviceGeneration) -> Self {
        Self {
            instance,
            generation,
        }
    }

    /// Returns the instance component.
    pub fn instance(self) -> DeviceInstanceId {
        self.instance
    }

    /// Returns the generation component.
    pub fn generation(self) -> DeviceGeneration {
        self.generation
    }
}

/// The in-process logical ID of an RHI object.
///
/// - Not equal to a native handle.
/// - Globally unique within the process.
/// - Cross-process stability is not guaranteed; capture artifacts reassign
///   their own capture-local typed IDs (section 52.1).
///
/// This is the identifier carried by [`super::RhiError::object`] and by every
/// tooling definition, so it is the one identity that appears in both the
/// ordinary and the tooling surface.
///
/// ```compile_fail
/// use fluxel_rhi::api::ObjectId;
/// let _ = ObjectId::new(1);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(u64);

impl ObjectId {
    /// Mints the process-local ID of a newly created object.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the contract tests; the object-creating operations that mint one are not written"
        )
    )]
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the process-local value for diagnostics and logging.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Human-facing diagnostic text attached to an object or a descriptor.
///
/// A label carries no semantics: it is excluded from canonical hashing
/// (section 19.8), it never participates in identity comparison, and capture
/// does not reconstruct it as correctness-relevant state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Label(pub Option<String>);

impl Label {
    /// Returns the label text, if any.
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl fmt::Display for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(text) => formatter.write_str(text),
            None => formatter.write_str("<unlabeled>"),
        }
    }
}
