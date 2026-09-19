//! Buffer usage, support, descriptor, object, and range (specification sections
//! 11.1, 11.2, and 12.1 through 12.4).
//!
//! A P0 buffer is byte-addressed and opaque. This module owns what a caller may
//! state about one before it exists ([`BufferDescriptor`]), what the device
//! answers about a usage combination ([`BufferSupport`]), and what a caller may
//! learn about one afterwards ([`Buffer`]'s accessors).
//!
//! # What this module does not own
//!
//! - Whether a *copy route* between two buffers is legal is
//!   [`crate::api::resource::route`], not a buffer fact. Section 12.1 gives the
//!   reason the buffer needs its own query anyway: on a backend where storage
//!   buffers do not exist, `BufferSupportQuery::new(BufferUsage::STORAGE)` must
//!   answer `Unsupported` up front, instead of creating the buffer and failing
//!   later at a bind group — which is the same "no second set of conditions"
//!   rule the texture chapter applies to its own query.
//! - A buffer's *contents*, mapping, and host visibility are not here. Section
//!   11.2 removes the old `host_access` / `HostAccessIntent` surface because P0
//!   has no mapping semantics to honor; upload and readback are the two
//!   supported mutation paths and they live in
//!   [`crate::api::resource::transfer`].
//! - The element stride is not here either. Section 12.2 deletes
//!   `element_stride_hint` because a byte-addressed P0 buffer has no elements,
//!   and a structured stride belongs to a future `BufferView`.
//!
//! # Validation
//!
//! The portable rules are `validate_buffer_descriptor` and
//! `validate_buffer_range`. They take the capability answers they need as
//! parameters rather than reading a device, which keeps the rule decidable
//! without a backend (root section 4) and testable without a GPU.

use core::fmt;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;

/// What a buffer will be used for.
///
/// A creation-time correctness contract, not a hint: section 11.1 makes the
/// declared bits the only thing that authorizes the matching operation, so a
/// buffer created without `COPY_DST` cannot be an upload destination even on a
/// platform whose driver "happens to allow it". A backend may therefore not
/// bypass portable usage validation.
///
/// A hand-rolled bitset rather than the `bitflags` crate, which section 11.1
/// rules out for the public surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferUsage(u32);

impl BufferUsage {
    /// Copy source.
    pub const COPY_SRC: Self = Self(1 << 0);
    /// Copy destination, including upload.
    pub const COPY_DST: Self = Self(1 << 1);
    /// Vertex buffer binding.
    pub const VERTEX: Self = Self(1 << 2);
    /// Index buffer binding.
    pub const INDEX: Self = Self(1 << 3);
    /// Uniform buffer binding.
    pub const UNIFORM: Self = Self(1 << 4);
    /// Storage buffer binding.
    pub const STORAGE: Self = Self(1 << 5);

    /// Whether every bit set in `other` is set in `self`.
    ///
    /// An empty `other` is contained in everything, which is what makes the
    /// "usage must not be empty" rule at creation meaningful: nothing here
    /// rejects an empty *query*, and the descriptor validation is the one place
    /// that does.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two usage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no usage bit is set.
    ///
    /// A buffer with an empty usage set has no legal operation at all, which is
    /// why creation refuses it (section 12.3).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for BufferUsage {
    /// Renders the set as `VERTEX|INDEX`, or `<none>` when empty.
    ///
    /// Diagnostic text for errors and logs. Section 11.1 does not declare this
    /// impl; it is added because a refused descriptor must be able to say *which*
    /// usage combination was refused, and a raw `u32` cannot.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = [
            (Self::COPY_SRC, "COPY_SRC"),
            (Self::COPY_DST, "COPY_DST"),
            (Self::VERTEX, "VERTEX"),
            (Self::INDEX, "INDEX"),
            (Self::UNIFORM, "UNIFORM"),
            (Self::STORAGE, "STORAGE"),
        ];
        let mut written = false;
        for (bit, name) in names {
            if self.contains(bit) {
                if written {
                    formatter.write_str("|")?;
                }
                formatter.write_str(name)?;
                written = true;
            }
        }
        if !written {
            formatter.write_str("<none>")?;
        }
        Ok(())
    }
}

/// Where a resource should be placed, when the backend has a choice.
///
/// Section 11.2 keeps exactly one member, and keeps it a *preference*: it is not
/// a correctness guarantee, and a UMA, WebGPU, or GL backend may treat it as
/// equivalent to [`Self::Automatic`] or ignore it.
///
/// The old `HostAccessIntent` / `HostPreferred` pair is deleted rather than
/// deprecated, because P0 exposes no mapping and therefore has no user semantics
/// to honor. A future General Mapping feature freezes host visibility,
/// persistent mapping, coherency, and flush/invalidate together; until then
/// nothing may take advantage of the gap.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceMemoryPreference {
    /// Chosen by the backend.
    Automatic,

    /// Prefer GPU/device-local placement where possible.
    ///
    /// This is a preference, not a correctness guarantee.
    /// UMA / WebGPU / GL backends may treat it equivalently or ignore it.
    DeviceLocalPreferred,
}

/// The key of a "can this buffer be created" question.
///
/// Section 12.1 requires the same level of fact for buffers as
/// [`crate::api::format::TextureSupportQuery`] gives textures, because a device
/// may support some usages and not others — WebGL2 has vertex, index, uniform,
/// and copy buffers but no storage buffers — and without a query there is no
/// stable entry point to ask.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSupportQuery {
    usage: BufferUsage,
}

impl BufferSupportQuery {
    /// Asks about one usage combination.
    pub fn new(usage: BufferUsage) -> Self {
        Self { usage }
    }

    /// The queried usage mask.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

/// The maxima a supported buffer query returns.
///
/// Section 12.1 keeps the size out of the query key for the same reason the
/// texture chapter keeps extent out of its key: the answer is a ceiling, and
/// asking again per size would make the capability cache answer an unbounded
/// number of questions about one usage combination.
#[derive(Clone, Copy, Debug)]
pub struct BufferSupportLimits {
    max_size: u64,
}

impl BufferSupportLimits {
    /// Records one usage combination's ceiling.
    ///
    /// Crate-private: the number is a probed device answer, and a caller-built
    /// one would be a capability claim about hardware nobody asked.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the device façade builds this when it answers a buffer query"
        )
    )]
    pub(crate) fn new(max_size: u64) -> Self {
        Self { max_size }
    }

    /// The largest buffer size this usage combination permits.
    pub fn max_size(&self) -> u64 {
        self.max_size
    }
}

/// Whether a buffer described by a [`BufferSupportQuery`] can be created.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum BufferSupport {
    /// No buffer with this usage combination can be created on this device.
    Unsupported,

    /// The usage combination is creatable, within the returned ceiling.
    Supported(BufferSupportLimits),
}

impl BufferSupport {
    /// Whether the usage combination is creatable.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// The ceiling for the usage combination, or `None` when it is unsupported.
    pub fn limits(&self) -> Option<&BufferSupportLimits> {
        match self {
            Self::Unsupported => None,
            Self::Supported(limits) => Some(limits),
        }
    }
}

/// Everything a caller states about a buffer before it exists.
///
/// `#[non_exhaustive]` so that a future field — a placement hint beyond
/// [`ResourceMemoryPreference`], or the `BufferView` stride section 12.2 defers
/// — is not a breaking change for a caller who built one.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct BufferDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,
    /// Size in bytes. Byte-addressed: P0 has no element type here.
    pub size: u64,
    /// What the buffer will be used for. Must not be empty.
    pub usage: BufferUsage,
    /// A performance preference only; it is never a correctness guarantee.
    pub memory: ResourceMemoryPreference,
}

impl BufferDescriptor {
    /// Describes a buffer with the backend's own placement choice.
    pub fn new(size: u64, usage: BufferUsage) -> Self {
        Self {
            label: Label::default(),
            size,
            usage,
            memory: ResourceMemoryPreference::Automatic,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// States a placement preference.
    ///
    /// Note what this cannot do: it cannot ask for a CPU-visible or mapped
    /// buffer, because section 11.2 deleted that surface. The only two supported
    /// mutation paths are upload and readback.
    pub fn with_memory_preference(mut self, preference: ResourceMemoryPreference) -> Self {
        self.memory = preference;
        self
    }
}

/// A created buffer.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. Cloning is not a second buffer: every
/// clone refers to the same logical object, and section 18.6 keeps the native
/// backing alive until the last logical owner is gone *and* every accepted GPU
/// work item referencing it is terminal.
#[derive(Clone)]
pub struct Buffer {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BufferDescriptor,
}

impl Buffer {
    /// Assembles a created buffer.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`crate::api::platform::Device::create_buffer`] may produce one. That
    /// verb exists and is the only caller this is written for, but it stops before
    /// allocating — nothing can mint the identity below until a backend allocator
    /// does — so nothing calls this yet and the attribute stays.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_buffer calls this once the backend allocator lands and can mint the buffer's identity"
        )
    )]
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, descriptor: BufferDescriptor) -> Self {
        Self {
            id,
            device,
            descriptor,
        }
    }

    /// This buffer's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this buffer.
    ///
    /// Section 3.3 makes this the only answer to a cross-device use: there is no
    /// implicit copy, binding, handle unwrap, staging bridge, or peer transfer,
    /// so the comparison in `validate_buffer_ownership` is a refusal rather
    /// than a migration.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The descriptor this buffer was created from.
    ///
    /// Section 18.8 requires a descriptor to be recoverable for capture, which is
    /// why the buffer retains it rather than only its effects.
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (adjudication A16): descriptors in this
/// chapter are `#[derive(Clone, Debug)]` and contain a [`Buffer`], so a handle
/// must be printable, but section 7.1 describes an object by its identity rather
/// than its contents. The backend port will add a native field that has no
/// reason to be `Debug`, and printing a native handle into a log would leak it.
/// `finish_non_exhaustive()` is what makes it honest that the descriptor is not
/// shown — a caller who needs it calls [`Buffer::descriptor`].
impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// The creation verb of this chapter. Section 12.3 declares `create_buffer` beside
// the type it produces, and adjudication A28 keeps it here rather than in
// `api::platform`: the definition site is the owner. Rust attaches an inherent
// method to the type wherever its impl block is written in the defining crate, so
// `crate::api::platform::Device::create_buffer` resolves to this method, and the
// links other modules write to that path keep working.
impl Device {
    /// Creates a buffer.
    ///
    /// Section 12.3's creation verb. It is an inherent method written in the
    /// resource chapter rather than in `api::platform` because section 12.3
    /// declares it beside the object it produces and because [`Buffer`] is this
    /// chapter's type: the definition site is the owner (adjudication A28), and a
    /// verb collected into `api::platform` instead would make that module the one
    /// file that must know about every resource in the crate.
    ///
    /// The portable refusals happen before the stop, in section 12.3's own order:
    /// the descriptor's rules first, then the device's answer to the
    /// [`BufferSupportQuery`] its usage builds. Section 4 forbids handing a defect
    /// portable validation can find to a driver for it to discover, and section
    /// 3.1 forbids touching a backend before the portable checks have run — so what
    /// panics here is the allocation, never the validation.
    ///
    /// The support answer is read from [`Device::capabilities`] rather than
    /// assumed, because a `BufferSupport` built by hand would be a capability claim
    /// about hardware nobody asked — the same reason
    /// [`BufferSupportLimits`] mints its value crate-private.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::InvalidUsage`] when the descriptor is inconsistent with
    /// itself — a size of zero, an empty usage set, or a size past the ceiling the
    /// device reports — and [`RhiErrorKind::Unsupported`] when the device cannot
    /// express the usage combination at all, which is not the caller's mistake.
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> RhiResult<Buffer> {
        // Section 6.5: a lost device refuses creation itself. It is also the
        // first verdict this verb can reach on a tree with no backend port,
        // because it returns before the capability read below.
        self.require_active()?;

        let support = self
            .capabilities()
            .buffer_support(&BufferSupportQuery::new(desc.usage));
        validate_buffer_descriptor(desc, &support)?;
        unimplemented!(
            "Device::create_buffer needs a backend allocator to allocate the {} bytes \
             this descriptor asks for; the portable contract is fixed and its refusal \
             paths above are built, but no backend port is built. The capability \
             snapshot this verb reads its buffer-support answer from is a backend-port \
             deliverable as well, so on today's tree the call stops inside \
             Device::capabilities before reaching this point",
            desc.size
        )
    }
}

/// A byte range within one buffer.
///
/// P0 has no `WHOLE_BUFFER` sentinel (section 12.4): a caller writes the size it
/// means, and every point of use checks the range against the object. The reason
/// is that a sentinel makes the *checked* range depend on which object it is
/// resolved against, which is exactly the fact a cross-device or stale-range bug
/// hides in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferRange {
    /// Byte offset of the first byte.
    pub offset: u64,
    /// Number of bytes covered.
    pub size: u64,
}

impl BufferRange {
    /// Builds a range. Checks nothing: the range is validated where it is used,
    /// against the buffer it will be resolved against.
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// The exclusive end offset, or `None` when `offset + size` overflows `u64`.
    ///
    /// Returning `Option` rather than wrapping is the point: section 12.4 lists
    /// "offset + size has no integer overflow" as a rule every point of use must
    /// check, and an accessor that wrapped would let a caller compare a wrapped
    /// end against a buffer size and conclude the range was fine.
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size)
    }
}

/// A buffer together with the sub-range a binding uses.
#[derive(Clone)]
pub struct BufferBinding {
    /// The buffer being bound. Held by clone: the binding is a logical owner.
    pub buffer: Buffer,
    /// The range within it.
    pub range: BufferRange,
}

impl BufferBinding {
    /// Pairs a buffer with a range. Checks nothing: the range is validated where
    /// it is used, against the buffer it will be resolved against, and the
    /// binding rules where the bind group is created.
    ///
    /// The first half is `validate_buffer_range`'s — section 12.4's `size > 0`,
    /// no overflow at `offset + size`, and `offset + size <= buffer.size` — and
    /// it needs the buffer's size, which is why it runs there rather than here.
    /// The second half is the binding chapter's, at bind-group creation: section
    /// 22.3's `UNIFORM`/`STORAGE` usage, the layout's minimum size, the device's
    /// `max_uniform_buffer_binding_size` / `max_storage_buffer_binding_size`,
    /// and offset alignment against the device's limits. A pair built here is
    /// therefore a candidate binding, and nothing more.
    pub fn new(buffer: Buffer, range: BufferRange) -> Self {
        Self { buffer, range }
    }
}

/// Checks a descriptor against the capability answer it must respect.
///
/// Section 12.3's list, in its own order:
///
/// ```text
/// size > 0
/// usage non-empty
/// BufferSupportQuery(usage) == Supported
/// size <= BufferSupportLimits.max_size
/// ```
///
/// The first, second, and fourth are descriptor constraints
/// ([`RhiErrorKind::InvalidUsage`]); the third is a device refusal
/// ([`RhiErrorKind::Unsupported`]), because a usage combination the device
/// cannot express is not the caller's mistake.
///
/// `DeviceIdentity` is the one entry of that list this function cannot check: it
/// is a comparison between the descriptor's buffer and the target device, so it
/// needs both and is checked by [`validate_buffer_ownership`].
pub(crate) fn validate_buffer_descriptor(
    desc: &BufferDescriptor,
    support: &BufferSupport,
) -> RhiResult<()> {
    if desc.size == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "buffer size must be greater than zero",
        ));
    }
    if desc.usage.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "buffer usage must not be empty",
        ));
    }
    let limits = match support {
        BufferSupport::Unsupported => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device cannot create a buffer with usage {}",
                    desc.usage
                ),
            ));
        }
        BufferSupport::Supported(limits) => limits,
    };
    if desc.size > limits.max_size() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "buffer size {} exceeds the supported maximum {} for usage {}",
                desc.size,
                limits.max_size(),
                desc.usage
            ),
        ));
    }
    Ok(())
}

/// Checks that a range lies inside a buffer of `buffer_size` bytes.
///
/// Section 12.4's first three rules:
///
/// ```text
/// size > 0
/// offset + size has no integer overflow
/// offset + size <= buffer.size
/// ```
///
/// The fourth rule in that list — "corresponding binding/copy alignment" — is a
/// property of the *use*, not of the range, and is checked by the alignment
/// limits of the route that will carry it
/// ([`crate::api::resource::route::BufferCopyLayoutLimits::validate`]).
pub(crate) fn validate_buffer_range(range: BufferRange, buffer_size: u64) -> RhiResult<()> {
    if range.size == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "buffer range size must be greater than zero",
        ));
    }
    let end = range.end().ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "buffer range offset {} plus size {} overflows u64",
                range.offset, range.size
            ),
        )
    })?;
    if end > buffer_size {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("buffer range ends at {end}, past the buffer size {buffer_size}"),
        ));
    }
    Ok(())
}

/// Checks that a buffer belongs to the device an operation targets.
///
/// Section 3.1 requires this comparison first and in O(1), before any backend is
/// touched, and section 3.3 fixes the answer as
/// [`RhiErrorKind::WrongDevice`]: there is no implicit copy, binding, handle
/// unwrap, staging bridge, or peer transfer in P0, so a foreign buffer is a
/// refusal and never a migration.
pub(crate) fn validate_buffer_ownership(buffer: &Buffer, target: DeviceIdentity) -> RhiResult<()> {
    if buffer.device_identity() != target {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "buffer belongs to a different device",
        )
        .with_object(buffer.id()));
    }
    Ok(())
}
