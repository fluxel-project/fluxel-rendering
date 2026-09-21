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
//! - A buffer's contents and host visibility do not appear in its descriptor.
//!   Explicit host mapping is a separate asynchronous lease in
//!   [`crate::api::resource::mapping`]; upload and readback remain the two
//!   transfer facilities in [`crate::api::resource::transfer`].
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
use std::sync::{Arc, Mutex};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::resource::backend::BufferBackend;
use crate::api::resource::transient::TransientResourceMetadata;

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
    /// Indirect draw or dispatch argument source.
    pub const INDIRECT: Self = Self(1 << 6);
    /// Destination for resolved query results.
    pub const QUERY_RESOLVE: Self = Self(1 << 7);
    /// CPU map-read source. Mapping is an explicitly negotiated capability.
    pub const MAP_READ: Self = Self(1 << 8);
    /// CPU map-write destination. Mapping is an explicitly negotiated capability.
    pub const MAP_WRITE: Self = Self(1 << 9);
    /// Bottom-level acceleration-structure build input.
    pub const BLAS_INPUT: Self = Self(1 << 10);
    /// Top-level acceleration-structure instance input.
    pub const TLAS_INPUT: Self = Self(1 << 11);
    /// Scratch storage used exclusively by acceleration-structure build/update.
    ///
    /// Native lowerings may use its stronger alignment requirement without
    /// treating arbitrary storage buffers as valid scratch allocations.
    pub const ACCELERATION_STRUCTURE_SCRATCH: Self = Self(1 << 12);

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

    /// Whether no bit outside `allowed` is present. Backend capability tables
    /// use this to reject native heap combinations that cannot be represented
    /// safely; it deliberately remains crate-private rather than exposing a
    /// second public bitset algebra spelling.
    pub(crate) fn is_subset_of(self, allowed: Self) -> bool {
        self.0 & !allowed.0 == 0
    }

    /// Every bit this type defines, as one mask.
    ///
    /// Built from the constants rather than written as `63`: a seventh usage bit
    /// added above widens this with no second edit, which is what keeps
    /// [`Self::all`] from silently ceasing to be an enumeration of the space.
    ///
    /// Carries no dead-code expectation even though its only reader is gated.
    /// Rustc counts a constant as live once any function body reads it, without
    /// asking whether that function is itself live, so an expectation here is
    /// unfulfilled in the very configuration the gate is for.
    const ALL_BITS: u32 = Self::COPY_SRC.0
        | Self::COPY_DST.0
        | Self::VERTEX.0
        | Self::INDEX.0
        | Self::UNIFORM.0
        | Self::STORAGE.0
        | Self::INDIRECT.0
        | Self::QUERY_RESOLVE.0
        | Self::MAP_READ.0
        | Self::MAP_WRITE.0
        | Self::BLAS_INPUT.0
        | Self::TLAS_INPUT.0
        | Self::ACCELERATION_STRUCTURE_SCRATCH.0;

    /// Every usage combination, including the empty one, in mask order.
    ///
    /// Crate-private, and a backend enumerating a buffer-support table is its
    /// caller. That table's completeness rule is a statement about *this key
    /// space* rather than about any one entry — section 7.2's rule makes an
    /// absent entry there a hole rather than an answer — so the walk belongs
    /// beside the layout it walks, where it cannot drift out of step with it.
    ///
    /// The empty mask is included. It is a query a caller can construct, so
    /// leaving it out of the walk would leave it out of whatever table the walk
    /// fills, which is the one outcome the completeness rule exists to prevent.
    #[cfg_attr(
        all(
            not(test),
            not(any(feature = "dx12", feature = "vulkan", feature = "gl-family"))
        ),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports enumerate this complete key space; without either feature it is unreachable outside tests"
        )
    )]
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        (0..=Self::ALL_BITS).map(Self)
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
            (Self::INDIRECT, "INDIRECT"),
            (Self::QUERY_RESOLVE, "QUERY_RESOLVE"),
            (Self::MAP_READ, "MAP_READ"),
            (Self::MAP_WRITE, "MAP_WRITE"),
            (Self::BLAS_INPUT, "BLAS_INPUT"),
            (Self::TLAS_INPUT, "TLAS_INPUT"),
            (
                Self::ACCELERATION_STRUCTURE_SCRATCH,
                "ACCELERATION_STRUCTURE_SCRATCH",
            ),
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
/// Section 11.2 keeps resource placement deliberately narrow and treats it as a
/// *preference*: it is not a correctness guarantee, and a UMA, WebGPU, or GL
/// backend may treat it as equivalent to [`Self::Automatic`] or ignore it.
///
/// The old `HostAccessIntent` / `HostPreferred` pair stays deleted: mapping is
/// governed by explicit usage bits, capability facts, and an asynchronous lease
/// rather than a placement promise in this descriptor. Persistent mapping and
/// coherency remain separately capability-gated mapping semantics.
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

/// Device-level allocator policy.
///
/// It is a performance hint only. It cannot enable an otherwise unsupported
/// resource, alter its visibility semantics, or expose a native heap model.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// Backend-selected policy.
    #[default]
    Automatic,
    /// Prefer throughput and device-local residency.
    Performance,
    /// Prefer smaller allocator reservation/commitment.
    MemoryUsage,
    /// Permit backend-private suballocation where it is already correct.
    ManualSuballocation,
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
        all(not(test), not(any(feature = "dx12", feature = "vulkan"))),
        expect(
            dead_code,
            reason = "the DX12 and Vulkan capability ports construct probed buffer ceilings; without either backend this is test-only"
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
    inner: Arc<BufferInner>,
}

/// The one shared ownership domain of a logical buffer.
///
/// Keeping identity, immutable descriptor, transient metadata, and the native
/// allocation together is deliberate: a cloned `Buffer` is one logical
/// resource, not several independently reference-counted pieces of one.
struct BufferInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: BufferDescriptor,
    /// The allocation behind this handle, shared with every clone of it.
    ///
    /// Section 18.6 makes a clone the same logical object rather than a second
    /// buffer, so this is an `Arc` and not a copy of anything: two clones that
    /// each owned an allocation would be two buffers wearing one identity, and
    /// the last-owner rule would have no single moment to retire.
    ///
    /// Every method here answers from the fields above it, and the backend is
    /// reached only by the *device* verbs of the later chapters — which is why
    /// this field is unread from inside this module without being dead: it is
    /// what [`Self::native`] hands to them.
    ///
    /// The expectation sits on that accessor and not here, and that placement was
    /// measured rather than assumed. rustc reads the field as used because the
    /// accessor reads it, and an `expect` on the field therefore sits
    /// unfulfilled; what is genuinely unreached is the accessor, which is where
    /// the reason is written.
    native: Box<dyn BufferBackend>,
    /// Plan-scoped transient execution metadata, when this is not persistent.
    transient: Option<TransientResourceMetadata>,
    /// Portable mapping exclusivity. A second mapping never reaches native code.
    mapped: Mutex<bool>,
}

impl Buffer {
    /// Acquires this buffer's sole portable mapping lease.
    pub(crate) fn begin_map(&self) -> RhiResult<()> {
        let mut mapped = self
            .inner
            .mapped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *mapped {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "buffer already has an active mapping lease",
            )
            .with_object(self.id()));
        }
        *mapped = true;
        Ok(())
    }

    /// Releases a lease acquired by [`Self::begin_map`].
    pub(crate) fn end_map(&self) {
        *self
            .inner
            .mapped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
    }

    /// Whether a pending or ready host mapping owns the portable lease.
    ///
    /// This is observed during submission Phase A. Ordinary mappings exclude
    /// GPU use; a device must explicitly enable persistent mapping before a
    /// mapped buffer can remain in submitted work.
    pub(crate) fn is_mapped(&self) -> bool {
        *self
            .inner
            .mapped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Assembles a created buffer.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`crate::api::platform::Device::create_buffer`] may produce one, and
    /// the identity is minted there rather than here — a constructor that minted
    /// its own would give a caller who assembled one directly an object the
    /// process has never recorded.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: BufferDescriptor,
        native: Box<dyn BufferBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(BufferInner {
                id,
                device,
                descriptor,
                native,
                transient: None,
                mapped: Mutex::new(false),
            }),
        }
    }

    /// Assembles a logical buffer owned by one transient submission plan.
    pub(crate) fn new_transient(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: BufferDescriptor,
        native: Box<dyn BufferBackend>,
        transient: TransientResourceMetadata,
    ) -> Self {
        Self {
            inner: Arc::new(BufferInner {
                id,
                device,
                descriptor,
                native,
                transient: Some(transient),
                mapped: Mutex::new(false),
            }),
        }
    }

    /// Internal transient lifetime for submission-plan validation.
    pub(crate) fn transient_lifetime(
        &self,
    ) -> Option<&crate::api::resource::transient::TransientLifetime> {
        self.inner
            .transient
            .as_ref()
            .map(TransientResourceMetadata::lifetime)
    }

    /// The native allocation behind this buffer.
    ///
    /// Crate-private for the reason the whole seam is: section 59 keeps native
    /// types off the exported surface, and a caller never learns the platform in
    /// order to write correct code. The device verbs of the later chapters reach
    /// it here and downcast inside their own backend.
    ///
    /// The transfer chapter is that caller, and the DX12 command spine is where it
    /// arrived: it reads this to reach the buffer's own backend type and record a
    /// copy against the native resource. The expectation therefore narrowed from
    /// `not(test)` to the backend feature list rather than being deleted — a build
    /// with no backend compiled genuinely has nothing on this side of the seam.
    #[cfg_attr(
        not(any(
            test,
            // The backend features that actually compile a lowering. A feature
            // that selects nothing must not appear here: it would remove this
            // expectation in a configuration where the item really is dead, and
            // the gate would then be silent about it. When Vulkan lands and starts
            // calling this, its feature joins the list — which is rule 4.6's
            // "the matrix gets the row" applied to the attribute itself.
            feature = "dx12",
            feature = "vulkan"
        )),
        expect(
            dead_code,
            reason = "read by a backend's own lowering, which is the only code that may \
                      cross the seam; a build with no backend compiled has none"
        )
    )]
    pub(crate) fn native(&self) -> &dyn BufferBackend {
        self.inner.native.as_ref()
    }

    /// This buffer's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this buffer.
    ///
    /// Section 3.3 makes this the only answer to a cross-device use: there is no
    /// implicit copy, binding, handle unwrap, staging bridge, or peer transfer,
    /// so the comparison in `validate_buffer_ownership` is a refusal rather
    /// than a migration.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The descriptor this buffer was created from.
    ///
    /// Section 18.8 requires a descriptor to be recoverable for capture, which is
    /// why the buffer retains it rather than only its effects.
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.inner.descriptor
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (adjudication A16): descriptors in this
/// chapter are `#[derive(Clone, Debug)]` and contain a [`Buffer`], so a handle
/// must be printable, but section 7.1 describes an object by its identity rather
/// than its contents. The native field added by the backend port has no `Debug`
/// for the same reason it has no accessor — printing a native handle into a log
/// would leak it — and `finish_non_exhaustive()` is what makes it honest that
/// neither the descriptor nor the allocation is shown. A caller who needs the
/// descriptor calls [`Buffer::descriptor`].
impl fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
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
    /// The portable refusals happen before the allocation, in section 12.3's own
    /// order: the descriptor's rules first, then the device's answer to the
    /// [`BufferSupportQuery`] its usage builds. Section 4 forbids handing a defect
    /// portable validation can find to a driver for it to discover, and section
    /// 3.1 forbids touching a backend before the portable checks have run — so by
    /// the time the backend is asked, the only question left is whether the driver
    /// will satisfy a request that is already known to be well-formed.
    ///
    /// The support answer is read from [`Device::capabilities`] rather than
    /// assumed, because a `BufferSupport` built by hand would be a capability claim
    /// about hardware nobody asked — the same reason
    /// [`BufferSupportLimits`] mints its value crate-private.
    ///
    /// # Where the identity comes from
    ///
    /// `ObjectId::next`, here, and not from the backend. Section 3 gives the id
    /// to the object that created the resource, and the backend is handed the
    /// descriptor and nothing else precisely so that there is one place an id is
    /// minted. The backend seam that receives the descriptor is
    /// `crate::api::resource::backend`, which is crate-private. Both names are written
    /// rather than linked because a rustdoc link to a private item fails the
    /// `-D warnings` doc gate, and linking them would mean publishing the private
    /// surface to satisfy two references.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::InvalidUsage`] when the descriptor is inconsistent with
    /// itself — a size of zero, an empty usage set, or a size past the ceiling the
    /// device reports — [`RhiErrorKind::Unsupported`] when the device cannot
    /// express the usage combination at all, which is not the caller's mistake,
    /// and whatever the backend reports for a native failure of its own —
    /// [`RhiErrorKind::OutOfMemory`] and [`RhiErrorKind::BackendFailure`] are the
    /// two a real allocator produces.
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> RhiResult<Buffer> {
        // Section 6.5: a lost device refuses creation itself, and section 3.1
        // puts it after the ownership verdicts — of which this verb has none,
        // because the device is its receiver and not one of its arguments.
        //
        // Both refusals below are named as this verb's, and the naming is applied
        // here rather than inside the two helpers that produce them. Neither can
        // name an operation honestly: `require_active` is the device's liveness
        // check and is reached from every verb, and `validate_buffer_descriptor`
        // is a rule about a descriptor and has no verb. Section 4's model attaches
        // the name "as the error crosses each layer, so the layer closest to the
        // caller names itself", and this is that layer.
        self.require_active()
            .map_err(|error| error.at("Device::create_buffer"))?;

        let support = self
            .capabilities()
            .buffer_support(&BufferSupportQuery::new(desc.usage));
        validate_buffer_descriptor(desc, &support)
            .map_err(|error| error.at("Device::create_buffer"))?;

        // The one backend call, and the last statement that can fail. Everything
        // above it is a portable verdict about the request; this is the driver's
        // answer to it.
        let native = self.native().create_buffer(desc)?;
        Ok(Buffer::new(
            ObjectId::next(),
            self.identity(),
            desc.clone(),
            native.into(),
        ))
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

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because every field read
// below is private to this module.

impl BufferUsage {
    /// Writes this usage mask's bits, little-endian.
    ///
    /// The bits rather than the mask's `Debug` rendering, because `Debug` is not a
    /// stability contract and the fingerprint is compared across processes. The
    /// mask rather than a member list, because a mask is already canonical.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}

impl BufferSupport {
    /// Writes this answer as a tag, followed by the ceiling when there is one.
    ///
    /// `Unsupported` is tag 0 and carries no body. That is not the same encoding as
    /// `Supported` with a zero ceiling, and the two must never collapse: one says
    /// the buffer cannot exist, the other says it exists and may be empty.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Unsupported => out.push(0),
            Self::Supported(limits) => {
                out.push(1);
                out.extend_from_slice(&limits.max_size.to_le_bytes());
            }
        }
    }
}
