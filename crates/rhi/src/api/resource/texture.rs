//! Texture usage, shape, creation compatibility, descriptor, and object
//! (specification sections 11.1 and 13.1 through 13.4).
//!
//! This module owns one chapter: what a caller may state about a texture before
//! it exists, and what may be learned about one afterwards.
//!
//! # What this module does not own
//!
//! - Whether the device can create it is
//!   [`crate::api::format::TextureSupport`], which owns the descriptor-dependent
//!   query and its limits. This module *asks* it
//!   (`validate_texture_descriptor`) rather than restating its rules, because
//!   section 13.3 forbids the three-sets-of-conditions mistake: the capability
//!   query, the creation validation, and the backend's image creation must
//!   consult one description, not three.
//! - Whether a *view* of it is legal is
//!   [`crate::api::resource::view`], which is the only code that knows the
//!   aspect, dimension, and range the view will actually use.
//! - Subresource ranges, origins, and host byte layout are
//!   [`crate::api::resource::subresource`], because the copy/tracking vocabulary
//!   is shared with upload and readback and is not a texture property.
//!
//! # The three questions that look like one
//!
//! Section 13 keeps them apart, and the separation is the reason the descriptor
//! is shaped the way it is:
//!
//! ```text
//! portable invariants        "is this descriptor self-consistent?"
//!                            (extent vs dimension, mips vs extent, CUBE intent)
//! capability query           "can this device make one?"
//!                            (section 8.3, keyed on dimension/format/usage/samples)
//! creation limits            "is this extent/mip/layer count within the ceiling?"
//! ```
//!
//! A descriptor that fails the first is wrong everywhere; one that fails the
//! second is wrong on this device; one that fails the third is too large here.
//! Collapsing them would make a portable rule report itself as a device
//! limitation.

use core::fmt;
use std::any::Any;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupport, TextureSupportQuery};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::resource::backend::TextureBackend;
use crate::api::resource::buffer::ResourceMemoryPreference;
use crate::api::resource::transient::TransientResourceMetadata;

/// What a texture will be used for.
///
/// A creation-time correctness contract, like [`crate::api::resource::buffer::
/// BufferUsage`] and for the same reason: a texture created without `COPY_DST`
/// cannot be an upload destination, one without `COPY_SRC` cannot be a readback
/// source, and one without `STORAGE` cannot be a storage binding, regardless of
/// what the platform's driver would tolerate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureUsage(u32);

impl TextureUsage {
    /// Copy source, including readback.
    pub const COPY_SRC: Self = Self(1 << 0);
    /// Copy destination, including upload.
    pub const COPY_DST: Self = Self(1 << 1);
    /// Sampled texture binding.
    pub const SAMPLED: Self = Self(1 << 2);
    /// Storage texture binding.
    pub const STORAGE: Self = Self(1 << 3);
    /// Color attachment.
    pub const COLOR_ATTACHMENT: Self = Self(1 << 4);
    /// Depth and/or stencil attachment.
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 5);

    /// Whether every bit set in `other` is set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two usage sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no usage bit is set.
    ///
    /// A texture with an empty usage set has no legal operation at all, which is
    /// why creation refuses it (section 13.4).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The union of every declared usage bit, in mask order.
    ///
    /// Carries no dead-code expectation even though its only reader is gated.
    /// Rustc counts a constant as live once any function body reads it, without
    /// asking whether that function is itself live, so an expectation here is
    /// unfulfilled in the very configuration the gate is for. The same asymmetry
    /// is documented on `BufferUsage::ALL_BITS`.
    const ALL_BITS: u32 = Self::COPY_SRC.0
        | Self::COPY_DST.0
        | Self::SAMPLED.0
        | Self::STORAGE.0
        | Self::COLOR_ATTACHMENT.0
        | Self::DEPTH_STENCIL_ATTACHMENT.0;

    /// Every usage combination, including the empty one, in mask order.
    ///
    /// The counterpart of `BufferUsage::all`, and for the same reason: a backend
    /// probing which textures a device can create is asked about a usage mask,
    /// and a capability table keyed on a space the backend can walk in full must
    /// be filled in full. Section 13.4's P0 set is six bits, so the walk is
    /// sixty-four masks.
    #[cfg_attr(
        all(not(test), not(feature = "dx12")),
        expect(
            dead_code,
            reason = "the DX12 capability port is the only caller, and it is compiled out without the dx12 feature"
        )
    )]
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        (0..=Self::ALL_BITS).map(Self)
    }
}

impl fmt::Display for TextureUsage {
    /// Renders the set as `SAMPLED|COLOR_ATTACHMENT`, or `<none>` when empty.
    ///
    /// Diagnostic text for errors and logs. Section 11.1 does not declare this
    /// impl; it is added because a refused descriptor must be able to say *which*
    /// usage combination was refused, and a raw `u32` cannot.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = [
            (Self::COPY_SRC, "COPY_SRC"),
            (Self::COPY_DST, "COPY_DST"),
            (Self::SAMPLED, "SAMPLED"),
            (Self::STORAGE, "STORAGE"),
            (Self::COLOR_ATTACHMENT, "COLOR_ATTACHMENT"),
            (Self::DEPTH_STENCIL_ATTACHMENT, "DEPTH_STENCIL_ATTACHMENT"),
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

/// The dimensionality of a texture.
///
/// Three cases, not four: P0 has no 1D array and no 2D array *texture* type,
/// because array layers are a property of every [`TextureDescriptor`] rather
/// than a separate dimension. Section 13.1 fixes the invariants each case must
/// satisfy; `validate_texture_descriptor` enforces them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    /// A 1D texture: `height = depth = array_layers = sample_count = 1`.
    D1,
    /// A 2D texture, possibly with array layers. `depth = 1`.
    D2,
    /// A 3D texture: `array_layers = sample_count = 1`, and Z slices are
    /// addressed by origin/extent rather than as array subresources.
    D3,
}

/// A three-component extent.
///
/// Public fields, unlike almost everything else in this chapter, because an
/// extent is arithmetic rather than identity: callers compute with it, and
/// hiding it behind accessors would force every arithmetic expression through
/// getters without protecting an invariant — every `u32` triple is a
/// syntactically valid extent, and which triples are *legal* depends on the
/// dimension, which this type does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent3d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels. Always 1 for a 1D texture.
    pub height: u32,
    /// Depth in texels. Always 1 for 1D and 2D textures.
    pub depth: u32,
}

impl Extent3d {
    /// A 1D extent.
    pub fn d1(width: u32) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// A 2D extent.
    pub fn d2(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// A 3D extent.
    pub fn d3(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// The largest component, or 1 when every component is zero.
    ///
    /// Used for the mip ceiling, which section 13.1 defines over
    /// `max(width, height, depth)`. The zero case is unreachable through
    /// validation (every component must be greater than zero) but is defined
    /// here so this stays a total function rather than a panic.
    pub fn max_component(&self) -> u32 {
        self.width.max(self.height).max(self.depth).max(1)
    }
}

/// View intents that must be declared when the texture is created.
///
/// This is not an alias for [`crate::api::resource::view::TextureViewDimension`]
/// and must not become one. Section 13.2 gives the reason: a Vulkan cube view
/// requires the *image* to be created with cube-compatible semantics, and that
/// flag cannot be added later by `create_texture_view`. A view dimension
/// describes the view; this describes a permission the creation step had to
/// grant. A future 3D-to-2D sliced view, block-texel reinterpretation, or planar
/// view adds its own bit here rather than quietly borrowing the view type and
/// hoping the backend can still comply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureViewCompatibility(u32);

impl TextureViewCompatibility {
    /// No special view intent. The texture may not be viewed as a cube.
    pub const NONE: Self = Self(0);

    /// Texture permits creation of Cube / CubeArray views.
    pub const CUBE: Self = Self(1 << 0);

    /// Whether every bit set in `other` is set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two compatibility sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Everything a caller states about a texture before it exists.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// Dimensionality. Selects which of the shape invariants apply.
    pub dimension: TextureDimension,
    /// Size of mip level 0.
    pub extent: Extent3d,

    /// Number of mip levels, including level 0. Must be at least 1.
    pub mip_levels: u32,
    /// Number of array layers. Must be at least 1.
    pub array_layers: u32,
    /// Samples per texel. Must be at least 1; more than 1 means multisampled.
    pub sample_count: u32,

    /// The texture's own format.
    pub format: TextureFormat,
    /// What the texture will be used for. Must not be empty.
    pub usage: TextureUsage,

    /// Set of formats allowed for alternate-format views.
    pub view_formats: Vec<TextureFormat>,

    /// View intent that must be known when Texture is created.
    pub view_compatibility: TextureViewCompatibility,

    /// A performance preference only; it is never a correctness guarantee.
    pub memory: ResourceMemoryPreference,
}

impl TextureDescriptor {
    /// A 1D texture of one mip, one layer, one sample.
    pub fn new_1d(width: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self::from_parts(TextureDimension::D1, Extent3d::d1(width), format, usage)
    }

    /// A 2D texture of one mip, one layer, one sample.
    pub fn new_2d(width: u32, height: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self::from_parts(
            TextureDimension::D2,
            Extent3d::d2(width, height),
            format,
            usage,
        )
    }

    /// A 3D texture of one mip, one layer, one sample.
    pub fn new_3d(
        width: u32,
        height: u32,
        depth: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self::from_parts(
            TextureDimension::D3,
            Extent3d::d3(width, height, depth),
            format,
            usage,
        )
    }

    /// The shared body of the three constructors.
    ///
    /// Section 13.3 shows the three public constructors but not their defaults.
    /// These are the smallest values the invariants permit — one mip, one layer,
    /// one sample, no alternate view format, no view intent — so that a
    /// descriptor built by a constructor validates as soon as its extent does,
    /// and every departure from the minimum is something the caller asked for
    /// explicitly.
    fn from_parts(
        dimension: TextureDimension,
        extent: Extent3d,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            label: Label::default(),
            dimension,
            extent,
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format,
            usage,
            view_formats: Vec::new(),
            view_compatibility: TextureViewCompatibility::NONE,
            memory: ResourceMemoryPreference::Automatic,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Sets the mip level count, including level 0.
    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    /// Sets the array layer count.
    pub fn with_array_layers(mut self, layers: u32) -> Self {
        self.array_layers = layers;
        self
    }

    /// Sets the sample count.
    ///
    /// More than one sample is only legal for a 2D texture with one mip level
    /// (section 13.1); this builder records the request and
    /// `validate_texture_descriptor` is what refuses the combination.
    pub fn with_sample_count(mut self, samples: u32) -> Self {
        self.sample_count = samples;
        self
    }

    /// Declares one more alternate-format view as permitted.
    ///
    /// The list is kept canonical at every step — sorted, without duplicates —
    /// because section 13.1 makes it a *set*. Keeping the invariant here rather
    /// than only at creation means [`TextureDescriptor::view_formats`] never
    /// shows a caller a list that creation would later rewrite, and it is what
    /// lets the same descriptor feed
    /// [`crate::api::format::TextureSupportQuery`] as a stable cache key
    /// (section 13.3).
    pub fn with_view_format(mut self, format: TextureFormat) -> Self {
        self.view_formats.push(format);
        canonicalize_view_formats(&mut self.view_formats);
        self
    }

    /// Declares a view intent the creation step must grant.
    pub fn with_view_compatibility(mut self, compatibility: TextureViewCompatibility) -> Self {
        self.view_compatibility = compatibility;
        self
    }

    /// States a placement preference.
    pub fn with_memory_preference(mut self, preference: ResourceMemoryPreference) -> Self {
        self.memory = preference;
        self
    }
}

/// A created texture.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. Cloning is not a second texture; section
/// 18.6 keeps the native backing alive until the last logical owner is gone
/// *and* all accepted work referencing it is terminal.
#[derive(Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
}

/// The single shared ownership domain of one logical texture.
struct TextureInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: TextureDescriptor,
    native: Box<dyn TextureBackend>,
    /// Plan-scoped transient execution metadata, when this is not persistent.
    transient: Option<TransientResourceMetadata>,
}

/// Native-seam token used solely by crate-local validation fixtures.
///
/// It is still a concrete backend object, rather than an `Option`, so every
/// `Texture` has the same ownership invariant. Real device creation uses
/// `new_backed` and never constructs this token.
struct ValidationTextureBackend;

impl TextureBackend for ValidationTextureBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Texture {
    /// Assembles a crate-local validation fixture. Production creation must use
    /// [`Self::new_backed`], which requires the backend allocation to succeed.
    pub(crate) fn new(id: ObjectId, device: DeviceIdentity, descriptor: TextureDescriptor) -> Self {
        Self::new_backed(id, device, descriptor, Box::new(ValidationTextureBackend))
    }

    /// Assembles a texture whose backend allocation has already succeeded.
    pub(crate) fn new_backed(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: TextureDescriptor,
        native: Box<dyn TextureBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(TextureInner {
                id,
                device,
                descriptor,
                native,
                transient: None,
            }),
        }
    }

    /// Backend allocation, available on objects created through a real device.
    pub(crate) fn native(&self) -> &dyn TextureBackend {
        self.inner.native.as_ref()
    }

    /// Assembles a logical texture owned by one transient submission plan.
    pub(crate) fn new_transient(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: TextureDescriptor,
        native: Box<dyn TextureBackend>,
        transient: TransientResourceMetadata,
    ) -> Self {
        Self::new_backed(id, device, descriptor, native).with_transient(transient)
    }

    fn with_transient(mut self, transient: TransientResourceMetadata) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("newly-created texture must have one owner")
            .transient = Some(transient);
        self
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

    /// This texture's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this texture.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The descriptor this texture was created from.
    ///
    /// Section 18.8 requires a descriptor to be recoverable for capture, and
    /// §15.2's `whole` constructor reads it to build a view covering everything.
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.inner.descriptor
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (adjudication A16): descriptors in this
/// chapter are `#[derive(Clone, Debug)]` and contain a [`Texture`], so a handle
/// must be printable, but section 7.1 describes an object by its identity rather
/// than its contents. The backend port will add a native field that has no
/// reason to be `Debug`, and printing a native handle into a log would leak it.
/// `finish_non_exhaustive()` is what makes it honest that the descriptor is not
/// shown — a caller who needs it calls [`Texture::descriptor`].
impl fmt::Debug for Texture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Texture")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .finish_non_exhaustive()
    }
}

// The creation verb of this chapter, written here for the reason adjudication A28
// records: section 13.4 declares `create_texture` beside the type it produces, so
// the definition site is the owner. An inherent impl block attaches to `Device`
// wherever it is written in the defining crate, so this is still
// `crate::api::platform::Device::create_texture` to every caller and to the
// intra-doc links that name that path.
impl Device {
    /// Creates a texture.
    ///
    /// Section 13.4's creation verb. It is an inherent method written in the
    /// resource chapter rather than in `api::platform` because section 13.4
    /// declares it beside the object it produces: the definition site is the owner
    /// (adjudication A28).
    ///
    /// The descriptor is copied before validation, and the copy is the point. The
    /// public signature borrows the descriptor (`&TextureDescriptor`, as section
    /// 13.4 declares it) while section 13.4's validator takes `&mut`, because it
    /// canonicalizes the alternate-view-format set *only on acceptance* so that a
    /// rejected call leaves the caller's descriptor untouched. The accepted copy is
    /// what the backend would have stored; since this verb stops before an image is
    /// allocated, it is dropped rather than kept.
    ///
    /// The capability key is built here exactly as the validator builds it, and
    /// deliberately so: section 13.3 requires the query, the creation validation,
    /// and the backend's image creation to consult *one* description rather than
    /// three, and two constructions that disagreed would let the device answer a
    /// question validation does not check against.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::InvalidUsage`] for every descriptor-local violation — the
    /// shape invariants of section 13.1, an empty usage set, a non-canonical view
    /// format list, a CUBE intent the shape cannot satisfy, or a value past the
    /// device's ceiling — and [`RhiErrorKind::Unsupported`] when the device cannot
    /// create a texture with this key at all.
    pub fn create_texture(&self, desc: &TextureDescriptor) -> RhiResult<Texture> {
        // Section 6.5: a lost device refuses creation itself, and this verdict is
        // reachable before the capability read below even with no backend port.
        self.require_active()?;

        let mut accepted = desc.clone();
        let mut query = TextureSupportQuery::new(
            accepted.dimension,
            accepted.format,
            accepted.usage,
            accepted.sample_count,
        )
        .with_view_compatibility(accepted.view_compatibility);
        for format in &accepted.view_formats {
            query = query.with_view_format(*format);
        }
        let support = self.capabilities().texture_support(&query);
        validate_texture_descriptor(&mut accepted, &support)?;
        let native = self.native().create_texture(&accepted)?;
        Ok(Texture::new_backed(
            ObjectId::next(),
            self.identity(),
            accepted,
            native,
        ))
    }
}

/// Checks that a texture belongs to the device an operation targets.
///
/// Section 3.1 requires this comparison first and in O(1), before any backend is
/// touched, and section 3.3 fixes the answer as [`RhiErrorKind::WrongDevice`]:
/// there is no implicit copy, binding, handle unwrap, staging bridge, or peer
/// transfer in P0, and section 18.7 states the same for a resource that outlived
/// a lost device — its native handle cannot be made valid again by reusing it
/// against the new device.
pub(crate) fn validate_texture_ownership(
    texture: &Texture,
    target: DeviceIdentity,
) -> RhiResult<()> {
    if texture.device_identity() != target {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "texture belongs to a different device",
        )
        .with_object(texture.id()));
    }
    Ok(())
}

/// The texel extent of one mip level.
///
/// Section 15.3 names the meaning of a view's extent but not the arithmetic; the
/// standard mip reduction is that level `n` halves each axis, rounded down and
/// floored at one texel. A 3D texture's Z axis reduces the same way, because a
/// level is a level in every axis. Every other dimension reports `depth = 1`:
/// array layers do not contribute to depth, so a six-layer cube view is
/// `depth = 1`.
///
/// One definition, used by both [`crate::api::resource::view::TextureView::extent`]
/// and copy-region validation — the two places that must agree about the size of
/// a mip level.
pub(crate) fn mip_extent(extent: Extent3d, dimension: TextureDimension, level: u32) -> Extent3d {
    Extent3d {
        width: (extent.width >> level).max(1),
        height: (extent.height >> level).max(1),
        depth: if dimension == TextureDimension::D3 {
            (extent.depth >> level).max(1)
        } else {
            1
        },
    }
}

/// The canonical form of an alternate-view-format set: sorted, no duplicates.
///
/// Section 13.1 asks for exactly this, and section 13.3 needs it twice — once
/// for the descriptor and once for the capability query built from it — because
/// the query is a hashable key and a set that depended on insertion order would
/// make two spellings of one question compare unequal.
///
/// `TextureFormat` is deliberately not `Ord` (section 8.1 derives only `Hash` and
/// `Eq`), so the sort order is the declaration order of the variants. Any total
/// order yields the same canonical *set*, which is the part identity depends on.
pub(crate) fn canonicalize_view_formats(formats: &mut Vec<TextureFormat>) {
    formats.sort_unstable_by_key(|format| *format as u32);
    formats.dedup();
}

/// The largest legal mip level count for an extent.
///
/// Section 13.1: `mip_levels <= floor(log2(max(width, height, depth))) + 1`. The
/// bit length of a positive integer is exactly `floor(log2(n)) + 1`, so this is
/// the rule rather than an approximation of it — no floating point, and no
/// rounding error at the powers of two where an off-by-one would live.
pub(crate) fn mip_ceiling(extent: Extent3d) -> u32 {
    u32::BITS - extent.max_component().leading_zeros()
}

/// Checks a descriptor against the portable invariants and the device's answer.
///
/// This is section 13.4's creation list, in three passes:
///
/// ```text
/// 1. the shape invariants of 13.1, which hold on every device
/// 2. the descriptor-local rules of 13.1 and 13.2
///      usage non-empty, view_formats canonical and excluding the base format,
///      the CUBE intent's shape requirements
/// 3. the device answer of 8.4
///      TextureSupportQuery == Supported, then the descriptor's extent, mip
///      level count, and array layer count against TextureSupportLimits
/// ```
///
/// `DeviceIdentity` is the one entry of section 13.4's list this function cannot
/// check: it compares the texture's device against the target device, so it
/// needs both ([`crate::api::resource::buffer::validate_buffer_ownership`] is the
/// same check for buffers).
///
/// On `Ok`, `desc.view_formats` holds the canonical set; on `Err`, it is
/// untouched. Canonicalizing only into the accepted descriptor keeps a rejected
/// call from having a side effect a caller did not ask for.
///
/// # Errors
///
/// [`RhiErrorKind::InvalidUsage`] for every descriptor-local violation, and
/// [`RhiErrorKind::Unsupported`] when the device cannot create a texture with
/// this key.
pub(crate) fn validate_texture_descriptor(
    desc: &mut TextureDescriptor,
    support: &TextureSupport,
) -> RhiResult<()> {
    let extent = desc.extent;

    // Pass 1: the shape invariants of section 13.1.
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "texture extent {}x{}x{} must have every component greater than zero",
                extent.width, extent.height, extent.depth
            ),
        ));
    }
    if desc.mip_levels == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture mip level count must be greater than zero",
        ));
    }
    if desc.array_layers == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture array layer count must be greater than zero",
        ));
    }
    if desc.sample_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture sample count must be greater than zero",
        ));
    }

    match desc.dimension {
        TextureDimension::D1 => {
            if extent.height != 1 || extent.depth != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 1D texture must have height = depth = 1, not {}x{}",
                        extent.height, extent.depth
                    ),
                ));
            }
            if desc.array_layers != 1 || desc.sample_count != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 1D texture must have one array layer and one sample, not {} \
                         layers and {} samples",
                        desc.array_layers, desc.sample_count
                    ),
                ));
            }
        }
        TextureDimension::D2 => {
            if extent.depth != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("a 2D texture must have depth = 1, not {}", extent.depth),
                ));
            }
        }
        TextureDimension::D3 => {
            if desc.array_layers != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 3D texture has Z slices rather than array layers, so \
                         array_layers must be 1, not {}",
                        desc.array_layers
                    ),
                ));
            }
            if desc.sample_count != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a 3D texture must have one sample, not {}",
                        desc.sample_count
                    ),
                ));
            }
        }
    }

    if desc.sample_count > 1 {
        if desc.dimension != TextureDimension::D2 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "multisampling is only defined for 2D textures in P0",
            ));
        }
        if desc.mip_levels != 1 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a multisampled texture must have exactly one mip level",
            ));
        }
    }

    let ceiling = mip_ceiling(extent);
    if desc.mip_levels > ceiling {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a {}x{}x{} texture permits at most {ceiling} mip levels, not {}",
                extent.width, extent.height, extent.depth, desc.mip_levels
            ),
        ));
    }

    // Pass 2: descriptor-local rules.
    if desc.usage.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture usage must not be empty",
        ));
    }

    if desc.view_formats.contains(&desc.format) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "view_formats may not include the base format; the base format is \
             always viewable as itself",
        ));
    }

    if desc
        .view_compatibility
        .contains(TextureViewCompatibility::CUBE)
    {
        if desc.dimension != TextureDimension::D2 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "CUBE view compatibility requires a 2D texture",
            ));
        }
        if extent.width != extent.height {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "CUBE view compatibility requires a square texture, not {}x{}",
                    extent.width, extent.height
                ),
            ));
        }
        if desc.array_layers < 6 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "CUBE view compatibility requires at least 6 array layers, not {}",
                    desc.array_layers
                ),
            ));
        }
        if desc.sample_count != 1 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "CUBE view compatibility requires a single-sampled texture",
            ));
        }
    }

    // Pass 3: the device answer, keyed exactly as section 13.3 builds it.
    let mut query =
        TextureSupportQuery::new(desc.dimension, desc.format, desc.usage, desc.sample_count)
            .with_view_compatibility(desc.view_compatibility);
    for format in &desc.view_formats {
        query = query.with_view_format(*format);
    }

    let limits = match support {
        TextureSupport::Unsupported => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!(
                    "this device cannot create a {:?} texture with format {:?}, usage {}, \
                     and sample count {}",
                    desc.dimension, desc.format, desc.usage, desc.sample_count
                ),
            ));
        }
        TextureSupport::Supported(limits) => limits,
    };

    let max = limits.max_extent();
    if extent.width > max.width || extent.height > max.height || extent.depth > max.depth {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "extent {}x{}x{} exceeds the supported maximum {}x{}x{}",
                extent.width, extent.height, extent.depth, max.width, max.height, max.depth
            ),
        ));
    }
    if desc.mip_levels > limits.max_mip_levels() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "{} mip levels exceed the supported maximum {}",
                desc.mip_levels,
                limits.max_mip_levels()
            ),
        ));
    }
    if desc.array_layers > limits.max_array_layers() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "{} array layers exceed the supported maximum {}",
                desc.array_layers,
                limits.max_array_layers()
            ),
        ));
    }

    // The descriptor is accepted, so it adopts the canonical view-format set.
    canonicalize_view_formats(&mut desc.view_formats);
    Ok(())
}

// ---------------------------------------------------------------------------
// Canonical capability encoding
// ---------------------------------------------------------------------------
//
// The rules of the encoding, and what it is for, are stated once in
// `api::capability::CapabilityFacts`. It lives here because every field read
// below is private to this module.

impl TextureDimension {
    /// Writes this dimension's canonical byte.
    ///
    /// A fieldless enum encodes as its discriminant; see
    /// [`crate::api::shader::ShaderStage::encode_into`] for why that dependency on
    /// declaration order is the intended one.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
}

impl TextureUsage {
    /// Writes this usage mask's bits, little-endian.
    ///
    /// The bits rather than the mask's `Debug` rendering: `Debug` is not a
    /// stability contract, and a fingerprint that tooling compares across
    /// processes must not move because a derive's output moved.
    ///
    /// The mask rather than a list of members: a mask has exactly one bit pattern
    /// per combination, so it is already canonical and no ordering question
    /// arises.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}

impl TextureViewCompatibility {
    /// Writes this view-intent mask's bits, little-endian.
    ///
    /// Same reasoning as [`TextureUsage::encode_into`]; the two are separate
    /// methods rather than a shared helper over "the crate's bitmask newtypes"
    /// because a shared helper would need them to be one type, and they are not.
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}
