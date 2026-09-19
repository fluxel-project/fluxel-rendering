//! Texture views (specification sections 15.1 through 15.3).
//!
//! A view is the only way to hand a texture to a shader, an attachment, or a
//! storage binding. This module owns the view dimension, the descriptor that
//! selects a subresource range and an optional alternate format, and the opaque
//! object that results.
//!
//! # Two different questions, two different places
//!
//! Section 8.5 splits view legality, and this module is one half of the split:
//!
//! ```text
//! can this format legally reinterpret that one?   EnabledCapabilities
//!                                                 ::texture_view_format_compatible
//! may this descriptor ask for it?                 validate_texture_view_descriptor
//! ```
//!
//! The first is a device fact (some pairs of formats are reinterpretable on one
//! driver and not another), and it is owned by module 01. The second is
//! portable: whether the base texture *declared* the view format when it was
//! created, whether the aspect is one the base format has at all, whether the
//! mip/layer range exists, and whether the view dimension is compatible with the
//! texture's. Section 8.5 assigns both to `create_texture_view` validation and
//! says neither may be skipped, so this module checks the portable half and
//! names the device half rather than guessing at it.
//!
//! # What this module does not own
//!
//! - Whether a *binding* of the resulting view is legal is the binding chapter
//!   (section 20): P0 has no sampled stencil semantics, and that is a fact about
//!   which shader interface the view is bound to, not about the view.
//! - The subresource vocabulary itself is
//!   [`crate::api::resource::subresource`].

use core::fmt;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, format_aspects};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::resource::subresource::TextureAspects;
use crate::api::resource::texture::{
    Extent3d, Texture, TextureDescriptor, TextureDimension, TextureViewCompatibility, mip_extent,
    validate_texture_ownership,
};

/// The dimensionality a view presents.
///
/// P0 has no `D1Array` because baseline WebGPU has no common 1D-array view
/// semantics, and no `D3 -> D2`/`D2Array` sliced view; section 15.1 defers both
/// to a separate future extension rather than exposing a shape that some
/// backends cannot express.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    /// A 1D view of a 1D texture.
    D1,
    /// A 2D view of a 2D texture, one layer.
    D2,
    /// A 2D view across a layer range of a 2D texture.
    D2Array,
    /// A cube view: exactly 6 layers of a cube-compatible 2D texture.
    Cube,
    /// A cube-array view: a multiple of 6 layers of a cube-compatible 2D
    /// texture.
    CubeArray,
    /// A 3D view of a 3D texture.
    D3,
}

/// Everything a caller states about a texture view before it exists.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct TextureViewDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,

    /// The dimensionality the view presents.
    pub dimension: TextureViewDimension,

    /// None = use the base Texture format.
    pub format: Option<TextureFormat>,

    /// Which aspects the view covers. Must be a subset of the base format's.
    pub aspects: TextureAspects,

    /// First mip level the view covers.
    pub base_mip: u32,
    /// Number of mip levels the view covers. Must be at least 1.
    pub mip_count: u32,

    /// First array layer the view covers.
    pub base_layer: u32,
    /// Number of array layers the view covers. Must be at least 1.
    pub layer_count: u32,
}

impl TextureViewDescriptor {
    /// Describes a view. Checks nothing: `validate_texture_view_descriptor` is
    /// the check, and it needs the base texture's descriptor to decide anything.
    pub fn new(
        dimension: TextureViewDimension,
        aspects: TextureAspects,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) -> Self {
        Self {
            label: Label::default(),
            dimension,
            format: None,
            aspects,
            base_mip,
            mip_count,
            base_layer,
            layer_count,
        }
    }

    /// Constructs a view covering the complete logical subresource range from a Texture descriptor.
    ///
    /// Cube/CubeArray compatibility validation is still performed.
    ///
    /// Fallible, unlike [`Self::new`], because "the whole texture" is not always
    /// a legal view of the requested dimension: asking for a `Cube` view of a
    /// texture whose creation did not declare the CUBE intent is a refusal, and
    /// it must be reported here rather than by a driver later. The resolution of
    /// "complete" comes from the texture's own descriptor, which is what makes
    /// the result correct for every dimensionality instead of the 2D case only —
    /// the old `whole_2d` it replaces could not describe 1D, 3D, array, or cube
    /// views at all.
    pub fn whole(texture: &Texture, dimension: TextureViewDimension) -> RhiResult<Self> {
        let base = texture.descriptor();
        let descriptor = Self::new(
            dimension,
            format_aspects(base.format),
            0,
            base.mip_levels,
            0,
            base.array_layers,
        );
        // Validation does not rewrite a view descriptor — it has no set-valued
        // field to canonicalize — so the checked value is the returned one.
        validate_texture_view_descriptor(&descriptor, base)?;
        Ok(descriptor)
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Declares an alternate view format.
    ///
    /// Recording the request is all this does. Whether the base texture declared
    /// that format, and whether the device permits the reinterpretation at all,
    /// are decided by validation — section 8.5 explicitly refuses the shortcut
    /// "same byte size, therefore view-compatible".
    pub fn with_format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }
}

/// A created texture view.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the
/// [`DeviceIdentity`] that created it. It keeps its base [`Texture`] alive: a
/// view without its texture is a dangling native handle, and section 18.6 makes
/// that a logical-ownership rule rather than a convention.
#[derive(Clone)]
pub struct TextureView {
    id: ObjectId,
    device: DeviceIdentity,
    texture: Texture,
    descriptor: TextureViewDescriptor,
}

impl TextureView {
    /// Assembles a created texture view.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only [`crate::api::platform::Device::create_texture_view`] may produce one.
    /// That verb exists and is the only caller this is written for, but it stops
    /// before a native view is bound — nothing can mint the identity below until a
    /// backend view path does — so nothing calls this yet.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_texture_view calls this once the backend view path lands and can mint the view's identity"
        )
    )]
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        texture: Texture,
        descriptor: TextureViewDescriptor,
    ) -> Self {
        Self {
            id,
            device,
            texture,
            descriptor,
        }
    }

    /// This view's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this view.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The texture this view was created from.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// The descriptor this view was created from.
    pub fn descriptor(&self) -> &TextureViewDescriptor {
        &self.descriptor
    }

    /// Actual view format.
    ///
    /// Resolved rather than echoed: a descriptor that left
    /// [`TextureViewDescriptor::format`] as `None` means "the base format", and
    /// this is the accessor that answers what that came to.
    pub fn format(&self) -> TextureFormat {
        self.descriptor
            .format
            .unwrap_or_else(|| self.texture.descriptor().format)
    }

    /// The aspects this view covers.
    ///
    /// The descriptor's own set, which validation has already checked is a
    /// subset of the base format's aspects. Unlike [`Self::format`] this needs no
    /// resolution: the descriptor carries a set rather than an option, and the
    /// view's effective aspect is exactly the one it selected.
    pub fn aspects(&self) -> TextureAspects {
        self.descriptor.aspects
    }

    /// Logical texel extent of base_mip; array layers do not contribute to depth.
    ///
    /// Section 15.3 names the meaning but not the arithmetic. It is the standard
    /// mip reduction: each level beyond 0 halves the extent, rounded down and
    /// floored at one texel, so level `n` of a `w x h` texture is
    /// `max(1, w >> n) x max(1, h >> n)`. A 3D texture's Z axis reduces the same
    /// way, because a level is a level in every axis; every other dimension
    /// reports `depth = 1`, which is what "array layers do not contribute to
    /// depth" means — a 6-layer cube view is `depth = 1`, not `depth = 6`.
    pub fn extent(&self) -> Extent3d {
        let base = self.texture.descriptor();
        mip_extent(base.extent, base.dimension, self.descriptor.base_mip)
    }

    /// The sample count of the base texture.
    ///
    /// A view cannot change it: section 15.3 lists "sample count restrictions"
    /// among the things creation validation checks, and in P0 the only such
    /// restriction is that cube views require a single-sampled texture.
    pub fn sample_count(&self) -> u32 {
        self.texture.descriptor().sample_count
    }

    /// The number of array layers this view covers.
    pub fn layer_count(&self) -> u32 {
        self.descriptor.layer_count
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (adjudication A16): a view is
/// `#[derive(Clone)]` and is named by descriptors in this chapter, but section
/// 7.1 describes an object by its identity rather than its contents. The backend
/// port will add a native field that has no reason to be `Debug`, and printing a
/// native handle into a log would leak it. `finish_non_exhaustive()` is what
/// makes it honest that the descriptor is not shown — a caller who needs it calls
/// [`TextureView::descriptor`].
impl fmt::Debug for TextureView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TextureView")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// The creation verb of this chapter, written here for the reason adjudication A28
// records: section 15.3 declares `create_texture_view` beside the type it produces,
// so the definition site is the owner. The inherent impl block attaches to `Device`
// wherever it is written, so callers and intra-doc links that name
// `crate::api::platform::Device::create_texture_view` still resolve here.
impl Device {
    /// Creates a texture view.
    ///
    /// Section 15.3's creation verb. It is an inherent method written in the view
    /// chapter rather than in `api::platform` because section 15.3 declares it
    /// beside the object it produces: the definition site is the owner
    /// (adjudication A28).
    ///
    /// This is the verb whose portable half is decided entirely from objects the
    /// caller already holds, so both of its checks run before anything else and
    /// neither needs a capability fact:
    ///
    /// 1. section 3.1's identity comparison, first and in O(1), so a foreign
    ///    texture is refused as [`RhiErrorKind::WrongDevice`] rather than as
    ///    whatever the range check would have said about it — there is no implicit
    ///    copy or staging bridge that could make a foreign texture work;
    /// 2. section 15.3's descriptor rules, against the base texture's own
    ///    descriptor: the mip and layer ranges, the aspect the base format carries,
    ///    whether creation declared the alternate view format, the dimension
    ///    compatibility, and the cube intent and layer counts.
    ///
    /// What it deliberately does not do is decide section 8.5's *device* half. That
    /// half — whether this driver permits the base format to be reinterpreted as the
    /// view format — is a probed device fact
    /// ([`crate::api::capability::EnabledCapabilities::texture_view_format_compatible`]),
    /// and section 8.5 requires both halves to be checked at creation while saying
    /// neither may be skipped. It is named in the stop below rather than guessed at,
    /// because guessing the driver's answer is exactly the "same byte size, therefore
    /// view-compatible" shortcut that section 8.5 refuses.
    ///
    /// # Errors
    ///
    /// [`RhiErrorKind::WrongDevice`] when the texture belongs to another device, and
    /// [`RhiErrorKind::InvalidUsage`] for every descriptor rule above — a view that
    /// names a range the texture does not have, an aspect the base format does not
    /// carry, or a reinterpretation the texture was not created to permit is a
    /// mismatch between two objects the caller holds, not a device limitation.
    pub fn create_texture_view(
        &self,
        texture: &Texture,
        desc: &TextureViewDescriptor,
    ) -> RhiResult<TextureView> {
        validate_texture_ownership(texture, self.identity())?;
        validate_texture_view_descriptor(desc, texture.descriptor())?;

        // Section 6.5's liveness verdict, after the ownership comparison and the
        // descriptor's own portable checks. A texture belonging to another
        // device is `WrongDevice` even when this device is also lost.
        self.require_active()?;
        unimplemented!(
            "Device::create_texture_view needs a backend to bind a native view over a \
             texture on device {:?}; the portable contract is fixed and both checks \
             listed above are built, but no backend port is built, and section 8.5's \
             device half — whether this driver permits the base format to be \
             reinterpreted as the view format — needs the enabled-capability snapshot, \
             which is a backend-port deliverable",
            self.identity()
        )
    }
}

/// Checks a view descriptor against the texture it will be created from.
///
/// Section 15.3's list, with the device-owned entry left to its owner:
///
/// ```text
/// DeviceIdentity              the caller compares the two, not a descriptor rule
/// format reinterpretation     EnabledCapabilities (module 01, section 8.5)
/// TextureDescriptor.view_formats      checked here
/// aspect                      checked here
/// mip/layer range             checked here
/// dimension compatibility     checked here
/// TextureDescriptor.view_compatibility  checked here
/// sample count restrictions   checked here
/// ```
///
/// Every refusal is [`RhiErrorKind::InvalidUsage`]: a view that names a range the
/// texture does not have, an aspect the format does not carry, or a
/// reinterpretation the texture did not declare is a mismatch between two
/// objects the caller already holds, and saying `Unsupported` would blame the
/// device for it.
pub(crate) fn validate_texture_view_descriptor(
    view: &TextureViewDescriptor,
    base: &TextureDescriptor,
) -> RhiResult<()> {
    if view.mip_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture view must cover at least one mip level",
        ));
    }
    if view.layer_count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture view must cover at least one array layer",
        ));
    }
    if view.aspects.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a texture view must select at least one aspect",
        ));
    }

    // Mip range. `checked_add` first: `base_mip + mip_count` overflowing u64 is
    // impossible in practice (the fields are u32), but writing the comparison as
    // a sum would let a future widening of these fields silently wrap into a
    // range that looks valid.
    let last_mip = view.base_mip.checked_add(view.mip_count).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "texture view mip range overflows",
        )
    })?;
    if last_mip > base.mip_levels {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "texture view covers mip levels {}..{}, but the texture has {}",
                view.base_mip, last_mip, base.mip_levels
            ),
        ));
    }

    // Layer range.
    let last_layer = view
        .base_layer
        .checked_add(view.layer_count)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::InvalidUsage,
                "texture view layer range overflows",
            )
        })?;
    if last_layer > base.array_layers {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "texture view covers array layers {}..{}, but the texture has {}",
                view.base_layer, last_layer, base.array_layers
            ),
        ));
    }

    // Aspect. The base format decides which aspects exist at all; a color view
    // of a depth format is not a range error but a category error, and this is
    // where it is caught.
    let format_aspects_of_base = format_aspects(base.format);
    if !format_aspects_of_base.contains(view.aspects) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "texture view selects aspects the {:?} base format does not carry",
                base.format
            ),
        ));
    }

    // Alternate view format. Section 8.5 splits this into "may the descriptor
    // ask" (here) and "does the device permit the reinterpretation" (module 01).
    // Only the first half is portable.
    if let Some(view_format) = view.format {
        if view_format != base.format && !base.view_formats.contains(&view_format) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the texture was not created with {:?} among its view_formats, so it \
                     may not be viewed as one",
                    view_format
                ),
            ));
        }
        if !format_aspects(view_format).contains(view.aspects) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the alternate view format {:?} does not carry the selected aspects",
                    view_format
                ),
            ));
        }
    }

    // Dimension compatibility, and the cube rules from section 13.2 (an intent
    // the creation step had to grant) plus section 15.3 (a layer count that is
    // exactly a cube face count).
    match view.dimension {
        TextureViewDimension::D1 => {
            if base.dimension != TextureDimension::D1 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a D1 view requires a D1 texture",
                ));
            }
        }
        TextureViewDimension::D3 => {
            if base.dimension != TextureDimension::D3 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a D3 view requires a D3 texture",
                ));
            }
        }
        TextureViewDimension::D2
        | TextureViewDimension::D2Array
        | TextureViewDimension::Cube
        | TextureViewDimension::CubeArray => {
            if base.dimension != TextureDimension::D2 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "a 2D, array, or cube view requires a D2 texture",
                ));
            }
        }
    }

    if matches!(
        view.dimension,
        TextureViewDimension::Cube | TextureViewDimension::CubeArray
    ) {
        if !base
            .view_compatibility
            .contains(TextureViewCompatibility::CUBE)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the texture was not created with CUBE view compatibility, and the intent \
                 cannot be added after creation",
            ));
        }
        if base.sample_count != 1 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a cube view requires a single-sampled texture",
            ));
        }
        if view.dimension == TextureViewDimension::Cube {
            if view.layer_count != 6 {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a cube view covers exactly 6 array layers, not {}",
                        view.layer_count
                    ),
                ));
            }
        } else if !view.layer_count.is_multiple_of(6) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "a cube array view covers a multiple of 6 array layers, not {}",
                    view.layer_count
                ),
            ));
        }
    }

    Ok(())
}
