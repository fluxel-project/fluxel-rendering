//! Raster attachment sets and their invariants (specification sections 31.1
//! through 31.4).
//!
//! This module owns the attachment *set* of one raster scope: which views are
//! attached at which color location, whether they are loaded or cleared and
//! stored or discarded, and the resolve pairing of a multisampled color target.
//!
//! # What this module does not own
//!
//! - Command ordering or the state machine. Attachments are validated here and
//!   recorded by [`crate::api::command::raster`]; nothing here knows whether a
//!   scope is currently open.
//! - Whether a *format* may be a color or depth attachment at all. That is
//!   [`crate::api::format::FormatFacts::color_attachment`], which is a probed
//!   device fact and cannot be read without a device. What is checked here is the
//!   portable half: the texture's own `COLOR_ATTACHMENT` /
//!   `DEPTH_STENCIL_ATTACHMENT` usage bit, the clear value's numeric class, and
//!   the resolve pairing's own consistency.
//! - Whether a *frame* may be the target of a direct multisampled resolve.
//!   Section 46.1 permits that route "only when the active presentation facts and
//!   the resolved route facts prove that target, format, sample count, and resolve
//!   route", and those are device facts. This module answers the portable half —
//!   [`ColorAttachmentView::allows_resolve_into`] — and refuses a frame, so the
//!   permission belongs to the device façade; without that split the one route the
//!   gate exists for would be allowed unconditionally.
//! - The pipeline's view of the same set. A pipeline carries a
//!   [`RenderTargetSignature`] and the scope carries the attachments; comparing
//!   them is [`crate::api::command::raster`]'s `set_pipeline`.
//!
//! # The invariant this module enforces
//!
//! **All active attachments of one scope are one attachment set**: one device,
//! one extent, one sample count, one layer. Section 31.4's rule is what makes a
//! raster scope lowerable as a single native render pass, and it is checked once
//! at `begin_raster` rather than per draw, because the set cannot change inside a
//! scope.

use crate::api::command::geometry::{ClearValueClass, ColorClearValue, LoadOp, StoreOp};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, color_output_type};
use crate::api::identity::{DeviceIdentity, Label};
use crate::api::pipeline::RenderTargetSignature;
use crate::api::presentation::FrameAttachment;
use crate::api::resource::texture::{Extent3d, TextureUsage};
use crate::api::resource::view::TextureView;
use crate::api::shader::{ShaderLocation, ShaderNumericType};

/// The view a color attachment renders into.
///
/// Two variants rather than one texture-wrapping type, because a frame is
/// neither a [`crate::api::resource::texture::Texture`] nor a [`TextureView`]
/// (root specification, section 3): it is an acquired presentation image whose
/// layer, mip, and usage the caller does not choose. Folding a frame into a
/// `TextureView` would have to invent a view descriptor for an image the caller
/// does not own.
#[non_exhaustive]
#[derive(Clone)]
pub enum ColorAttachmentView {
    /// An ordinary texture view.
    Texture(TextureView),
    /// A frame image acquired for this submission.
    Frame(FrameAttachment),
}

impl ColorAttachmentView {
    /// The device the view belongs to.
    ///
    /// O(1) and portable, which is what section 3.1 asks for: a cross-device
    /// attachment is refused before a backend is consulted.
    pub fn device_identity(&self) -> DeviceIdentity {
        match self {
            Self::Texture(view) => view.device_identity(),
            Self::Frame(frame) => frame.device_identity(),
        }
    }

    /// The format the attachment holds.
    pub fn format(&self) -> TextureFormat {
        match self {
            Self::Texture(view) => view.format(),
            Self::Frame(frame) => frame.format(),
        }
    }

    /// The extent of the attachment.
    pub fn extent(&self) -> Extent3d {
        match self {
            Self::Texture(view) => view.extent(),
            Self::Frame(frame) => frame.extent(),
        }
    }

    /// The sample count of the attachment.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::Texture(view) => view.sample_count(),
            Self::Frame(frame) => frame.sample_count(),
        }
    }

    /// The identity of the underlying texture, or `None` for a frame.
    ///
    /// Used to decide whether a resolve source and target are the same
    /// allocation, which is the only reason section 31.1's overlap rule needs an
    /// identity rather than an extent.
    pub(crate) fn texture_identity(&self) -> Option<crate::api::identity::ObjectId> {
        match self {
            Self::Texture(view) => Some(view.texture().id()),
            Self::Frame(_) => None,
        }
    }

    /// Whether this view may be a color attachment, as far as is portable.
    ///
    /// A texture view must be of a texture created with
    /// [`TextureUsage::COLOR_ATTACHMENT`]; a frame is an acquired color render
    /// attachment by construction (section 44.3), so it answers `true`.
    ///
    /// This answers *only* "may this be rendered into". It is deliberately not the
    /// answer to "may a multisampled attachment resolve into this", which section
    /// 46.1 gates on facts a frame may or may not have: see
    /// [`Self::allows_resolve_into`].
    ///
    /// The remaining half of section 31.1's rule — that the *format* may be a
    /// color attachment — is
    /// [`crate::api::format::FormatFacts::color_attachment`], a probed device
    /// fact. It is left to the device façade, which is the only place that has
    /// one; a portable check cannot answer it and guessing from the format name
    /// would be a second copy of section 8's table.
    pub(crate) fn allows_color_attachment(&self) -> bool {
        match self {
            Self::Texture(view) => view
                .texture()
                .descriptor()
                .usage
                .contains(TextureUsage::COLOR_ATTACHMENT),
            Self::Frame(_) => true,
        }
    }

    /// Whether a multisampled attachment may resolve into this view.
    ///
    /// A second predicate rather than a reuse of
    /// [`Self::allows_color_attachment`], because the two questions are not the
    /// same one and the difference is a route the specification gates: section
    /// 46.1 makes the *unconditional* P0 route a single-sample raster write to a
    /// frame, and permits a direct multisampled `RasterScope` resolve into it
    /// "only when the active presentation facts and the resolved route facts prove
    /// that target, format, sample count, and resolve route". When they are not
    /// proved, the required portable route is an intermediate single-sample
    /// texture resolved into, followed by a final single-sample raster write.
    ///
    /// ```text
    /// Texture   allowed, when its texture may be a color attachment at all
    /// Frame     refused here
    /// ```
    ///
    /// The facts that would permit a frame are device facts — the active
    /// presentation route and what the acquired image's configuration proved about
    /// resolving into it — and a portable check has none of them. Answering `true`
    /// for a frame, as one bool answering *both* questions did, is a **fail open**
    /// on the one route this gate exists to hold; a frame therefore answers `false`
    /// and the permission belongs to the device façade, which is the only layer
    /// that holds the facts. That is the same division of labour
    /// [`Self::allows_color_attachment`]'s own documentation states for the format
    /// half of section 31.1.
    ///
    /// A texture view answers through [`Self::allows_color_attachment`], because
    /// section 31.1's resolve target is an ordinary single-sampled color
    /// attachment that happens to be written by a resolve rather than by the
    /// shader — the usage bit and the format fact are the whole requirement, and
    /// the raster resolve needs no `COPY_DST` on either side.
    pub(crate) fn allows_resolve_into(&self) -> bool {
        match self {
            Self::Texture(_) => self.allows_color_attachment(),
            Self::Frame(_) => false,
        }
    }
}

/// One color attachment: what it draws into, and what happens at each end of the
/// scope.
///
/// `load` and `store` are not optional, because section 32.4 gives both of them
/// semantics that produce actual-use records: `Clear` is a scope-begin write and
/// `Load` is a scope-begin read, and a `Discard` target produces no result for a
/// later pass to depend on. A default would silently pick one of those.
#[derive(Clone)]
pub struct ColorAttachment {
    /// What is rendered into.
    pub view: ColorAttachmentView,
    /// What the contents are set to when the scope begins.
    pub load: LoadOp<ColorClearValue>,
    /// What happens to the contents when the scope ends.
    pub store: StoreOp,
    /// Z slice selected when `view` is a 3D texture view. Must be `None` for
    /// all non-3D views and within the selected mip's depth extent for a 3D
    /// view. A frame never has a depth slice.
    pub depth_slice: Option<u32>,
    /// The single-sampled target this multisampled attachment resolves into.
    ///
    /// Section 31.1: a resolve target is part of the *attachment*, not a
    /// separate command, and it is deliberately not part of the pipeline's
    /// [`RenderTargetSignature`] — the pipeline sees the multisampled source.
    pub resolve: Option<ColorAttachmentView>,
}

/// How a depth attachment is treated by a scope.
///
/// Read-only and read-write are separate variants rather than a boolean plus a
/// load/store pair, which is section 31.2's correction: `read_only: true` with
/// `LoadOp::Clear` describes a state that cannot be lowered, and the old model
/// could express it. Here it cannot be constructed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum DepthAttachmentMode {
    /// The depth plane is read and never written.
    ///
    /// Produces no attachment **write** use; section 31.2 states that explicitly,
    /// because a depth test that only reads is a hazard input and not an output.
    ReadOnly,

    /// The depth plane is read and written.
    ReadWrite {
        /// What the plane is set to when the scope begins.
        load: LoadOp<f32>,
        /// What happens to it when the scope ends.
        store: StoreOp,
    },
}

/// How a stencil attachment is treated by a scope.
///
/// The stencil counterpart of [`DepthAttachmentMode`], with `u32` clear values
/// because a stencil clear is an integer, not a float.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum StencilAttachmentMode {
    /// The stencil plane is read and never written.
    ReadOnly,

    /// The stencil plane is read and written.
    ReadWrite {
        /// What the plane is set to when the scope begins.
        load: LoadOp<u32>,
        /// What happens to it when the scope ends.
        store: StoreOp,
    },
}

/// A depth and/or stencil attachment.
///
/// One view with two independent modes, because depth and stencil are separate
/// planes of one image: a scope may read depth while clearing stencil, and the
/// combination `depth = None, stencil = None` is the one state section 31.2
/// refuses outright.
#[derive(Clone)]
pub struct DepthStencilAttachment {
    /// The view that is attached. It must carry
    /// [`TextureUsage::DEPTH_STENCIL_ATTACHMENT`].
    pub view: TextureView,
    /// How the depth plane is treated, or `None` if this attachment has no depth
    /// plane.
    pub depth: Option<DepthAttachmentMode>,
    /// How the stencil plane is treated, or `None` if this attachment has no
    /// stencil plane.
    pub stencil: Option<StencilAttachmentMode>,
}

/// Everything a caller states about one raster scope before it opens.
///
/// `colors` is indexed by attachment location, with `None` for a hole, because
/// sparse MRT locations are a real shape in Vulkan, D3D12, and Metal: location 0
/// and location 3 with nothing between them is what a deferred renderer's two
/// outputs look like. Canonicalization removes *trailing* `None`s only; interior
/// holes are kept and mean what they say.
#[non_exhaustive]
#[derive(Clone)]
pub struct RasterScopeDescriptor {
    /// Diagnostic label. It also establishes the scope's diagnostics nesting,
    /// which is separate from the debug-group stack (section 36).
    pub label: Label,

    /// index = color attachment location.
    /// None = this location has no attachment.
    pub colors: Vec<Option<ColorAttachment>>,

    /// The depth and/or stencil attachment, if any.
    pub depth_stencil: Option<DepthStencilAttachment>,
}

impl RasterScopeDescriptor {
    /// States a scope with no attachments.
    ///
    /// An empty descriptor is constructible but not usable: section 31.4 refuses
    /// a scope with no attachment at all. That is deliberate — the refusal
    /// belongs at `begin_raster`, where the message can say which descriptor was
    /// empty, rather than at construction, where a builder is still being filled
    /// in.
    pub fn new() -> Self {
        Self {
            label: Label::default(),
            colors: Vec::new(),
            depth_stencil: None,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Attaches a color attachment at a location.
    ///
    /// Locations between the previous highest and this one are filled with
    /// `None`, so attaching location 3 first and location 0 second produces the
    /// same descriptor either way. The gap is a real hole and survives
    /// canonicalization.
    pub fn with_color(mut self, location: ShaderLocation, attachment: ColorAttachment) -> Self {
        let index = location.get() as usize;
        if self.colors.len() <= index {
            self.colors.resize(index + 1, None);
        }
        self.colors[index] = Some(attachment);
        self
    }

    /// Attaches the depth and/or stencil attachment.
    pub fn with_depth_stencil(mut self, attachment: DepthStencilAttachment) -> Self {
        self.depth_stencil = Some(attachment);
        self
    }

    /// Removes trailing unattached color locations.
    ///
    /// Section 31.3's canonicalization, and the reason it exists: `[Some, None,
    /// None]` and `[Some]` are one attachment set, and two representations of one
    /// set would make signature comparison and diagnostics disagree about
    /// identical scopes.
    pub(crate) fn canonicalized(mut self) -> Self {
        while matches!(self.colors.last(), Some(None)) {
            self.colors.pop();
        }
        self
    }

    /// The pipeline-facing signature of this attachment set.
    ///
    /// Section 32.2's comparison, computed the same way
    /// [`crate::api::pipeline::RasterPipelineDescriptor::target_signature`] is:
    /// sparse color formats, the depth/stencil format, and the sample count. The
    /// resolve target is left out, because section 32.2 says the pipeline sees
    /// the multisampled source rather than its resolve target.
    pub(crate) fn target_signature(&self) -> RenderTargetSignature {
        RenderTargetSignature {
            color_formats: self
                .colors
                .iter()
                .map(|attachment| attachment.as_ref().map(|color| color.view.format()))
                .collect(),
            depth_stencil_format: self.depth_stencil.as_ref().map(|depth| depth.view.format()),
            sample_count: self.primary_sample_count(),
        }
        .canonicalized()
    }

    /// The sample count every main attachment must share, or `1` for an empty
    /// set.
    ///
    /// Taken from the first color attachment, or from the depth/stencil
    /// attachment if there is no color attachment. Section 31.4 requires the set
    /// to agree, so any member answers the same number once the set is validated;
    /// this accessor exists for the case where validation is about to refuse the
    /// set, and it must still produce *some* signature rather than panic.
    pub(crate) fn primary_sample_count(&self) -> u32 {
        let from_color = self
            .colors
            .iter()
            .flatten()
            .next()
            .map(|color| color.view.sample_count());
        from_color.unwrap_or_else(|| {
            self.depth_stencil
                .as_ref()
                .map(|depth| depth.view.sample_count())
                .unwrap_or(1)
        })
    }

    /// Whether the set has no attachment at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.colors.iter().all(Option::is_none) && self.depth_stencil.is_none()
    }

    /// Every attached color location, in ascending order.
    pub(crate) fn attached_colors(&self) -> impl Iterator<Item = (u32, &ColorAttachment)> {
        self.colors
            .iter()
            .enumerate()
            .filter_map(|(location, attachment)| {
                attachment.as_ref().map(|color| (location as u32, color))
            })
    }
}

impl Default for RasterScopeDescriptor {
    /// The same empty descriptor as [`RasterScopeDescriptor::new`].
    ///
    /// Section 31.3 declares `new()`; `Default` is added because a descriptor
    /// with no attachment is a legal *value* even though it is not a legal scope,
    /// and clippy's `new_without_default` lint is right that the two should agree.
    fn default() -> Self {
        Self::new()
    }
}

/// The numeric class a format's clear value must have.
///
/// Section 31.1's "`Float` for floating-point or normalized formats, `Sint` for
/// signed-integer formats, and `Uint` for unsigned-integer formats", read off
/// the format itself. Returns `None` for the depth and depth/stencil formats,
/// which are never color attachments: an attachment whose format cannot be a
/// color attachment is refused by the device's format facts, and stating a class
/// for it here would invent an answer for a case that has none.
///
/// The format module owns this classification. This adapter translates its
/// shader-output vocabulary into the clear-value vocabulary used by command
/// validation, so adding a format cannot leave two independent class tables.
pub(crate) fn color_clear_class(format: TextureFormat) -> Option<ClearValueClass> {
    color_output_type(format).map(|class| match class {
        ShaderNumericType::Float32 => ClearValueClass::Float,
        ShaderNumericType::Sint32 => ClearValueClass::Sint,
        ShaderNumericType::Uint32 => ClearValueClass::Uint,
    })
}

/// Checks one color attachment against section 31.1.
///
/// `location` is only used to name the attachment in a refusal message; it is not
/// validated here, because a location is an index into the scope's own vector
/// and carries no bound of its own.
fn validate_color_attachment(location: u32, attachment: &ColorAttachment) -> RhiResult<()> {
    match &attachment.view {
        ColorAttachmentView::Texture(view)
            if view.descriptor().dimension == crate::api::resource::TextureViewDimension::D3 =>
        {
            let slice = attachment.depth_slice.ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!("color attachment {location} selects a 3D view but has no depth slice"),
                )
            })?;
            if slice >= view.extent().depth {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "color attachment {location} depth slice {slice} is outside the view depth {}",
                        view.extent().depth
                    ),
                ));
            }
        }
        _ if attachment.depth_slice.is_some() => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("color attachment {location} supplies a depth slice for a non-3D view"),
            ));
        }
        _ => {}
    }
    if !attachment.view.allows_color_attachment() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} uses a view whose texture was not created with \
                 COLOR_ATTACHMENT usage",
                location
            ),
        ));
    }

    if let Some(clear) = attachment.load.clear_value() {
        if let Some(class) = color_clear_class(attachment.view.format()) {
            if class != clear.class() {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "color attachment {} clears with a {} value but its format's \
                         clear class is {}",
                        location,
                        clear.class_name(),
                        class.as_str()
                    ),
                ));
            }
        }
    }

    if let Some(resolve) = &attachment.resolve {
        validate_resolve(location, attachment, resolve)?;
    }

    Ok(())
}

/// Checks the resolve pairing of one color attachment against section 31.1.
fn validate_resolve(
    location: u32,
    attachment: &ColorAttachment,
    resolve: &ColorAttachmentView,
) -> RhiResult<()> {
    if attachment.view.sample_count() <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} has a resolve target but its source is not multisampled",
                location
            ),
        ));
    }
    if resolve.sample_count() != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} resolves into a target with sample_count {}, not 1",
                location,
                resolve.sample_count()
            ),
        ));
    }
    if attachment.view.format() != resolve.format() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} resolves into a target of a different format",
                location
            ),
        ));
    }
    let source = attachment.view.extent();
    let target = resolve.extent();
    if source.width != target.width || source.height != target.height {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} and its resolve target differ in width or height",
                location
            ),
        ));
    }
    if !resolve.allows_resolve_into() {
        return Err(resolve_target_refusal(location, resolve));
    }
    if let (Some(source_id), Some(target_id)) = (
        attachment.view.texture_identity(),
        resolve.texture_identity(),
    ) {
        if source_id == target_id {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color attachment {} resolves into a view of its own texture, which is an \
                     overlapping read and write",
                    location
                ),
            ));
        }
    }
    if attachment.view.device_identity() != resolve.device_identity() {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!(
                "color attachment {} and its resolve target belong to different devices",
                location
            ),
        ));
    }

    // Section 31.1 is explicit that a raster resolve is a different route from
    // the standalone resolve command, and that it does *not* require
    // COPY_SRC/COPY_DST on either side. That is why the target check above is
    // exactly `allows_resolve_into` — the COLOR_ATTACHMENT half of section 31.1
    // plus section 46.1's frame route gate — and no further usage bit.
    Ok(())
}

/// The refusal for a resolve target [`ColorAttachmentView::allows_resolve_into`]
/// turned down.
///
/// The two variants are refused for different reasons, so they carry different
/// kinds and different sentences rather than one message covering both:
///
/// - A **texture** target fails the portable half of section 31.1 — its texture
///   was not created with `COLOR_ATTACHMENT` usage — which is the caller's own
///   descriptor, so it is [`RhiErrorKind::InvalidUsage`].
/// - A **frame** target is not a caller mistake at all. Section 46.1 permits the
///   direct multisampled resolve into a frame only once the active presentation
///   and route facts prove it, and this layer holds no such facts; the route is
///   therefore *not supported* here, which is [`RhiErrorKind::Unsupported`], the
///   kind the error module reserves for an unsupported route. The message names
///   the route the caller can use instead, because the refusal is about which
///   route is lawful and not about the frame being unusable.
fn resolve_target_refusal(location: u32, resolve: &ColorAttachmentView) -> RhiError {
    match resolve {
        ColorAttachmentView::Texture(_) => RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color attachment {} resolves into a view whose texture was not created with \
                 COLOR_ATTACHMENT usage",
                location
            ),
        ),
        ColorAttachmentView::Frame(_) => RhiError::new(
            RhiErrorKind::Unsupported,
            format!(
                "color attachment {} resolves directly into a frame attachment, which is legal \
                 only when the active presentation and route facts prove it; resolve into an \
                 intermediate single-sample texture and write that to the frame instead",
                location
            ),
        ),
    }
}

/// Checks the depth/stencil attachment against section 31.2.
fn validate_depth_stencil(attachment: &DepthStencilAttachment) -> RhiResult<()> {
    if !attachment
        .view
        .texture()
        .descriptor()
        .usage
        .contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the depth/stencil attachment's texture was not created with \
             DEPTH_STENCIL_ATTACHMENT usage",
        ));
    }

    let aspects = attachment.view.descriptor().aspects;
    if attachment.depth.is_some()
        && !aspects.contains(crate::api::resource::subresource::TextureAspects::DEPTH)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the depth attachment names a view that does not include the DEPTH aspect",
        ));
    }
    if attachment.stencil.is_some()
        && !aspects.contains(crate::api::resource::subresource::TextureAspects::STENCIL)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the stencil attachment names a view that does not include the STENCIL aspect",
        ));
    }
    if attachment.depth.is_none() && attachment.stencil.is_none() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "the depth/stencil attachment names neither a depth nor a stencil plane",
        ));
    }

    match &attachment.depth {
        Some(DepthAttachmentMode::ReadWrite {
            load: LoadOp::Clear(value),
            ..
        }) => {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    "the depth attachment clears to a value outside the finite range 0.0..=1.0",
                ));
            }
        }
        Some(DepthAttachmentMode::ReadWrite {
            load: LoadOp::Load, ..
        })
        | Some(DepthAttachmentMode::ReadOnly)
        | None => {}
    }

    Ok(())
}

/// Checks a whole attachment set against sections 31.1 through 31.4.
///
/// Called once by `begin_raster`, on the canonicalized descriptor. The order is
/// per-attachment first and set-wide second, so that a descriptor with one bad
/// attachment reports that attachment rather than a size mismatch caused by it.
pub(crate) fn validate_raster_scope(desc: &RasterScopeDescriptor) -> RhiResult<()> {
    if desc.is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a raster scope needs at least one color or depth/stencil attachment",
        ));
    }

    let mut set = SetFacts::default();

    for (location, attachment) in desc.attached_colors() {
        validate_color_attachment(location, attachment)?;

        let view = &attachment.view;
        if view.extent().width == 0 || view.extent().height == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("color attachment {} has a zero width or height", location),
            ));
        }
        let layers = match view {
            ColorAttachmentView::Texture(texture_view) => texture_view.layer_count(),
            // A frame is one acquired image: there is no layer the caller chose, so
            // there is nothing to report as a count.
            ColorAttachmentView::Frame(_) => 1,
        };
        // Section 31.1: a frame is rendered in order to be presented, so a frame
        // attachment marked Discard asks for contents that are thrown away. That
        // is refused here rather than at submit, where nothing could be done
        // about it.
        if matches!(view, ColorAttachmentView::Frame(_)) && attachment.store == StoreOp::Discard {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color attachment {} is a frame attachment marked Discard, and a frame that \
                     will be presented must Store",
                    location
                ),
            ));
        }

        compare_set_member(
            &mut set,
            view.device_identity(),
            view.extent(),
            view.sample_count(),
            layers,
            "color attachment",
            location,
        )?;
    }

    if let Some(depth) = &desc.depth_stencil {
        validate_depth_stencil(depth)?;

        let view = &depth.view;
        if view.extent().width == 0 || view.extent().height == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "the depth/stencil attachment has a zero width or height",
            ));
        }
        compare_set_member(
            &mut set,
            view.device_identity(),
            view.extent(),
            view.sample_count(),
            view.layer_count(),
            "the depth/stencil attachment",
            0,
        )?;
    }

    Ok(())
}

/// The four set-wide facts an attachment set is folded into.
///
/// Section 31.4's rule compares a member against all three at once, so they are
/// carried and passed as one value: a member that disagrees about any one of
/// them cannot be rasterized in the same scope, and the three are never read or
/// written apart from each other.
#[derive(Default)]
struct SetFacts {
    device: Option<DeviceIdentity>,
    extent: Option<(u32, u32)>,
    sample_count: Option<u32>,
    layer_count: Option<u32>,
}

/// Folds one attachment into the set-wide identity, extent, and sample count.
///
/// The first member seen fixes the set; every later member must agree. This is
/// section 31.4's three-set rule with the resolve target excluded, which is
/// expressed by simply never calling it for a resolve target.
fn compare_set_member(
    set: &mut SetFacts,
    member_device: DeviceIdentity,
    member_extent: Extent3d,
    member_samples: u32,
    member_layers: u32,
    what: &'static str,
    location: u32,
) -> RhiResult<()> {
    match set.device {
        Some(expected) if expected != member_device => {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                format!(
                    "{} {} belongs to a different device than the rest of the attachment set",
                    what, location
                ),
            ));
        }
        Some(_) => {}
        None => set.device = Some(member_device),
    }

    let size = (member_extent.width, member_extent.height);
    match set.extent {
        Some(expected) if expected != size => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "{} {} is {}x{}, which differs from the rest of the attachment set",
                    what, location, member_extent.width, member_extent.height
                ),
            ));
        }
        Some(_) => {}
        None => set.extent = Some(size),
    }

    match set.sample_count {
        Some(expected) if expected != member_samples => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "{} {} has sample_count {}, which differs from the rest of the attachment set",
                    what, location, member_samples
                ),
            ));
        }
        Some(_) => {}
        None => set.sample_count = Some(member_samples),
    }

    match set.layer_count {
        Some(expected) if expected != member_layers => {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "{} {} covers {} layers, which differs from the rest of the attachment set",
                    what, location, member_layers
                ),
            ));
        }
        Some(_) => {}
        None => set.layer_count = Some(member_layers),
    }

    Ok(())
}
