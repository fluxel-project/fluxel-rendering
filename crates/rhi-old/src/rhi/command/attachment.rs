//! Raster attachment descriptions and the raster scope descriptor.
//!
//! This module owns rhi-design section 31.
//!
//! # What it is
//!
//! The attachment set of one raster scope, expressed in portable terms: a color
//! location holds either an ordinary texture view or a presentation frame, and
//! depth/stencil carries its own load/store mode per aspect.
//!
//! # What it deliberately does not own
//!
//! Layered and multiview rasterization. Every main attachment view must address
//! exactly one layer, so a scope cannot silently mean "render to all six faces"
//! on one backend and "render to face zero" on another.
//!
//! # Read-only is a mode, not a flag
//!
//! A `read_only` boolean alongside a load operation can express
//! `read_only + Clear`, which has no meaning. Splitting the mode into
//! [`DepthAttachmentMode::ReadOnly`] and [`DepthAttachmentMode::ReadWrite`]
//! makes that combination unconstructible rather than something a validator has
//! to reject.

use super::super::format::{TextureFormat, format_facts};
use super::super::platform::{DeviceIdentity, Label, RhiError, RhiErrorKind, RhiResult};
use super::super::presentation::{AcquiredFrameState, FrameAttachment};
use super::super::resource::{Extent3d, TextureAspects, TextureUsage, TextureView};
use super::super::shader::{ShaderLocation, ShaderNumericType};
use super::values::{ClearValueClass, ColorClearValue, LoadOp, StoreOp};

/// Where a color attachment's texels come from.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ColorAttachmentView {
    /// An ordinary texture view.
    Texture(TextureView),
    /// A presentation frame's drawable.
    Frame(FrameAttachment),
}

impl ColorAttachmentView {
    /// The device identity this attachment belongs to.
    pub fn device_identity(&self) -> DeviceIdentity {
        match self {
            Self::Texture(view) => view.device_identity(),
            Self::Frame(frame) => frame.device_identity(),
        }
    }

    /// The attachment's format.
    pub fn format(&self) -> TextureFormat {
        match self {
            Self::Texture(view) => view.format(),
            Self::Frame(frame) => frame.format(),
        }
    }

    /// The attachment's texel extent.
    pub fn extent(&self) -> Extent3d {
        match self {
            Self::Texture(view) => view.extent(),
            Self::Frame(frame) => frame.extent(),
        }
    }

    /// The attachment's sample count.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::Texture(view) => view.sample_count(),
            Self::Frame(frame) => frame.sample_count(),
        }
    }

    /// Whether this attachment is a presentation frame.
    pub fn is_frame(&self) -> bool {
        matches!(self, Self::Frame(_))
    }
}

/// One color attachment location.
#[derive(Clone, Debug)]
pub struct ColorAttachment {
    /// The attachment's texels.
    pub view: ColorAttachmentView,
    /// What happens to the contents when the scope begins.
    pub load: LoadOp<ColorClearValue>,
    /// What happens to the contents when the scope ends.
    pub store: StoreOp,
    /// The single-sample resolve destination, when the attachment is multisampled.
    pub resolve: Option<ColorAttachmentView>,
}

impl ColorAttachment {
    /// A color attachment that preserves its contents and keeps the result.
    pub fn new(view: ColorAttachmentView) -> Self {
        Self {
            view,
            load: LoadOp::Load,
            store: StoreOp::Store,
            resolve: None,
        }
    }

    /// Sets the load operation.
    pub fn with_load(mut self, load: LoadOp<ColorClearValue>) -> Self {
        self.load = load;
        self
    }

    /// Sets the store operation.
    pub fn with_store(mut self, store: StoreOp) -> Self {
        self.store = store;
        self
    }

    /// Sets the resolve destination.
    pub fn with_resolve(mut self, resolve: ColorAttachmentView) -> Self {
        self.resolve = Some(resolve);
        self
    }
}

/// What a depth aspect does across a raster scope.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum DepthAttachmentMode {
    /// The aspect is readable and never written.
    ReadOnly,
    /// The aspect is read and written with the given load/store behaviour.
    ReadWrite {
        /// What happens to the contents when the scope begins.
        load: LoadOp<f32>,
        /// What happens to the contents when the scope ends.
        store: StoreOp,
    },
}

/// What a stencil aspect does across a raster scope.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum StencilAttachmentMode {
    /// The aspect is readable and never written.
    ReadOnly,
    /// The aspect is read and written with the given load/store behaviour.
    ReadWrite {
        /// What happens to the contents when the scope begins.
        load: LoadOp<u32>,
        /// What happens to the contents when the scope ends.
        store: StoreOp,
    },
}

/// A depth and/or stencil attachment.
#[derive(Clone, Debug)]
pub struct DepthStencilAttachment {
    /// The attachment's texels.
    pub view: TextureView,
    /// The depth aspect's mode. `None` leaves the aspect untouched.
    pub depth: Option<DepthAttachmentMode>,
    /// The stencil aspect's mode. `None` leaves the aspect untouched.
    pub stencil: Option<StencilAttachmentMode>,
}

impl DepthStencilAttachment {
    /// An attachment that touches neither aspect until a mode is set.
    pub fn new(view: TextureView) -> Self {
        Self {
            view,
            depth: None,
            stencil: None,
        }
    }

    /// Sets the depth aspect's mode.
    pub fn with_depth(mut self, mode: DepthAttachmentMode) -> Self {
        self.depth = Some(mode);
        self
    }

    /// Sets the stencil aspect's mode.
    pub fn with_stencil(mut self, mode: StencilAttachmentMode) -> Self {
        self.stencil = Some(mode);
        self
    }
}

/// The attachment set of one raster scope.
///
/// `colors` is indexed by color location, so `colors[2] == None` means location
/// two is unwritten while locations zero and one are live. Trailing `None`s are
/// removed at construction, because `[Some(a)]` and `[Some(a), None, None]` are
/// the same scope and must not compare unequal.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RasterScopeDescriptor {
    /// A diagnostic label.
    pub label: Label,
    /// The color attachments, indexed by location.
    pub colors: Vec<Option<ColorAttachment>>,
    /// The depth/stencil attachment.
    pub depth_stencil: Option<DepthStencilAttachment>,
}

impl Default for RasterScopeDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl RasterScopeDescriptor {
    /// A scope with no attachments.
    pub fn new() -> Self {
        Self {
            label: Label::none(),
            colors: Vec::new(),
            depth_stencil: None,
        }
    }

    /// Sets the diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label::new(label);
        self
    }

    /// Places an attachment at a color location, growing the vector as needed.
    ///
    /// Locations below `location` that were never set stay `None`; a hole in the
    /// middle of the set is a different scope from a shorter set, so it is kept.
    pub fn with_color(mut self, location: ShaderLocation, attachment: ColorAttachment) -> Self {
        let index = location.get() as usize;
        if self.colors.len() <= index {
            self.colors.resize(index + 1, None);
        }
        self.colors[index] = Some(attachment);
        while matches!(self.colors.last(), Some(None)) {
            self.colors.pop();
        }
        self
    }

    /// Sets the depth/stencil attachment.
    pub fn with_depth_stencil(mut self, attachment: DepthStencilAttachment) -> Self {
        self.depth_stencil = Some(attachment);
        self
    }

    /// How many color locations are live.
    pub fn active_color_count(&self) -> u32 {
        self.colors.iter().filter(|slot| slot.is_some()).count() as u32
    }
}

/// The numeric class a format's color attachment clear value must carry.
///
/// `None` means the format has no color output at all, which is the case for
/// every depth/stencil format; such a format cannot be a color attachment and is
/// refused before a clear value is ever inspected.
fn clear_class_of(format: TextureFormat) -> Option<ClearValueClass> {
    let numeric = format_facts(format).color_output_type()?;
    Some(match numeric {
        ShaderNumericType::Float32 => ClearValueClass::Float,
        ShaderNumericType::Sint32 => ClearValueClass::Sint,
        ShaderNumericType::Uint32 => ClearValueClass::Uint,
    })
}

/// Validates one color attachment against its format and resolve partner.
pub(crate) fn validate_color_attachment(
    location: u32,
    attachment: &ColorAttachment,
    max_color_attachments: Option<u32>,
) -> RhiResult<()> {
    let view = &attachment.view;
    let format = view.format();
    let facts = format_facts(format);
    if !facts.color_attachment() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("color location {location}: format {format:?} is not a color attachment format"),
        ));
    }

    if let ColorAttachmentView::Texture(texture_view) = view {
        if !texture_view
            .texture()
            .descriptor()
            .usage
            .contains(TextureUsage::COLOR_ATTACHMENT)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("color location {location}: texture was not created with COLOR_ATTACHMENT"),
            ));
        }
    }

    if let ColorAttachmentView::Frame(frame) = view {
        let state = frame.state();
        if state != AcquiredFrameState::Acquired {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("color location {location}: frame is {state:?}, not Acquired"),
            ));
        }
        if attachment.store != StoreOp::Store {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color location {location}: a presentation frame cannot be discarded before present"
                ),
            ));
        }
        if attachment.resolve.is_some() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!("color location {location}: a presentation frame cannot be a resolve source"),
            ));
        }
    }

    if let LoadOp::Clear(value) = attachment.load {
        let required = clear_class_of(format).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::Unsupported,
                format!("color location {location}: format {format:?} has no clear value class"),
            )
        })?;
        if value.class() != required {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color location {location}: a {:?} clear cannot initialize a {required:?} attachment",
                    value.class()
                ),
            ));
        }
    }

    if let Some(resolve) = &attachment.resolve {
        validate_resolve(location, view, resolve)?;
    }

    if let Some(limit) = max_color_attachments {
        if location >= limit {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("color location {location} exceeds the device's {limit} color attachments"),
            ));
        }
    }

    Ok(())
}

/// Validates a multisample resolve pair.
fn validate_resolve(
    location: u32,
    view: &ColorAttachmentView,
    resolve: &ColorAttachmentView,
) -> RhiResult<()> {
    if resolve.device_identity() != view.device_identity() {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            format!("color location {location}: resolve target belongs to another device"),
        ));
    }
    if view.sample_count() <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("color location {location}: a single-sample attachment has nothing to resolve"),
        ));
    }
    if resolve.sample_count() != 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("color location {location}: a resolve target must be single-sampled"),
        ));
    }
    if resolve.format() != view.format() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "color location {location}: resolve format {:?} differs from attachment format {:?}",
                resolve.format(),
                view.format()
            ),
        ));
    }
    let source_extent = view.extent();
    let resolve_extent = resolve.extent();
    if source_extent.width != resolve_extent.width || source_extent.height != resolve_extent.height {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("color location {location}: resolve extent differs from the attachment extent"),
        ));
    }
    if let ColorAttachmentView::Texture(resolve_view) = resolve {
        if !resolve_view
            .texture()
            .descriptor()
            .usage
            .contains(TextureUsage::COLOR_ATTACHMENT)
        {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color location {location}: resolve target was not created with COLOR_ATTACHMENT"
                ),
            ));
        }
    }
    // A resolve source and its destination are the same texels read and written
    // in one command; overlapping them is undefined on every backend, so it is
    // refused here rather than left to a driver.
    if let (ColorAttachmentView::Texture(source), ColorAttachmentView::Texture(target)) =
        (view, resolve)
    {
        if source.id() == target.id() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "color location {location}: attachment and resolve target are the same texture"
                ),
            ));
        }
    }
    Ok(())
}

/// Validates a depth/stencil attachment against its view's aspects.
pub(crate) fn validate_depth_stencil_attachment(
    attachment: &DepthStencilAttachment,
) -> RhiResult<()> {
    let view = &attachment.view;
    let format = view.format();
    let facts = format_facts(format);
    if !view
        .texture()
        .descriptor()
        .usage
        .contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "depth/stencil view was not created with DEPTH_STENCIL_ATTACHMENT",
        ));
    }

    let aspects = view.aspects();
    if attachment.depth.is_none() && attachment.stencil.is_none() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a depth/stencil attachment must declare at least one aspect",
        ));
    }
    if attachment.depth.is_some() && !aspects.contains(TextureAspects::DEPTH) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("view format {format:?} has no depth aspect"),
        ));
    }
    if attachment.stencil.is_some() && !aspects.contains(TextureAspects::STENCIL) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("view format {format:?} has no stencil aspect"),
        ));
    }
    if attachment.depth.is_some() && !facts.depth_attachment() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("format {format:?} is not a depth attachment format"),
        ));
    }
    if attachment.stencil.is_some() && !facts.stencil_attachment() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!("format {format:?} is not a stencil attachment format"),
        ));
    }

    if let Some(DepthAttachmentMode::ReadWrite {
        load: LoadOp::Clear(value),
        ..
    }) = attachment.depth
    {
        if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a depth clear value must be finite and lie in 0.0..=1.0",
            ));
        }
    }

    if let Some(StencilAttachmentMode::ReadWrite {
        load: LoadOp::Clear(value),
        ..
    }) = attachment.stencil
    {
        if value > 255 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "a stencil clear value must fit in eight bits",
            ));
        }
    }

    Ok(())
}

/// The shared geometry every main attachment of a scope must agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttachmentGeometry {
    /// The device every attachment belongs to.
    pub device: DeviceIdentity,
    /// The shared texel width.
    pub width: u32,
    /// The shared texel height.
    pub height: u32,
    /// The shared sample count.
    pub sample_count: u32,
}

/// Accumulates the geometry of a scope's main attachments.
struct GeometryCheck {
    geometry: Option<AttachmentGeometry>,
}

impl GeometryCheck {
    fn new() -> Self {
        Self { geometry: None }
    }

    fn observe(
        &mut self,
        identity: DeviceIdentity,
        extent: Extent3d,
        sample_count: u32,
        what: &str,
    ) -> RhiResult<()> {
        let Some(current) = self.geometry else {
            self.geometry = Some(AttachmentGeometry {
                device: identity,
                width: extent.width,
                height: extent.height,
                sample_count,
            });
            return Ok(());
        };
        if current.device != identity {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                format!("{what} belongs to another device than the other attachments"),
            ));
        }
        if current.width != extent.width || current.height != extent.height {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "{what} is {}x{} but the scope is {}x{}",
                    extent.width, extent.height, current.width, current.height
                ),
            ));
        }
        if current.sample_count != sample_count {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "{what} has {sample_count} samples but the scope has {}",
                    current.sample_count
                ),
            ));
        }
        Ok(())
    }
}

/// Validates a whole scope's attachment set and returns its shared geometry.
///
/// Resolve targets are checked for identity and format by
/// [`validate_color_attachment`] but do not participate in the shared geometry:
/// they are deliberately single-sampled where their source is not.
pub(crate) fn validate_scope_attachments(
    desc: &RasterScopeDescriptor,
    max_color_attachments: Option<u32>,
) -> RhiResult<AttachmentGeometry> {
    let mut check = GeometryCheck::new();

    for (index, slot) in desc.colors.iter().enumerate() {
        let Some(attachment) = slot else {
            continue;
        };
        validate_color_attachment(index as u32, attachment, max_color_attachments)?;
        // A layer count other than one would make the same scope mean different
        // things on different backends, so P0 refuses it instead of picking one.
        if let ColorAttachmentView::Texture(view) = &attachment.view {
            if view.layer_count() != 1 {
                return Err(RhiError::new(
                    RhiErrorKind::Unsupported,
                    format!("color location {index} must address exactly one layer"),
                ));
            }
        }
        check.observe(
            attachment.view.device_identity(),
            attachment.view.extent(),
            attachment.view.sample_count(),
            &format!("color location {index}"),
        )?;
    }

    if let Some(depth_stencil) = &desc.depth_stencil {
        validate_depth_stencil_attachment(depth_stencil)?;
        let view = &depth_stencil.view;
        if view.layer_count() != 1 {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "a depth/stencil attachment must address exactly one layer",
            ));
        }
        check.observe(
            view.device_identity(),
            view.extent(),
            view.sample_count(),
            "the depth/stencil attachment",
        )?;
    }

    check.geometry.ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a raster scope needs at least one color or depth/stencil attachment",
        )
    })
}
