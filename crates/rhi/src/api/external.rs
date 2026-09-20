//! Opaque host image, external texture, and external-memory extension semantics.
//!
//! This module intentionally names no DOM, OS, Vulkan, or D3D type.  A host
//! bridge creates native-backed source handles through a crate-private seam.

use std::any::Any;
use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::Device;
use crate::api::platform::requirements::OptionalFeature;
use crate::api::resource::{
    Extent3d, Origin3d, Texture, TextureDescriptor, TextureSubresourceLayers,
};

/// Opaque host-owned external image source.
///
/// The public handle carries its owning device/context identity. Native browser
/// objects and host tokens remain inside its backend-private backing.
#[derive(Clone)]
pub struct ExternalImageSource {
    inner: Arc<ExternalImageSourceInner>,
}

struct ExternalImageSourceInner {
    id: ObjectId,
    device: DeviceIdentity,
    extent: Extent3d,
    native: Box<dyn ExternalImageSourceBackend>,
}

/// Backend-private native backing for one external image source.
pub(crate) trait ExternalImageSourceBackend: Send + Sync + 'static {
    /// Exposes native state solely to the owning backend.
    fn as_any(&self) -> &dyn Any;
}

impl ExternalImageSource {
    /// Creates a source only after a platform bridge attached native backing.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        extent: Extent3d,
        native: Box<dyn ExternalImageSourceBackend>,
    ) -> RhiResult<Self> {
        if extent.width == 0 || extent.height == 0 || extent.depth != 1 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an external image source needs non-zero two-dimensional extent",
            ));
        }
        Ok(Self {
            inner: Arc::new(ExternalImageSourceInner {
                id,
                device,
                extent,
                native,
            }),
        })
    }
    /// Source dimensions in texels.
    pub fn extent(&self) -> Extent3d {
        self.inner.extent
    }
    /// Process-local source identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Owning device/context identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    pub(crate) fn native(&self) -> &dyn ExternalImageSourceBackend {
        self.inner.native.as_ref()
    }
}

impl std::fmt::Debug for ExternalImageSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalImageSource")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .field("extent", &self.extent())
            .finish()
    }
}

/// How premultiplied-alpha input is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalAlphaMode {
    /// Input RGB already has alpha applied.
    Premultiplied,
    /// Input RGB is independent of alpha.
    Unpremultiplied,
}
/// Requested color-space conversion for an external copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalColorSpaceConversion {
    /// Do not request a color-space conversion.
    None,
    /// Request the backend's default conversion where supported.
    Default,
}

/// Restrictions the active device applies to external-image copies.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalImageCopyCapabilities {
    /// Whether source and destination origins/extents are unrestricted.
    pub unrestricted_copies: bool,
    /// Whether vertical flipping is supported.
    pub flip_y: bool,
    /// Whether alpha interpretation can be selected.
    pub alpha_mode: bool,
    /// Whether color-space conversion can be requested.
    pub color_space_conversion: bool,
}

/// A copy from opaque host image data into a normal RHI texture.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ExternalImageCopyDescriptor {
    /// Opaque host image to read.
    pub source: ExternalImageSource,
    /// First source texel.
    pub source_origin: Origin3d,
    /// Ordinary portable destination texture.
    pub destination: Texture,
    /// Destination mip, aspect, and array layers.
    pub destination_subresource: TextureSubresourceLayers,
    /// First destination texel.
    pub destination_origin: Origin3d,
    /// Copied region size.
    pub extent: Extent3d,
    /// Whether source rows are vertically flipped.
    pub flip_y: bool,
    /// Requested source alpha interpretation.
    pub alpha_mode: ExternalAlphaMode,
    /// Requested color-space conversion.
    pub color_space_conversion: ExternalColorSpaceConversion,
}

impl ExternalImageCopyDescriptor {
    /// Creates a descriptor with no coordinate transform or color conversion.
    pub fn new(source: ExternalImageSource, destination: Texture, extent: Extent3d) -> Self {
        Self {
            source,
            source_origin: Origin3d { x: 0, y: 0, z: 0 },
            destination,
            destination_subresource: TextureSubresourceLayers {
                aspect: crate::api::resource::TextureAspect::Color,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            destination_origin: Origin3d { x: 0, y: 0, z: 0 },
            extent,
            flip_y: false,
            alpha_mode: ExternalAlphaMode::Premultiplied,
            color_space_conversion: ExternalColorSpaceConversion::Default,
        }
    }
}

/// A sampled, host-backed external texture handle.
#[derive(Clone)]
pub struct ExternalTexture {
    inner: Arc<ExternalTextureInner>,
}
struct ExternalTextureInner {
    id: ObjectId,
    device: DeviceIdentity,
    source: ExternalImageSource,
    label: Label,
}

/// Input used to import one [`ExternalTexture`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ExternalTextureDescriptor {
    /// Diagnostic label.
    pub label: Label,
    /// Opaque host source to expose for sampling.
    pub source: ExternalImageSource,
}
impl ExternalTextureDescriptor {
    /// Imports `source` with no label.
    pub fn new(source: ExternalImageSource) -> Self {
        Self {
            label: Label::default(),
            source,
        }
    }
    /// Sets a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}
impl ExternalTexture {
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: ExternalTextureDescriptor,
    ) -> Self {
        Self {
            inner: Arc::new(ExternalTextureInner {
                id,
                device,
                source: descriptor.source,
                label: descriptor.label,
            }),
        }
    }
    /// Process-local object identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Owning device identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// Original opaque host source.
    pub fn source(&self) -> &ExternalImageSource {
        &self.inner.source
    }
    /// Diagnostic label.
    pub fn label(&self) -> &Label {
        &self.inner.label
    }
}
impl std::fmt::Debug for ExternalTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalTexture")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .finish()
    }
}

/// Platform-neutral external-memory handle class used only by the extension SPI.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalMemoryHandleType {
    /// An opaque Unix file descriptor.
    OpaqueFd,
    /// A Linux DMA-BUF descriptor.
    DmaBuf,
    /// A Windows kernel handle.
    Win32Handle,
}
/// Which external-memory handle classes the device can safely import through
/// the platform-extension SPI.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalMemoryCapabilities {
    /// Handle classes available through the extension SPI.
    pub supported_handle_types: Vec<ExternalMemoryHandleType>,
}
impl ExternalMemoryCapabilities {
    /// Whether one generic handle class is supported.
    pub fn supports(&self, ty: ExternalMemoryHandleType) -> bool {
        self.supported_handle_types.contains(&ty)
    }
}

/// A platform bridge-owned external-memory texture source.
///
/// It owns neither a raw `HANDLE` nor an fd in the public contract.  Those stay
/// in native backing, while this handle records the exact portable texture
/// contract the exported allocation represents.
#[derive(Clone)]
pub struct ExternalMemoryTextureSource {
    inner: Arc<ExternalMemoryTextureSourceInner>,
}

struct ExternalMemoryTextureSourceInner {
    id: ObjectId,
    device: DeviceIdentity,
    handle_type: ExternalMemoryHandleType,
    texture: TextureDescriptor,
    native: Box<dyn ExternalMemoryTextureSourceBackend>,
}

/// Backend-private backing for an importable external-memory allocation.
pub(crate) trait ExternalMemoryTextureSourceBackend: Send + Sync + 'static {
    /// Exposes native state solely to its owning backend.
    fn as_any(&self) -> &dyn Any;
}

impl ExternalMemoryTextureSource {
    /// Publishes a source only after a platform extension has verified its
    /// native handle type and immutable texture contract.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        handle_type: ExternalMemoryHandleType,
        texture: TextureDescriptor,
        native: Box<dyn ExternalMemoryTextureSourceBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(ExternalMemoryTextureSourceInner {
                id,
                device,
                handle_type,
                texture,
                native,
            }),
        }
    }
    /// Process-local source identity.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }
    /// Owning device/context identity.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }
    /// Generic class of the native handle, not its value.
    pub fn handle_type(&self) -> ExternalMemoryHandleType {
        self.inner.handle_type
    }
    /// Immutable portable texture contract verified by the extension source.
    pub fn texture_descriptor(&self) -> &TextureDescriptor {
        &self.inner.texture
    }
    pub(crate) fn native(&self) -> &dyn ExternalMemoryTextureSourceBackend {
        self.inner.native.as_ref()
    }
}

impl std::fmt::Debug for ExternalMemoryTextureSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalMemoryTextureSource")
            .field("id", &self.id())
            .field("device", &self.device_identity())
            .field("handle_type", &self.handle_type())
            .finish_non_exhaustive()
    }
}

/// Descriptor for importing an extension-owned external-memory texture.
#[non_exhaustive]
#[derive(Clone)]
pub struct ExternalTextureImportDescriptor {
    /// The extension-created source whose immutable contract is imported.
    pub source: ExternalMemoryTextureSource,
    /// Diagnostic-only label for the created ordinary texture.
    pub label: Label,
}

impl ExternalTextureImportDescriptor {
    /// Imports exactly the texture contract carried by `source`.
    pub fn new(source: ExternalMemoryTextureSource) -> Self {
        Self {
            source,
            label: Label::default(),
        }
    }
    /// Sets the imported texture's diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

impl std::fmt::Debug for ExternalTextureImportDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalTextureImportDescriptor")
            .field("source", &self.source)
            .field("label", &self.label)
            .finish()
    }
}

impl Device {
    /// Returns external-image restrictions after the feature gate.
    pub fn external_image_copy_capabilities(&self) -> RhiResult<ExternalImageCopyCapabilities> {
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::ExternalImageCopy)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "external image copies are not enabled",
            )
            .at("Device::external_image_copy_capabilities"));
        }
        self.native().external_image_copy_capabilities()
    }
    /// Imports an opaque external texture.  It has no ordinary `TextureView`;
    /// only the dedicated external-texture binding semantic may consume it.
    pub fn import_external_texture(
        &self,
        descriptor: ExternalTextureDescriptor,
    ) -> RhiResult<ExternalTexture> {
        if descriptor.source.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the external image source belongs to a different device/context",
            )
            .with_object(descriptor.source.id())
            .at("Device::import_external_texture"));
        }
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::ExternalTexture)
        {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "external textures are not enabled",
            )
            .at("Device::import_external_texture"));
        }
        self.native()
            .validate_external_texture_source(&descriptor.source)?;
        Ok(ExternalTexture::new(
            ObjectId::next(),
            self.identity(),
            descriptor,
        ))
    }
    /// Returns extension-SPI external-memory capability without leaking a native handle.
    pub fn external_memory_capabilities(&self) -> RhiResult<ExternalMemoryCapabilities> {
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::ExternalMemory)
        {
            return Err(
                RhiError::new(RhiErrorKind::Unsupported, "external memory is not enabled")
                    .at("Device::external_memory_capabilities"),
            );
        }
        self.native().external_memory_capabilities()
    }

    /// Imports external memory as an ordinary RHI texture.
    ///
    /// The texture descriptor is not caller-editable here: an imported native
    /// allocation cannot safely be reinterpreted as another shape or format.
    pub fn import_external_memory_texture(
        &self,
        descriptor: &ExternalTextureImportDescriptor,
    ) -> RhiResult<Texture> {
        if descriptor.source.device_identity() != self.identity() {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the external-memory source belongs to a different device/context",
            )
            .with_object(descriptor.source.id())
            .at("Device::import_external_memory_texture"));
        }
        self.require_active()?;
        if !self
            .capabilities()
            .supports_feature(OptionalFeature::ExternalMemory)
        {
            return Err(
                RhiError::new(RhiErrorKind::Unsupported, "external memory is not enabled")
                    .at("Device::import_external_memory_texture"),
            );
        }
        let capabilities = self.native().external_memory_capabilities()?;
        if !capabilities.supports(descriptor.source.handle_type()) {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "the external-memory handle class is not enabled on this device",
            )
            .at("Device::import_external_memory_texture"));
        }
        let mut accepted = descriptor.source.texture_descriptor().clone();
        let mut query = crate::api::format::TextureSupportQuery::new(
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
        crate::api::resource::texture::validate_texture_descriptor(&mut accepted, &support)?;
        let native = self
            .native()
            .import_external_memory_texture(descriptor, &accepted)?;
        Ok(Texture::new_backed(
            ObjectId::next(),
            self.identity(),
            accepted,
            native,
        ))
    }
}
