//! Dedicated Metal resource ownership and descriptor lowering.

use std::any::Any;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBuffer, MTLCompareFunction, MTLDevice, MTLPixelFormat, MTLResource, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLSamplerState, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::resource::backend::{
    BufferBackend, MappedBufferBackend, MappingRequestBackend, SamplerBackend, TextureBackend,
    TextureViewBackend,
};
use crate::api::resource::buffer::{BufferDescriptor, BufferUsage};
use crate::api::resource::sampler::{
    AddressMode, CompareFunction, FilterMode, SamplerBorderColor, SamplerDescriptor,
};
use crate::api::resource::texture::{TextureDescriptor, TextureDimension, TextureUsage};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::resource::{BufferRange, MapMode};

use super::command::{SpineState, completion_or_register_waker};
use crate::api::submission::CompletionState;

pub(super) struct MetalBuffer {
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) size: u64,
    host_visible: bool,
    access: Arc<MetalBufferAccess>,
}

/// Mapping visibility is intentionally one small shared record per buffer,
/// rather than another ownership wrapper around the full native object.
struct MetalBufferAccess {
    last_accepted_serial: AtomicU64,
    completion: Arc<Mutex<SpineState>>,
}

impl MetalBuffer {
    pub(super) fn mark_accepted(&self, serial: u64) {
        self.access
            .last_accepted_serial
            .fetch_max(serial, Ordering::Release);
    }
}

// Metal objects may be retained and submitted across threads. Command encoder
// mutation is serialized by MetalShared; resource handles themselves are
// immutable after creation, matching wgpu-hal's ownership boundary.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl BufferBackend for MetalBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) struct MetalTexture {
    pub(super) raw: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(super) format: MTLPixelFormat,
    pub(super) ty: MTLTextureType,
    pub(super) mip_levels: u32,
    pub(super) array_layers: u32,
}

unsafe impl Send for MetalTexture {}
unsafe impl Sync for MetalTexture {}

impl TextureBackend for MetalTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) struct MetalTextureView {
    pub(super) raw: Retained<ProtocolObject<dyn MTLTexture>>,
}

unsafe impl Send for MetalTextureView {}
unsafe impl Sync for MetalTextureView {}

impl TextureViewBackend for MetalTextureView {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) struct MetalSampler {
    pub(super) raw: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

unsafe impl Send for MetalSampler {}
unsafe impl Sync for MetalSampler {}

impl SamplerBackend for MetalSampler {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) fn create_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &BufferDescriptor,
    completion: Arc<Mutex<SpineState>>,
) -> RhiResult<MetalBuffer> {
    let host_visible = descriptor.usage.contains(BufferUsage::MAP_READ)
        || descriptor.usage.contains(BufferUsage::MAP_WRITE);
    let mut options = if host_visible {
        MTLResourceOptions::StorageModeShared
    } else {
        MTLResourceOptions::StorageModePrivate
    };
    if descriptor.usage.contains(BufferUsage::MAP_WRITE) {
        options |= MTLResourceOptions::CPUCacheModeWriteCombined;
    }
    let length = usize::try_from(descriptor.size).map_err(|_| {
        RhiError::new(
            RhiErrorKind::OutOfMemory,
            "buffer exceeds host address space",
        )
        .at("MetalDevice::create_buffer")
    })?;
    let raw = device
        .newBufferWithLength_options(length, options)
        .ok_or_else(|| {
            RhiError::new(RhiErrorKind::OutOfMemory, "Metal buffer allocation failed")
                .at("MetalDevice::create_buffer")
        })?;
    if let Some(label) = descriptor.label.as_deref() {
        raw.setLabel(Some(&NSString::from_str(label)));
    }
    Ok(MetalBuffer {
        raw,
        size: descriptor.size,
        host_visible,
        access: Arc::new(MetalBufferAccess {
            last_accepted_serial: AtomicU64::new(0),
            completion,
        }),
    })
}

pub(super) fn map_buffer(
    buffer: &MetalBuffer,
    mode: MapMode,
    range: BufferRange,
) -> RhiResult<Box<dyn MappingRequestBackend>> {
    if !buffer.host_visible {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal private-storage buffers cannot be mapped",
        ));
    }
    let offset = usize::try_from(range.offset).map_err(|_| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mapping offset exceeds address space",
        )
    })?;
    let length = usize::try_from(range.size).map_err(|_| {
        RhiError::new(
            RhiErrorKind::InvalidUsage,
            "mapping size exceeds address space",
        )
    })?;
    Ok(Box::new(MetalMapRequest {
        raw: buffer.raw.clone(),
        access: Arc::clone(&buffer.access),
        accepted: buffer.access.last_accepted_serial.load(Ordering::Acquire),
        offset,
        length,
        writable: matches!(mode, MapMode::Write),
        consumed: false,
    }))
}

struct MetalMapRequest {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    access: Arc<MetalBufferAccess>,
    accepted: u64,
    offset: usize,
    length: usize,
    writable: bool,
    consumed: bool,
}

impl MappingRequestBackend for MetalMapRequest {
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<RhiResult<Box<dyn MappedBufferBackend>>> {
        if self.consumed {
            return Poll::Ready(Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "mapping request reused",
            )));
        }
        // Submission can be accepted after the request object was created but
        // before its first poll. Re-sample the monotonic serial here; otherwise
        // that race would expose shared storage while Metal still owns it.
        self.accepted = self
            .accepted
            .max(self.access.last_accepted_serial.load(Ordering::Acquire));
        if self.accepted != 0 {
            match completion_or_register_waker(
                &self.access.completion,
                self.accepted,
                context.waker(),
            ) {
                CompletionState::Pending => return Poll::Pending,
                CompletionState::Complete => {}
                CompletionState::DeviceLost(info) => {
                    return Poll::Ready(Err(RhiError::new(
                        RhiErrorKind::DeviceLost,
                        info.message().to_owned(),
                    )));
                }
                CompletionState::Failed(failure) => {
                    return Poll::Ready(Err(RhiError::new(
                        RhiErrorKind::BackendFailure,
                        failure.message().to_owned(),
                    )));
                }
            }
        }
        let base = self.raw.contents().cast::<u8>();
        let pointer = NonNull::new(unsafe { base.as_ptr().add(self.offset) }).ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal returned a null buffer mapping",
            )
        })?;
        self.consumed = true;
        Poll::Ready(Ok(Box::new(MetalMappedBuffer {
            // The request owns this retained native buffer until its mapping
            // lease is consumed, so the pointer cannot outlive storage.
            raw: self.raw.clone(),
            pointer,
            length: self.length,
            writable: self.writable,
        })))
    }
}

struct MetalMappedBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pointer: NonNull<u8>,
    length: usize,
    writable: bool,
}

impl MappedBufferBackend for MetalMappedBuffer {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
    fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.writable
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) })
    }
    fn flush(&mut self) -> RhiResult<()> {
        Ok(())
    }
    fn invalidate(&mut self) -> RhiResult<()> {
        Ok(())
    }
}

pub(super) fn create_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &TextureDescriptor,
) -> RhiResult<MetalTexture> {
    let format = super::format::metal_format(descriptor.format).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "texture format has no Metal lowering",
        )
        .at("MetalDevice::create_texture")
    })?;
    let native = MTLTextureDescriptor::new();
    let ty = match descriptor.dimension {
        TextureDimension::D1 => MTLTextureType::Type1D,
        TextureDimension::D2 if descriptor.sample_count > 1 && descriptor.array_layers > 1 => {
            unsafe { native.setArrayLength(descriptor.array_layers as usize) };
            MTLTextureType::Type2DMultisampleArray
        }
        TextureDimension::D2 if descriptor.sample_count > 1 => MTLTextureType::Type2DMultisample,
        TextureDimension::D2 if descriptor.array_layers > 1 => {
            unsafe { native.setArrayLength(descriptor.array_layers as usize) };
            MTLTextureType::Type2DArray
        }
        TextureDimension::D2 => MTLTextureType::Type2D,
        TextureDimension::D3 => {
            unsafe { native.setDepth(descriptor.extent.depth as usize) };
            MTLTextureType::Type3D
        }
    };
    native.setTextureType(ty);
    unsafe {
        native.setWidth(descriptor.extent.width as usize);
        native.setHeight(descriptor.extent.height as usize);
        native.setMipmapLevelCount(descriptor.mip_levels as usize);
        native.setSampleCount(descriptor.sample_count as usize);
    }
    native.setPixelFormat(format);
    native.setStorageMode(MTLStorageMode::Private);
    native.setUsage(texture_usage(
        descriptor.usage,
        !descriptor.view_formats.is_empty(),
    ));
    let raw = device.newTextureWithDescriptor(&native).ok_or_else(|| {
        RhiError::new(RhiErrorKind::OutOfMemory, "Metal texture allocation failed")
            .at("MetalDevice::create_texture")
    })?;
    if let Some(label) = descriptor.label.as_deref() {
        raw.setLabel(Some(&NSString::from_str(label)));
    }
    Ok(MetalTexture {
        raw,
        format,
        ty,
        mip_levels: descriptor.mip_levels,
        array_layers: descriptor.array_layers,
    })
}

pub(super) fn create_texture_view(
    texture: &MetalTexture,
    descriptor: &TextureViewDescriptor,
    format: MTLPixelFormat,
) -> RhiResult<MetalTextureView> {
    let ty = view_type(descriptor.dimension);
    let full = format == texture.format
        && ty == texture.ty
        && descriptor.base_mip == 0
        && descriptor.mip_count == texture.mip_levels
        && descriptor.base_layer == 0
        && descriptor.layer_count == texture.array_layers;
    let raw = if full {
        texture.raw.clone()
    } else {
        unsafe {
            texture
                .raw
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    format,
                    ty,
                    NSRange {
                        location: descriptor.base_mip as usize,
                        length: descriptor.mip_count as usize,
                    },
                    NSRange {
                        location: descriptor.base_layer as usize,
                        length: descriptor.layer_count as usize,
                    },
                )
        }
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal rejected the texture view",
            )
            .at("MetalDevice::create_texture_view")
        })?
    };
    if let Some(label) = descriptor.label.as_deref() {
        raw.setLabel(Some(&NSString::from_str(label)));
    }
    Ok(MetalTextureView { raw })
}

pub(super) fn create_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &SamplerDescriptor,
) -> RhiResult<MetalSampler> {
    let native = MTLSamplerDescriptor::new();
    native.setSAddressMode(address(descriptor.address_u));
    native.setTAddressMode(address(descriptor.address_v));
    native.setRAddressMode(address(descriptor.address_w));
    // Metal defaults this descriptor field to transparent black.  Set it
    // explicitly so every portable ClampToBorder value has the same meaning
    // once the capability fact has admitted the sampler.
    native.setBorderColor(border_color(descriptor.border_color)?);
    native.setMagFilter(filter(descriptor.mag_filter));
    native.setMinFilter(filter(descriptor.min_filter));
    native.setMipFilter(match descriptor.mip_filter {
        FilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => MTLSamplerMipFilter::Linear,
    });
    native.setLodMinClamp(descriptor.lod_min);
    native.setLodMaxClamp(descriptor.lod_max);
    native.setMaxAnisotropy(descriptor.max_anisotropy as usize);
    if let Some(compare) = descriptor.compare {
        native.setCompareFunction(compare_function(compare));
    }
    if let Some(label) = descriptor.label.as_deref() {
        native.setLabel(Some(&NSString::from_str(label)));
    }
    let raw = device
        .newSamplerStateWithDescriptor(&native)
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "Metal sampler creation failed",
            )
            .at("MetalDevice::create_sampler")
        })?;
    Ok(MetalSampler { raw })
}

fn texture_usage(usage: TextureUsage, pixel_format_view: bool) -> MTLTextureUsage {
    let mut result = MTLTextureUsage::Unknown;
    if usage.contains(TextureUsage::SAMPLED) {
        result |= MTLTextureUsage::ShaderRead;
    }
    if usage.contains(TextureUsage::STORAGE) {
        result |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT)
        || usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        result |= MTLTextureUsage::RenderTarget;
    }
    if pixel_format_view {
        result |= MTLTextureUsage::PixelFormatView;
    }
    result
}

fn view_type(value: TextureViewDimension) -> MTLTextureType {
    match value {
        TextureViewDimension::D1 => MTLTextureType::Type1D,
        TextureViewDimension::D2 => MTLTextureType::Type2D,
        TextureViewDimension::D2Array => MTLTextureType::Type2DArray,
        TextureViewDimension::Cube => MTLTextureType::TypeCube,
        TextureViewDimension::CubeArray => MTLTextureType::TypeCubeArray,
        TextureViewDimension::D3 => MTLTextureType::Type3D,
    }
}

fn address(value: AddressMode) -> MTLSamplerAddressMode {
    match value {
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToBorder => MTLSamplerAddressMode::ClampToBorderColor,
    }
}

fn border_color(value: SamplerBorderColor) -> RhiResult<MTLSamplerBorderColor> {
    match value {
        SamplerBorderColor::TransparentBlack => Ok(MTLSamplerBorderColor::TransparentBlack),
        SamplerBorderColor::OpaqueBlack => Ok(MTLSamplerBorderColor::OpaqueBlack),
        SamplerBorderColor::OpaqueWhite => Ok(MTLSamplerBorderColor::OpaqueWhite),
        // Metal has no integer-zero border semantic.  Device::create_sampler
        // rejects it from capability facts first; retain this backend check so
        // a future caller cannot silently lower it as transparent black.
        SamplerBorderColor::Zero => Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "Metal does not lower integer-zero sampler borders",
        )
        .at("MetalDevice::create_sampler")),
    }
}

fn filter(value: FilterMode) -> MTLSamplerMinMagFilter {
    match value {
        FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn compare_function(value: CompareFunction) -> MTLCompareFunction {
    match value {
        CompareFunction::Never => MTLCompareFunction::Never,
        CompareFunction::Less => MTLCompareFunction::Less,
        CompareFunction::Equal => MTLCompareFunction::Equal,
        CompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        CompareFunction::Greater => MTLCompareFunction::Greater,
        CompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        CompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        CompareFunction::Always => MTLCompareFunction::Always,
    }
}
