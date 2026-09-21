//! WebGPU resource objects.
//!
//! Native WebGPU objects stay in the owner-thread registry.  The portable
//! handle therefore carries this small registration instead of a `JsValue`.

use std::any::Any;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use std::task::{Context, Poll};
use wasm_bindgen::{JsCast, JsValue};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::query::QuerySetDescriptor;
use crate::api::resource::backend::{
    BufferBackend, QuerySetBackend, SamplerBackend, TextureBackend, TextureViewBackend,
};
use crate::api::resource::{
    BufferDescriptor, SamplerDescriptor, Texture, TextureDescriptor, TextureViewDescriptor,
};

use super::registry::{WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::{js, registry, translate};

macro_rules! registered_object {
    ($name:ident, $trait:ident) => {
        pub(crate) struct $name {
            driver: WebGpuDriver,
            registration: WebGpuRegistration,
            object: WebGpuObjectId,
        }
        impl $name {
            pub(crate) const fn registration(&self) -> WebGpuRegistration {
                self.registration
            }
            pub(crate) const fn object(&self) -> WebGpuObjectId {
                self.object
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                let _ = registry::remove_object(self.registration, self.object);
            }
        }
        impl $trait for $name {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
    };
}
registered_object!(WebGpuBuffer, BufferBackend);
registered_object!(WebGpuTexture, TextureBackend);
registered_object!(WebGpuTextureView, TextureViewBackend);
registered_object!(WebGpuSampler, SamplerBackend);
registered_object!(WebGpuQuerySet, QuerySetBackend);

fn error(kind: RhiErrorKind, where_: &'static str, value: impl Into<String>) -> RhiError {
    RhiError::new(kind, value.into()).at(where_)
}
fn field(object: &Object, name: &str, value: JsValue) -> RhiResult<()> {
    Reflect::set(object, &JsValue::from_str(name), &value)
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "WebGPU descriptor",
                js::message(&e),
            )
        })?
        .then_some(())
        .ok_or_else(|| {
            error(
                RhiErrorKind::BackendFailure,
                "WebGPU descriptor",
                "browser rejected descriptor field",
            )
        })
}
fn call(device: &JsValue, method: &'static str, descriptor: &Object) -> RhiResult<JsValue> {
    let function = js::property(device, method)
        .map_err(|e| error(RhiErrorKind::Unsupported, method, js::message(&e)))?
        .dyn_into::<Function>()
        .map_err(|e| error(RhiErrorKind::Unsupported, method, js::message(&e)))?;
    function
        .call1(device, descriptor)
        .map_err(|e| error(RhiErrorKind::BackendFailure, method, js::message(&e)))
}
fn device(registration: WebGpuRegistration) -> RhiResult<JsValue> {
    registry::with_device_handles(registration, |handles| handles.device.clone()).ok_or_else(|| {
        error(
            RhiErrorKind::DeviceLost,
            "WebGPU resource",
            "WebGPU device registration is no longer active",
        )
    })
}
fn retain(
    registration: WebGpuRegistration,
    value: JsValue,
    where_: &'static str,
) -> RhiResult<WebGpuObjectId> {
    registry::insert_object(registration, value).ok_or_else(|| {
        error(
            RhiErrorKind::DeviceLost,
            where_,
            "WebGPU device registration is no longer active",
        )
    })
}
fn unsupported(where_: &'static str, why: translate::Unsupported) -> RhiError {
    error(RhiErrorKind::Unsupported, where_, why.what)
}

pub(crate) fn create_buffer(
    driver: &WebGpuDriver,
    descriptor: &BufferDescriptor,
) -> RhiResult<WebGpuBuffer> {
    let registration = driver.registration();
    let d = Object::new();
    field(&d, "size", JsValue::from_f64(descriptor.size as f64))?;
    field(
        &d,
        "usage",
        JsValue::from_f64(
            translate::buffer_usage(descriptor.usage)
                .map_err(|e| unsupported("WebGpuDevice::create_buffer", e))? as f64,
        ),
    )?;
    if let Some(label) = descriptor.label.as_deref() {
        field(&d, "label", JsValue::from_str(label))?;
    }
    let object = retain(
        registration,
        call(&device(registration)?, "createBuffer", &d)?,
        "WebGpuDevice::create_buffer",
    )?;
    Ok(WebGpuBuffer {
        driver: driver.clone(),
        registration,
        object,
    })
}

pub(crate) fn create_texture(
    driver: &WebGpuDriver,
    descriptor: &TextureDescriptor,
) -> RhiResult<WebGpuTexture> {
    let registration = driver.registration();
    let d = Object::new();
    let size = Object::new();
    field(
        &size,
        "width",
        JsValue::from_f64(descriptor.extent.width as f64),
    )?;
    field(
        &size,
        "height",
        JsValue::from_f64(descriptor.extent.height as f64),
    )?;
    field(
        &size,
        "depthOrArrayLayers",
        JsValue::from_f64(match descriptor.dimension {
            crate::api::resource::TextureDimension::D3 => descriptor.extent.depth,
            _ => descriptor.array_layers,
        } as f64),
    )?;
    field(&d, "size", size.into())?;
    field(
        &d,
        "dimension",
        JsValue::from_str(
            translate::texture_dimension(descriptor.dimension)
                .map_err(|e| unsupported("WebGpuDevice::create_texture", e))?,
        ),
    )?;
    field(
        &d,
        "format",
        JsValue::from_str(
            translate::texture_format(descriptor.format)
                .map_err(|e| unsupported("WebGpuDevice::create_texture", e))?,
        ),
    )?;
    field(
        &d,
        "usage",
        JsValue::from_f64(translate::texture_usage(descriptor.usage) as f64),
    )?;
    field(
        &d,
        "mipLevelCount",
        JsValue::from_f64(descriptor.mip_levels as f64),
    )?;
    field(
        &d,
        "sampleCount",
        JsValue::from_f64(descriptor.sample_count as f64),
    )?;
    if !descriptor.view_formats.is_empty() {
        let formats = js_sys::Array::new();
        for &format in &descriptor.view_formats {
            formats.push(&JsValue::from_str(
                translate::texture_format(format)
                    .map_err(|e| unsupported("WebGpuDevice::create_texture", e))?,
            ));
        }
        field(&d, "viewFormats", formats.into())?;
    }
    if let Some(label) = descriptor.label.as_deref() {
        field(&d, "label", JsValue::from_str(label))?;
    }
    let object = retain(
        registration,
        call(&device(registration)?, "createTexture", &d)?,
        "WebGpuDevice::create_texture",
    )?;
    Ok(WebGpuTexture {
        driver: driver.clone(),
        registration,
        object,
    })
}

pub(crate) fn create_texture_view(
    driver: &WebGpuDriver,
    texture: &Texture,
    descriptor: &TextureViewDescriptor,
) -> RhiResult<WebGpuTextureView> {
    let registration = driver.registration();
    let native = texture
        .native()
        .as_any()
        .downcast_ref::<WebGpuTexture>()
        .ok_or_else(|| {
            error(
                RhiErrorKind::WrongDevice,
                "WebGpuDevice::create_texture_view",
                "texture is not owned by this WebGPU backend",
            )
        })?;
    if native.registration != registration {
        return Err(error(
            RhiErrorKind::WrongDevice,
            "WebGpuDevice::create_texture_view",
            "texture belongs to another WebGPU device",
        ));
    }
    let texture =
        registry::with_object(registration, native.object, Clone::clone).ok_or_else(|| {
            error(
                RhiErrorKind::DeviceLost,
                "WebGpuDevice::create_texture_view",
                "texture registration was retired",
            )
        })?;
    let d = Object::new();
    field(
        &d,
        "dimension",
        JsValue::from_str(
            translate::texture_view_dimension(descriptor.dimension)
                .map_err(|e| unsupported("WebGpuDevice::create_texture_view", e))?,
        ),
    )?;
    field(
        &d,
        "aspect",
        JsValue::from_str(
            translate::texture_aspect(descriptor.aspects)
                .map_err(|e| unsupported("WebGpuDevice::create_texture_view", e))?,
        ),
    )?;
    field(
        &d,
        "baseMipLevel",
        JsValue::from_f64(descriptor.base_mip as f64),
    )?;
    field(
        &d,
        "mipLevelCount",
        JsValue::from_f64(descriptor.mip_count as f64),
    )?;
    field(
        &d,
        "baseArrayLayer",
        JsValue::from_f64(descriptor.base_layer as f64),
    )?;
    field(
        &d,
        "arrayLayerCount",
        JsValue::from_f64(descriptor.layer_count as f64),
    )?;
    if let Some(format) = descriptor.format {
        field(
            &d,
            "format",
            JsValue::from_str(
                translate::texture_format(format)
                    .map_err(|e| unsupported("WebGpuDevice::create_texture_view", e))?,
            ),
        )?;
    }
    if let Some(usage) = descriptor.usage {
        field(
            &d,
            "usage",
            JsValue::from_f64(translate::texture_usage(usage) as f64),
        )?;
    }
    if let Some(label) = descriptor.label.as_deref() {
        field(&d, "label", JsValue::from_str(label))?;
    }
    let function = js::property(&texture, "createView")
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUTexture.createView",
                js::message(&e),
            )
        })?
        .dyn_into::<Function>()
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUTexture.createView",
                js::message(&e),
            )
        })?;
    let object = retain(
        registration,
        function.call1(&texture, &d).map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUTexture.createView",
                js::message(&e),
            )
        })?,
        "WebGpuDevice::create_texture_view",
    )?;
    Ok(WebGpuTextureView {
        driver: driver.clone(),
        registration,
        object,
    })
}

pub(crate) fn create_sampler(
    driver: &WebGpuDriver,
    descriptor: &SamplerDescriptor,
) -> RhiResult<WebGpuSampler> {
    let registration = driver.registration();
    let d = Object::new();
    field(
        &d,
        "addressModeU",
        JsValue::from_str(
            translate::address_mode(descriptor.address_u)
                .map_err(|e| unsupported("WebGpuDevice::create_sampler", e))?,
        ),
    )?;
    field(
        &d,
        "addressModeV",
        JsValue::from_str(
            translate::address_mode(descriptor.address_v)
                .map_err(|e| unsupported("WebGpuDevice::create_sampler", e))?,
        ),
    )?;
    field(
        &d,
        "addressModeW",
        JsValue::from_str(
            translate::address_mode(descriptor.address_w)
                .map_err(|e| unsupported("WebGpuDevice::create_sampler", e))?,
        ),
    )?;
    field(
        &d,
        "magFilter",
        JsValue::from_str(translate::filter_mode(descriptor.mag_filter)),
    )?;
    field(
        &d,
        "minFilter",
        JsValue::from_str(translate::filter_mode(descriptor.min_filter)),
    )?;
    field(
        &d,
        "mipmapFilter",
        JsValue::from_str(translate::filter_mode(descriptor.mip_filter)),
    )?;
    field(
        &d,
        "lodMinClamp",
        JsValue::from_f64(descriptor.lod_min as f64),
    )?;
    field(
        &d,
        "lodMaxClamp",
        JsValue::from_f64(descriptor.lod_max as f64),
    )?;
    if let Some(compare) = descriptor.compare {
        field(
            &d,
            "compare",
            JsValue::from_str(
                translate::compare_function(compare)
                    .map_err(|e| unsupported("WebGpuDevice::create_sampler", e))?,
            ),
        )?;
    }
    if descriptor.max_anisotropy > 1 {
        // Public RHI validation currently makes this branch unreachable for
        // WebGPU: capability discovery intentionally does not publish an
        // unqueryable browser clamp as MaxSamplerAnisotropy. Retain exact
        // descriptor lowering so a future queryable capability seam does not
        // need a second sampler representation. WebGPU additionally requires
        // mag/min/mipmap filtering all to be Linear for this value; that must
        // be validated before this native call if the capability is published.
        field(
            &d,
            "maxAnisotropy",
            JsValue::from_f64(descriptor.max_anisotropy as f64),
        )?;
    }
    if let Some(label) = descriptor.label.as_deref() {
        field(&d, "label", JsValue::from_str(label))?;
    }
    let object = retain(
        registration,
        call(&device(registration)?, "createSampler", &d)?,
        "WebGpuDevice::create_sampler",
    )?;
    Ok(WebGpuSampler {
        driver: driver.clone(),
        registration,
        object,
    })
}

pub(crate) fn create_query_set(
    driver: &WebGpuDriver,
    descriptor: &QuerySetDescriptor,
) -> RhiResult<WebGpuQuerySet> {
    let registration = driver.registration();
    let d = Object::new();
    field(
        &d,
        "type",
        JsValue::from_str(
            translate::query_type(descriptor.ty)
                .map_err(|e| unsupported("WebGpuDevice::create_query_set", e))?,
        ),
    )?;
    field(&d, "count", JsValue::from_f64(descriptor.count as f64))?;
    if let Some(label) = descriptor.label.as_deref() {
        field(&d, "label", JsValue::from_str(label))?;
    }
    let object = retain(
        registration,
        call(&device(registration)?, "createQuerySet", &d)?,
        "WebGpuDevice::create_query_set",
    )?;
    Ok(WebGpuQuerySet {
        driver: driver.clone(),
        registration,
        object,
    })
}

pub(crate) fn map_buffer(
    driver: &WebGpuDriver,
    buffer: &crate::api::resource::Buffer,
    mode: crate::api::resource::MapMode,
    range: crate::api::resource::BufferRange,
) -> RhiResult<Box<dyn crate::api::resource::backend::MappingRequestBackend>> {
    let registration = driver.registration();
    let native = buffer
        .native()
        .as_any()
        .downcast_ref::<WebGpuBuffer>()
        .ok_or_else(|| {
            error(
                RhiErrorKind::WrongDevice,
                "WebGpuDevice::map_buffer",
                "buffer belongs to another backend",
            )
        })?;
    if native.registration != registration {
        return Err(error(
            RhiErrorKind::WrongDevice,
            "WebGpuDevice::map_buffer",
            "buffer belongs to another device",
        ));
    }
    let value =
        registry::with_object(registration, native.object, Clone::clone).ok_or_else(|| {
            error(
                RhiErrorKind::DeviceLost,
                "WebGpuDevice::map_buffer",
                "buffer registration was retired",
            )
        })?;
    let call = js::property(&value, "mapAsync")
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUBuffer.mapAsync",
                js::message(&e),
            )
        })?
        .dyn_into::<Function>()
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUBuffer.mapAsync",
                js::message(&e),
            )
        })?;
    let promise = call
        .call3(
            &value,
            &JsValue::from_f64(match mode {
                crate::api::resource::MapMode::Read => 1.0,
                crate::api::resource::MapMode::Write => 2.0,
            }),
            &JsValue::from_f64(range.offset as f64),
            &JsValue::from_f64(range.size as f64),
        )
        .map_err(|e| {
            error(
                RhiErrorKind::BackendFailure,
                "GPUBuffer.mapAsync",
                js::message(&e),
            )
        })?;
    Ok(Box::new(WebGpuMapRequest {
        driver: driver.clone(),
        buffer: value,
        mode,
        range,
        promise: registry::start_device_promise(registration, Promise::from(promise)),
    }))
}

struct WebGpuMapRequest {
    driver: WebGpuDriver,
    buffer: JsValue,
    mode: crate::api::resource::MapMode,
    range: crate::api::resource::BufferRange,
    promise: registry::WebGpuRequestId,
}
impl crate::api::resource::backend::MappingRequestBackend for WebGpuMapRequest {
    fn poll(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<RhiResult<Box<dyn crate::api::resource::backend::MappedBufferBackend>>> {
        match registry::poll_promise(self.promise, context.waker()) {
            registry::PromisePoll::Pending => Poll::Pending,
            registry::PromisePoll::Failed(message) => {
                let kind = if matches!(
                    registry::device_status(self.driver.registration()),
                    Some(crate::api::platform::DeviceStatus::Lost)
                ) || message.starts_with("DeviceLost:")
                {
                    RhiErrorKind::DeviceLost
                } else {
                    RhiErrorKind::BackendFailure
                };
                Poll::Ready(Err(error(kind, "GPUBuffer.mapAsync", message)))
            }
            registry::PromisePoll::Ready(_) => {
                let get = match js::property(&self.buffer, "getMappedRange")
                    .and_then(|x| x.dyn_into::<Function>())
                {
                    Ok(value) => value,
                    Err(value) => {
                        return Poll::Ready(Err(error(
                            RhiErrorKind::BackendFailure,
                            "GPUBuffer.getMappedRange",
                            js::message(&value),
                        )));
                    }
                };
                let mapped = match get.call2(
                    &self.buffer,
                    &JsValue::from_f64(self.range.offset as f64),
                    &JsValue::from_f64(self.range.size as f64),
                ) {
                    Ok(value) => value,
                    Err(value) => {
                        return Poll::Ready(Err(error(
                            RhiErrorKind::BackendFailure,
                            "GPUBuffer.getMappedRange",
                            js::message(&value),
                        )));
                    }
                };
                let bytes = Uint8Array::new(&mapped).to_vec();
                Poll::Ready(Ok(Box::new(WebGpuMappedBuffer {
                    driver: self.driver.clone(),
                    buffer: self.buffer.clone(),
                    mode: self.mode,
                    offset: self.range.offset,
                    bytes,
                    unmapped: false,
                })))
            }
        }
    }
}
struct WebGpuMappedBuffer {
    driver: WebGpuDriver,
    buffer: JsValue,
    mode: crate::api::resource::MapMode,
    offset: u64,
    bytes: Vec<u8>,
    unmapped: bool,
}
impl WebGpuMappedBuffer {
    fn unmap(&mut self) {
        if !self.unmapped {
            if let Ok(value) =
                js::property(&self.buffer, "unmap").and_then(|x| x.dyn_into::<Function>())
            {
                let _ = value.call0(&self.buffer);
            }
            self.unmapped = true;
        }
    }
}
impl Drop for WebGpuMappedBuffer {
    fn drop(&mut self) {
        self.unmap();
    }
}
impl crate::api::resource::backend::MappedBufferBackend for WebGpuMappedBuffer {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        matches!(self.mode, crate::api::resource::MapMode::Write)
            .then_some(self.bytes.as_mut_slice())
    }
    fn flush(&mut self) -> RhiResult<()> {
        if matches!(self.mode, crate::api::resource::MapMode::Write) {
            let queue =
                registry::with_device_handles(self.driver.registration(), |h| h.queue.clone())
                    .ok_or_else(|| {
                        error(
                            RhiErrorKind::DeviceLost,
                            "WebGPU mapped buffer",
                            "device registration was retired",
                        )
                    })?;
            let write = js::property(&queue, "writeBuffer")
                .map_err(|e| {
                    error(
                        RhiErrorKind::BackendFailure,
                        "GPUQueue.writeBuffer",
                        js::message(&e),
                    )
                })?
                .dyn_into::<Function>()
                .map_err(|e| {
                    error(
                        RhiErrorKind::BackendFailure,
                        "GPUQueue.writeBuffer",
                        js::message(&e),
                    )
                })?;
            let view = Uint8Array::from(self.bytes.as_slice());
            self.unmap();
            write
                .call3(
                    &queue,
                    &self.buffer,
                    &JsValue::from_f64(self.offset as f64),
                    &view,
                )
                .map_err(|e| {
                    error(
                        RhiErrorKind::BackendFailure,
                        "GPUQueue.writeBuffer",
                        js::message(&e),
                    )
                })?;
        }
        Ok(())
    }
    fn invalidate(&mut self) -> RhiResult<()> {
        Ok(())
    }
}
