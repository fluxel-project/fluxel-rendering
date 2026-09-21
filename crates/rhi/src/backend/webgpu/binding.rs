//! WebGPU bind-group layout and packet lowering.
//!
//! WebGPU compares bind-group layouts by object identity.  The RHI already
//! interns an exact same-device compatibility token, so this module retains one
//! JS layout for each `(device registration, compatibility id)` pair.

use js_sys::{Array, Function, Object, Reflect};
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, JsValue};

use crate::api::binding::backend::BindGroupBackend;
use crate::api::binding::{
    BindGroupDescriptor, BindGroupLayout, BindingCount, BindingKind, BindingResource,
    BufferBindingAccess, SamplerKind, StorageAccess, TextureSampleType,
};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::resource::TextureView;
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::ShaderStages;

use super::registry::{WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::{js, registry, resource, translate};

thread_local! { static LAYOUTS: RefCell<BTreeMap<(WebGpuRegistration, u64), JsValue>> = const { RefCell::new(BTreeMap::new()) }; }

pub(crate) struct WebGpuBindGroup {
    driver: WebGpuDriver,
    registration: WebGpuRegistration,
    object: WebGpuObjectId,
}
impl WebGpuBindGroup {
    pub(crate) const fn registration(&self) -> WebGpuRegistration {
        self.registration
    }
    pub(crate) const fn object(&self) -> WebGpuObjectId {
        self.object
    }
}
impl Drop for WebGpuBindGroup {
    fn drop(&mut self) {
        let _ = registry::remove_object(self.registration, self.object);
    }
}
impl BindGroupBackend for WebGpuBindGroup {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn fail(kind: RhiErrorKind, at: &'static str, message: impl Into<String>) -> RhiError {
    RhiError::new(kind, message.into()).at(at)
}
fn set(object: &Object, key: &str, value: JsValue) -> RhiResult<()> {
    Reflect::set(object, &JsValue::from_str(key), &value)
        .map_err(|e| {
            fail(
                RhiErrorKind::BackendFailure,
                "WebGPU descriptor",
                js::message(&e),
            )
        })?
        .then_some(())
        .ok_or_else(|| {
            fail(
                RhiErrorKind::BackendFailure,
                "WebGPU descriptor",
                "browser rejected descriptor field",
            )
        })
}
fn device(registration: WebGpuRegistration) -> RhiResult<JsValue> {
    registry::with_device_handles(registration, |h| h.device.clone()).ok_or_else(|| {
        fail(
            RhiErrorKind::DeviceLost,
            "WebGPU",
            "device registration was retired",
        )
    })
}
fn invoke(receiver: &JsValue, method: &'static str, descriptor: &Object) -> RhiResult<JsValue> {
    js::property(receiver, method)
        .map_err(|e| fail(RhiErrorKind::Unsupported, method, js::message(&e)))?
        .dyn_into::<Function>()
        .map_err(|e| fail(RhiErrorKind::Unsupported, method, js::message(&e)))?
        .call1(receiver, descriptor)
        .map_err(|e| fail(RhiErrorKind::BackendFailure, method, js::message(&e)))
}

pub(crate) fn layout_for(driver: &WebGpuDriver, layout: &BindGroupLayout) -> RhiResult<JsValue> {
    let registration = driver.registration();
    let key = (registration, layout.compatibility_id().get());
    if let Some(value) = LAYOUTS.with(|layouts| layouts.borrow().get(&key).cloned()) {
        return Ok(value);
    }
    let descriptor = Object::new();
    let entries = Array::new();
    for slot in &layout.descriptor().entries {
        entries.push(&layout_entry(slot)?);
    }
    set(&descriptor, "entries", entries.into())?;
    if let Some(label) = layout.descriptor().label.as_deref() {
        set(&descriptor, "label", JsValue::from_str(label))?;
    }
    let created = invoke(&device(registration)?, "createBindGroupLayout", &descriptor)?;
    LAYOUTS.with(|layouts| {
        layouts.borrow_mut().insert(key, created.clone());
    });
    Ok(created)
}

fn layout_entry(slot: &crate::api::binding::BindingSlot) -> RhiResult<JsValue> {
    let entry = Object::new();
    set(&entry, "binding", JsValue::from_f64(slot.slot.get() as f64))?;
    set(
        &entry,
        "visibility",
        JsValue::from_f64(stages(slot.visibility)? as f64),
    )?;
    if !matches!(slot.count, BindingCount::One) {
        let count = match slot.count {
            BindingCount::Fixed(n) => n,
            BindingCount::RuntimeSized => {
                return Err(fail(
                    RhiErrorKind::Unsupported,
                    "WebGPU BindGroupLayout",
                    "runtime-sized binding arrays",
                ));
            }
            _ => unreachable!(),
        };
        set(&entry, "count", JsValue::from_f64(count as f64))?;
    }
    match &slot.kind {
        BindingKind::UniformBuffer { min_size } => {
            let b = Object::new();
            set(&b, "type", JsValue::from_str("uniform"))?;
            set(
                &b,
                "hasDynamicOffset",
                JsValue::from_bool(slot.dynamic_offset),
            )?;
            set(&b, "minBindingSize", JsValue::from_f64(*min_size as f64))?;
            set(&entry, "buffer", b.into())?;
        }
        BindingKind::StorageBuffer { access, min_size } => {
            let b = Object::new();
            set(
                &b,
                "type",
                JsValue::from_str(match access {
                    BufferBindingAccess::ReadOnly => "read-only-storage",
                    BufferBindingAccess::ReadWrite => "storage",
                }),
            )?;
            set(
                &b,
                "hasDynamicOffset",
                JsValue::from_bool(slot.dynamic_offset),
            )?;
            set(&b, "minBindingSize", JsValue::from_f64(*min_size as f64))?;
            set(&entry, "buffer", b.into())?;
        }
        BindingKind::SampledTexture {
            dimension,
            sample_type,
            multisampled,
        } => {
            let t = Object::new();
            set(
                &t,
                "viewDimension",
                JsValue::from_str(view_dimension(*dimension)?),
            )?;
            set(
                &t,
                "sampleType",
                JsValue::from_str(match sample_type {
                    TextureSampleType::Float => "float",
                    TextureSampleType::UnfilterableFloat => "unfilterable-float",
                    TextureSampleType::Sint => "sint",
                    TextureSampleType::Uint => "uint",
                    TextureSampleType::Depth => "depth",
                }),
            )?;
            set(&t, "multisampled", JsValue::from_bool(*multisampled))?;
            set(&entry, "texture", t.into())?;
        }
        BindingKind::StorageTexture {
            dimension,
            format,
            access,
        } => {
            let t = Object::new();
            set(
                &t,
                "viewDimension",
                JsValue::from_str(view_dimension(*dimension)?),
            )?;
            set(
                &t,
                "format",
                JsValue::from_str(translate::texture_format(*format).map_err(|e| {
                    fail(RhiErrorKind::Unsupported, "WebGPU BindGroupLayout", e.what)
                })?),
            )?;
            set(
                &t,
                "access",
                JsValue::from_str(match access {
                    StorageAccess::WriteOnly => "write-only",
                    StorageAccess::ReadOnly => "read-only",
                    StorageAccess::ReadWrite => "read-write",
                }),
            )?;
            set(&entry, "storageTexture", t.into())?;
        }
        BindingKind::Sampler { kind } => {
            let s = Object::new();
            set(
                &s,
                "type",
                JsValue::from_str(match kind {
                    SamplerKind::Filtering => "filtering",
                    SamplerKind::NonFiltering => "non-filtering",
                    SamplerKind::Comparison => "comparison",
                }),
            )?;
            set(&entry, "sampler", s.into())?;
        }
        BindingKind::AccelerationStructure | BindingKind::ExternalTexture => {
            return Err(fail(
                RhiErrorKind::Unsupported,
                "WebGPU BindGroupLayout",
                "binding kind is not available in baseline WebGPU",
            ));
        }
    }
    Ok(entry.into())
}
fn stages(value: ShaderStages) -> RhiResult<u32> {
    let mut result = 0;
    if value.contains(ShaderStages::VERTEX) {
        result |= 1
    };
    if value.contains(ShaderStages::FRAGMENT) {
        result |= 2
    };
    if value.contains(ShaderStages::COMPUTE) {
        result |= 4
    };
    if result == 0 || result != stage_bits(value) {
        return Err(fail(
            RhiErrorKind::Unsupported,
            "WebGPU BindGroupLayout",
            "non-WebGPU shader stage visibility",
        ));
    }
    Ok(result)
}
fn stage_bits(v: ShaderStages) -> u32 {
    let mut n = 0;
    if v.contains(ShaderStages::VERTEX) {
        n |= 1
    };
    if v.contains(ShaderStages::FRAGMENT) {
        n |= 2
    };
    if v.contains(ShaderStages::COMPUTE) {
        n |= 4
    };
    n
}
fn view_dimension(value: TextureViewDimension) -> RhiResult<&'static str> {
    translate::texture_view_dimension(value)
        .map_err(|e| fail(RhiErrorKind::Unsupported, "WebGPU BindGroupLayout", e.what))
}

pub(crate) fn create_bind_group(
    driver: &WebGpuDriver,
    descriptor: &BindGroupDescriptor,
) -> RhiResult<WebGpuBindGroup> {
    let registration = driver.registration();
    let d = Object::new();
    set(&d, "layout", layout_for(driver, &descriptor.layout)?)?;
    let entries = Array::new();
    for entry in &descriptor.entries {
        append_resource_entries(&entries, registration, entry.slot.get(), &entry.resource)?;
    }
    set(&d, "entries", entries.into())?;
    if let Some(label) = descriptor.label.as_deref() {
        set(&d, "label", JsValue::from_str(label))?;
    }
    let object = registry::insert_object(
        registration,
        invoke(&device(registration)?, "createBindGroup", &d)?,
    )
    .ok_or_else(|| {
        fail(
            RhiErrorKind::DeviceLost,
            "WebGpuDevice::create_bind_group",
            "device registration was retired",
        )
    })?;
    Ok(WebGpuBindGroup {
        driver: driver.clone(),
        registration,
        object,
    })
}
fn append_resource_entries(
    out: &Array,
    reg: WebGpuRegistration,
    binding: u32,
    resource_value: &BindingResource,
) -> RhiResult<()> {
    match resource_value {
        BindingResource::Buffer(v) => {
            out.push(&buffer_entry(reg, binding, v)?);
        }
        BindingResource::Texture(v) => {
            out.push(&view_entry(reg, binding, v)?);
        }
        BindingResource::Sampler(v) => {
            out.push(&sampler_entry(reg, binding, v)?);
        }
        BindingResource::BufferArray(v) => {
            let resources = Array::new();
            for x in v {
                resources.push(&entry_resource(&buffer_entry(reg, binding, x)?)?);
            }
            out.push(&array_entry(binding, resources)?);
        }
        BindingResource::TextureArray(v) => {
            let resources = Array::new();
            for x in v {
                resources.push(&entry_resource(&view_entry(reg, binding, x)?)?);
            }
            out.push(&array_entry(binding, resources)?);
        }
        BindingResource::SamplerArray(v) => {
            let resources = Array::new();
            for x in v {
                resources.push(&entry_resource(&sampler_entry(reg, binding, x)?)?);
            }
            out.push(&array_entry(binding, resources)?);
        }
        BindingResource::AccelerationStructure(_)
        | BindingResource::ExternalTexture(_)
        | BindingResource::AccelerationStructureArray(_) => {
            return Err(fail(
                RhiErrorKind::Unsupported,
                "WebGpuDevice::create_bind_group",
                "binding resource unavailable in baseline WebGPU",
            ));
        }
    };
    Ok(())
}
fn entry_resource(entry: &JsValue) -> RhiResult<JsValue> {
    Reflect::get(entry, &JsValue::from_str("resource")).map_err(|error| {
        fail(
            RhiErrorKind::BackendFailure,
            "WebGPU bind group",
            js::message(&error),
        )
    })
}
fn array_entry(binding: u32, resources: Array) -> RhiResult<JsValue> {
    let entry = Object::new();
    set(&entry, "binding", JsValue::from_f64(binding as f64))?;
    set(&entry, "resource", resources.into())?;
    Ok(entry.into())
}
fn buffer_entry(
    reg: WebGpuRegistration,
    binding: u32,
    value: &crate::api::resource::BufferBinding,
) -> RhiResult<JsValue> {
    let n = value
        .buffer
        .native()
        .as_any()
        .downcast_ref::<resource::WebGpuBuffer>()
        .ok_or_else(|| {
            fail(
                RhiErrorKind::WrongDevice,
                "WebGPU bind group",
                "buffer has another backend",
            )
        })?;
    if n.registration() != reg {
        return Err(fail(
            RhiErrorKind::WrongDevice,
            "WebGPU bind group",
            "buffer has another device",
        ));
    };
    let b = registry::with_object(reg, n.object(), Clone::clone).ok_or_else(|| {
        fail(
            RhiErrorKind::DeviceLost,
            "WebGPU bind group",
            "buffer retired",
        )
    })?;
    let e = Object::new();
    let r = Object::new();
    set(&r, "buffer", b)?;
    set(&r, "offset", JsValue::from_f64(value.range.offset as f64))?;
    set(&r, "size", JsValue::from_f64(value.range.size as f64))?;
    set(&e, "binding", JsValue::from_f64(binding as f64))?;
    set(&e, "resource", r.into())?;
    Ok(e.into())
}
fn view_entry(reg: WebGpuRegistration, binding: u32, value: &TextureView) -> RhiResult<JsValue> {
    let n = value
        .native()
        .as_any()
        .downcast_ref::<resource::WebGpuTextureView>()
        .ok_or_else(|| {
            fail(
                RhiErrorKind::WrongDevice,
                "WebGPU bind group",
                "view has another backend",
            )
        })?;
    if n.registration() != reg {
        return Err(fail(
            RhiErrorKind::WrongDevice,
            "WebGPU bind group",
            "view has another device",
        ));
    };
    let e = Object::new();
    set(&e, "binding", JsValue::from_f64(binding as f64))?;
    set(
        &e,
        "resource",
        registry::with_object(reg, n.object(), Clone::clone).ok_or_else(|| {
            fail(
                RhiErrorKind::DeviceLost,
                "WebGPU bind group",
                "view retired",
            )
        })?,
    )?;
    Ok(e.into())
}
fn sampler_entry(
    reg: WebGpuRegistration,
    binding: u32,
    value: &crate::api::resource::Sampler,
) -> RhiResult<JsValue> {
    let n = value
        .native()
        .as_any()
        .downcast_ref::<resource::WebGpuSampler>()
        .ok_or_else(|| {
            fail(
                RhiErrorKind::WrongDevice,
                "WebGPU bind group",
                "sampler has another backend",
            )
        })?;
    if n.registration() != reg {
        return Err(fail(
            RhiErrorKind::WrongDevice,
            "WebGPU bind group",
            "sampler has another device",
        ));
    };
    let e = Object::new();
    set(&e, "binding", JsValue::from_f64(binding as f64))?;
    set(
        &e,
        "resource",
        registry::with_object(reg, n.object(), Clone::clone).ok_or_else(|| {
            fail(
                RhiErrorKind::DeviceLost,
                "WebGPU bind group",
                "sampler retired",
            )
        })?,
    )?;
    Ok(e.into())
}
