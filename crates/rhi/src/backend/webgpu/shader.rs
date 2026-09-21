//! WebGPU WGSL shader modules.

use js_sys::{Function, Object, Reflect};
use std::any::Any;
use wasm_bindgen::{JsCast, JsValue};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::shader::backend::ShaderModuleBackend;
use crate::api::shader::{ShaderArtifact, ShaderCode};

use super::registry::{WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::{js, registry};

pub(crate) struct WebGpuShaderModule {
    driver: WebGpuDriver,
    registration: WebGpuRegistration,
    object: WebGpuObjectId,
}
impl WebGpuShaderModule {
    pub(crate) const fn registration(&self) -> WebGpuRegistration {
        self.registration
    }
    pub(crate) const fn object(&self) -> WebGpuObjectId {
        self.object
    }
}
impl Drop for WebGpuShaderModule {
    fn drop(&mut self) {
        let _ = registry::remove_object(self.registration, self.object);
    }
}
impl ShaderModuleBackend for WebGpuShaderModule {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn create_shader(
    driver: &WebGpuDriver,
    artifact: &ShaderArtifact,
) -> RhiResult<WebGpuShaderModule> {
    let registration = driver.registration();
    let ShaderCode::Wgsl(code) = &artifact.code else {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "WebGPU accepts WGSL shader artifacts only",
        )
        .at("WebGpuDevice::create_shader"));
    };
    let descriptor = Object::new();
    Reflect::set(
        &descriptor,
        &JsValue::from_str("code"),
        &JsValue::from_str(code),
    )
    .map_err(|e| {
        RhiError::new(RhiErrorKind::BackendFailure, js::message(&e))
            .at("WebGpuDevice::create_shader")
    })?;
    if let Some(label) = artifact.label.as_deref() {
        Reflect::set(
            &descriptor,
            &JsValue::from_str("label"),
            &JsValue::from_str(label),
        )
        .map_err(|e| {
            RhiError::new(RhiErrorKind::BackendFailure, js::message(&e))
                .at("WebGpuDevice::create_shader")
        })?;
    }
    let device = registry::with_device_handles(registration, |handles| handles.device.clone())
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::DeviceLost,
                "WebGPU device registration is no longer active",
            )
            .at("WebGpuDevice::create_shader")
        })?;
    let function = js::property(&device, "createShaderModule")
        .map_err(|e| {
            RhiError::new(RhiErrorKind::Unsupported, js::message(&e))
                .at("WebGpuDevice::create_shader")
        })?
        .dyn_into::<Function>()
        .map_err(|e| {
            RhiError::new(RhiErrorKind::Unsupported, js::message(&e))
                .at("WebGpuDevice::create_shader")
        })?;
    let value = function.call1(&device, &descriptor).map_err(|e| {
        RhiError::new(RhiErrorKind::BackendFailure, js::message(&e))
            .at("WebGpuDevice::create_shader")
    })?;
    let object = registry::insert_object(registration, value).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::DeviceLost,
            "WebGPU device registration is no longer active",
        )
        .at("WebGpuDevice::create_shader")
    })?;
    Ok(WebGpuShaderModule {
        driver: driver.clone(),
        registration,
        object,
    })
}
