//! Minimal reflective WebGPU boundary.
//!
//! `web-sys`' generated WebGPU surface evolves with browser bindings.  Keeping
//! the handful of bootstrap calls reflective avoids leaking those generated
//! types through the backend and makes absence an explicit, structured error.

use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};

pub(super) fn browser_gpu() -> Result<JsValue, JsValue> {
    let navigator = property(&js_sys::global().into(), "navigator")?;
    property(&navigator, "gpu")
}

pub(super) fn request_adapter(
    gpu: &JsValue,
    power_preference: Option<&str>,
) -> Result<Promise, JsValue> {
    let options: JsValue = Object::new().into();
    if let Some(preference) = power_preference {
        set(&options, "powerPreference", &JsValue::from_str(preference))?;
    }
    call1(gpu, "requestAdapter", &options).map(Promise::from)
}

/// Request a browser device with the exact optional features and minimum limits
/// selected by the provider.  Keeping descriptor construction at this boundary
/// avoids exposing `GPUFeatureName` or JS dictionaries through the RHI API.
pub(super) fn request_device(
    adapter: &JsValue,
    required_features: &[String],
    required_limits: &[(&str, u64)],
) -> Result<Promise, JsValue> {
    let descriptor: JsValue = Object::new().into();
    if !required_features.is_empty() {
        let features = Array::new();
        for feature in required_features {
            features.push(&JsValue::from_str(feature));
        }
        set(&descriptor, "requiredFeatures", &features.into())?;
    }
    if !required_limits.is_empty() {
        let limits: JsValue = Object::new().into();
        for &(name, value) in required_limits {
            set(&limits, name, &JsValue::from_f64(value as f64))?;
        }
        set(&descriptor, "requiredLimits", &limits)?;
    }
    call1(adapter, "requestDevice", &descriptor).map(Promise::from)
}

/// Reads a finite `GPUSupportedFeatures` set without retaining a browser
/// object.  The caller chooses the finite vocabulary it understands, so a new
/// browser-private feature cannot accidentally become a public capability.
pub(super) fn supported_features(owner: &JsValue, names: &[&str]) -> Vec<String> {
    let Ok(features) = property(owner, "features") else {
        return Vec::new();
    };
    let Ok(has) = property(&features, "has").and_then(|value| value.dyn_into::<Function>()) else {
        return Vec::new();
    };
    names
        .iter()
        .copied()
        .filter(|name| {
            has.call1(&features, &JsValue::from_str(name))
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        })
        .map(str::to_owned)
        .collect()
}

pub(super) fn queue(device: &JsValue) -> Result<JsValue, JsValue> {
    property(device, "queue")
}

pub(super) fn lost(device: &JsValue) -> Result<Promise, JsValue> {
    property(device, "lost").map(Promise::from)
}

pub(super) fn property(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(name)).and_then(|value| {
        (!value.is_null() && !value.is_undefined())
            .then_some(value)
            .ok_or_else(|| JsValue::from_str("missing WebGPU property"))
    })
}

pub(super) fn optional_string(value: &JsValue, name: &str) -> Option<String> {
    Reflect::get(value, &JsValue::from_str(name))
        .ok()?
        .as_string()
}

pub(super) fn message(value: &JsValue) -> String {
    optional_string(value, "message").unwrap_or_else(|| format!("{value:?}"))
}

fn set(value: &JsValue, name: &str, field: &JsValue) -> Result<(), JsValue> {
    Reflect::set(value, &JsValue::from_str(name), field).and_then(|accepted| {
        accepted
            .then_some(())
            .ok_or_else(|| JsValue::from_str("WebGPU descriptor property rejected"))
    })
}

fn call1(receiver: &JsValue, name: &str, argument: &JsValue) -> Result<JsValue, JsValue> {
    let function = property(receiver, name)?.dyn_into::<Function>()?;
    function.call1(receiver, argument)
}
