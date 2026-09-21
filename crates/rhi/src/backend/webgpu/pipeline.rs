//! WebGPU compute and raster pipeline objects.

use super::registry::{WebGpuDriver, WebGpuObjectId, WebGpuRegistration};
use super::{binding, js, registry, shader, translate};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::backend::{ComputePipelineBackend, RasterPipelineBackend};
use crate::api::pipeline::{ComputePipelineDescriptor, RasterPipelineDescriptor};
use crate::api::shader::ShaderModule;
use js_sys::{Array, Function, Object, Reflect};
use std::any::Any;
use wasm_bindgen::{JsCast, JsValue};

macro_rules! pipeline {
    ($name:ident,$trait:ident) => {
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
pipeline!(WebGpuComputePipeline, ComputePipelineBackend);
pipeline!(WebGpuRasterPipeline, RasterPipelineBackend);
fn err(k: RhiErrorKind, a: &'static str, m: impl Into<String>) -> RhiError {
    RhiError::new(k, m.into()).at(a)
}
fn set(o: &Object, k: &str, v: JsValue) -> RhiResult<()> {
    Reflect::set(o, &JsValue::from_str(k), &v)
        .map_err(|e| {
            err(
                RhiErrorKind::BackendFailure,
                "WebGPU pipeline descriptor",
                js::message(&e),
            )
        })?
        .then_some(())
        .ok_or_else(|| {
            err(
                RhiErrorKind::BackendFailure,
                "WebGPU pipeline descriptor",
                "browser rejected descriptor field",
            )
        })
}
fn device(r: WebGpuRegistration) -> RhiResult<JsValue> {
    registry::with_device_handles(r, |h| h.device.clone()).ok_or_else(|| {
        err(
            RhiErrorKind::DeviceLost,
            "WebGPU pipeline",
            "device registration was retired",
        )
    })
}
fn create(r: WebGpuRegistration, method: &'static str, d: &Object) -> RhiResult<WebGpuObjectId> {
    let dev = device(r)?;
    let f = js::property(&dev, method)
        .map_err(|e| err(RhiErrorKind::Unsupported, method, js::message(&e)))?
        .dyn_into::<Function>()
        .map_err(|e| err(RhiErrorKind::Unsupported, method, js::message(&e)))?;
    let value = f
        .call1(&dev, d)
        .map_err(|e| err(RhiErrorKind::BackendFailure, method, js::message(&e)))?;
    registry::insert_object(r, value).ok_or_else(|| {
        err(
            RhiErrorKind::DeviceLost,
            method,
            "device registration was retired",
        )
    })
}
pub(crate) fn pipeline_layout_for(
    driver: &WebGpuDriver,
    interface: &crate::api::pipeline::PipelineInterface,
) -> RhiResult<JsValue> {
    let r = driver.registration();
    if !interface.descriptor().immediate_ranges.is_empty() {
        return Err(err(
            RhiErrorKind::Unsupported,
            "WebGPU pipeline layout",
            "immediate-data ranges",
        ));
    }
    let d = Object::new();
    let layouts = Array::new();
    for group in &interface.descriptor().groups {
        layouts.push(&binding::layout_for(driver, group)?);
    }
    set(&d, "bindGroupLayouts", layouts.into())?;
    if let Some(l) = interface.descriptor().label.as_deref() {
        set(&d, "label", JsValue::from_str(l))?;
    }
    let dev = device(r)?;
    let f = js::property(&dev, "createPipelineLayout")
        .map_err(|e| {
            err(
                RhiErrorKind::Unsupported,
                "createPipelineLayout",
                js::message(&e),
            )
        })?
        .dyn_into::<Function>()
        .map_err(|e| {
            err(
                RhiErrorKind::Unsupported,
                "createPipelineLayout",
                js::message(&e),
            )
        })?;
    f.call1(&dev, &d).map_err(|e| {
        err(
            RhiErrorKind::BackendFailure,
            "createPipelineLayout",
            js::message(&e),
        )
    })
}
fn module(r: WebGpuRegistration, s: &ShaderModule) -> RhiResult<JsValue> {
    let m = s
        .native()
        .as_any()
        .downcast_ref::<shader::WebGpuShaderModule>()
        .ok_or_else(|| {
            err(
                RhiErrorKind::WrongDevice,
                "WebGPU pipeline",
                "shader has another backend",
            )
        })?;
    if m.registration() != r {
        return Err(err(
            RhiErrorKind::WrongDevice,
            "WebGPU pipeline",
            "shader has another device",
        ));
    };
    registry::with_object(r, m.object(), Clone::clone).ok_or_else(|| {
        err(
            RhiErrorKind::DeviceLost,
            "WebGPU pipeline",
            "shader registration retired",
        )
    })
}
fn stage(r: WebGpuRegistration, s: &ShaderModule) -> RhiResult<JsValue> {
    let x = Object::new();
    set(&x, "module", module(r, s)?)?;
    set(
        &x,
        "entryPoint",
        JsValue::from_str(&s.artifact().entry_point),
    )?;
    Ok(x.into())
}
pub(crate) fn create_compute_pipeline(
    driver: &WebGpuDriver,
    d: &ComputePipelineDescriptor,
) -> RhiResult<WebGpuComputePipeline> {
    let r = driver.registration();
    let x = Object::new();
    set(&x, "layout", pipeline_layout_for(driver, &d.interface)?)?;
    set(&x, "compute", stage(r, &d.shader)?)?;
    if let Some(l) = d.label.as_deref() {
        set(&x, "label", JsValue::from_str(l))?;
    }
    Ok(WebGpuComputePipeline {
        driver: driver.clone(),
        registration: r,
        object: create(r, "createComputePipeline", &x)?,
    })
}
pub(crate) fn create_raster_pipeline(
    driver: &WebGpuDriver,
    d: &RasterPipelineDescriptor,
) -> RhiResult<WebGpuRasterPipeline> {
    let r = driver.registration();
    // Every state below is native WebGPU state, never a guessed fallback.  The
    // portable validator has already rejected optional modes this baseline has
    // no exact spelling.
    let x = Object::new();
    set(&x, "layout", pipeline_layout_for(driver, &d.interface)?)?;
    let vertex = Object::new();
    set(&vertex, "module", module(r, &d.vertex)?)?;
    set(
        &vertex,
        "entryPoint",
        JsValue::from_str(&d.vertex.artifact().entry_point),
    )?;
    let buffers = Array::new();
    for b in &d.vertex_input.buffers {
        let q = Object::new();
        set(&q, "arrayStride", JsValue::from_f64(b.stride as f64))?;
        set(
            &q,
            "stepMode",
            JsValue::from_str(match b.step_mode {
                crate::api::pipeline::VertexStepMode::Vertex => "vertex",
                crate::api::pipeline::VertexStepMode::Instance => "instance",
            }),
        )?;
        let attrs = Array::new();
        for a in &b.attributes {
            let z = Object::new();
            set(
                &z,
                "shaderLocation",
                JsValue::from_f64(a.location.get() as f64),
            )?;
            set(&z, "offset", JsValue::from_f64(a.offset as f64))?;
            set(
                &z,
                "format",
                JsValue::from_str(translate::vertex_format(a.format).map_err(|e| {
                    err(RhiErrorKind::Unsupported, "WebGPU raster pipeline", e.what)
                })?),
            )?;
            attrs.push(&z);
        }
        set(&q, "attributes", attrs.into())?;
        buffers.push(&q);
    }
    set(&vertex, "buffers", buffers.into())?;
    set(&x, "vertex", vertex.into())?;
    if let Some(fragment) = &d.fragment {
        let f = Object::new();
        set(&f, "module", module(r, fragment)?)?;
        set(
            &f,
            "entryPoint",
            JsValue::from_str(&fragment.artifact().entry_point),
        )?;
        let targets = Array::new();
        for target in &d.color_targets {
            match target {
                None => {
                    targets.push(&JsValue::NULL);
                }
                Some(t) => {
                    let q = Object::new();
                    set(
                        &q,
                        "format",
                        JsValue::from_str(translate::texture_format(t.format).map_err(|e| {
                            err(RhiErrorKind::Unsupported, "WebGPU raster pipeline", e.what)
                        })?),
                    )?;
                    set(
                        &q,
                        "writeMask",
                        JsValue::from_f64(translate::color_write_mask(t.write_mask) as f64),
                    )?;
                    if let Some(blend) = t.blend {
                        set(&q, "blend", blend_state(blend)?)?;
                    }
                    targets.push(&q);
                }
            }
        }
        set(&f, "targets", targets.into())?;
        set(&x, "fragment", f.into())?;
    }
    let primitive = Object::new();
    set(
        &primitive,
        "topology",
        JsValue::from_str(
            translate::primitive_topology(d.primitive.topology)
                .map_err(|e| err(RhiErrorKind::Unsupported, "WebGPU raster pipeline", e.what))?,
        ),
    )?;
    set(
        &primitive,
        "frontFace",
        JsValue::from_str(
            translate::front_face(d.primitive.front_face)
                .map_err(|e| err(RhiErrorKind::Unsupported, "WebGPU raster pipeline", e.what))?,
        ),
    )?;
    if let Some(c) = translate::cull_mode(d.primitive.cull_mode)
        .map_err(|e| err(RhiErrorKind::Unsupported, "WebGPU raster pipeline", e.what))?
    {
        set(&primitive, "cullMode", JsValue::from_str(c))?;
    }
    if let Some(format) = d.primitive.strip_index_format {
        set(
            &primitive,
            "stripIndexFormat",
            JsValue::from_str(translate::index_format(format)),
        )?;
    }
    set(&x, "primitive", primitive.into())?;
    if let Some(depth_stencil) = &d.depth_stencil {
        set(
            &x,
            "depthStencil",
            depth_stencil_state(depth_stencil, d.primitive.depth_bias)?,
        )?;
    } else if d.primitive.depth_bias.is_some() {
        // WebGPU carries depth bias in `GPUDepthStencilState`; the portable
        // validator deliberately permits the state value independently, but it
        // cannot be lowered without the attachment format that owns it.
        return Err(err(
            RhiErrorKind::InvalidUsage,
            "WebGPU raster pipeline",
            "depth bias requires a depth-stencil attachment on WebGPU",
        ));
    }
    let ms = Object::new();
    set(&ms, "count", JsValue::from_f64(d.multisample.count as f64))?;
    set(&ms, "mask", JsValue::from_f64(d.multisample.mask as f64))?;
    set(
        &ms,
        "alphaToCoverageEnabled",
        JsValue::from_bool(d.multisample.alpha_to_coverage_enabled),
    )?;
    set(&x, "multisample", ms.into())?;
    if let Some(l) = d.label.as_deref() {
        set(&x, "label", JsValue::from_str(l))?;
    }
    Ok(WebGpuRasterPipeline {
        driver: driver.clone(),
        registration: r,
        object: create(r, "createRenderPipeline", &x)?,
    })
}

/// Builds the exact `GPUBlendState` dictionary rather than relying on WebGPU
/// defaults.  Defaults are only safe when the portable descriptor also asked
/// for them; keeping both components explicit prevents colour/alpha drift.
fn blend_state(value: crate::api::pipeline::BlendState) -> RhiResult<JsValue> {
    fn component(value: crate::api::pipeline::BlendComponent) -> RhiResult<JsValue> {
        let object = Object::new();
        set(
            &object,
            "srcFactor",
            JsValue::from_str(translate::blend_factor(value.src_factor).map_err(|error| {
                err(RhiErrorKind::Unsupported, "WebGPU blend state", error.what)
            })?),
        )?;
        set(
            &object,
            "dstFactor",
            JsValue::from_str(translate::blend_factor(value.dst_factor).map_err(|error| {
                err(RhiErrorKind::Unsupported, "WebGPU blend state", error.what)
            })?),
        )?;
        set(
            &object,
            "operation",
            JsValue::from_str(
                translate::blend_operation(value.operation).map_err(|error| {
                    err(RhiErrorKind::Unsupported, "WebGPU blend state", error.what)
                })?,
            ),
        )?;
        Ok(object.into())
    }

    let object = Object::new();
    set(&object, "color", component(value.color)?)?;
    set(&object, "alpha", component(value.alpha)?)?;
    Ok(object.into())
}

/// Builds `GPUDepthStencilState`, including the independent depth/stencil
/// defaults required when the portable descriptor enables only one aspect.
fn depth_stencil_state(
    value: &crate::api::pipeline::DepthStencilState,
    bias: Option<crate::api::pipeline::DepthBiasState>,
) -> RhiResult<JsValue> {
    use crate::api::resource::CompareFunction;

    fn face(value: crate::api::pipeline::StencilFaceState) -> RhiResult<JsValue> {
        let object = Object::new();
        set(
            &object,
            "compare",
            JsValue::from_str(translate::compare_function(value.compare).map_err(|error| {
                err(
                    RhiErrorKind::Unsupported,
                    "WebGPU stencil state",
                    error.what,
                )
            })?),
        )?;
        for (name, operation) in [
            ("failOp", value.fail_op),
            ("depthFailOp", value.depth_fail_op),
            ("passOp", value.pass_op),
        ] {
            set(
                &object,
                name,
                JsValue::from_str(stencil_operation(operation)),
            )?;
        }
        Ok(object.into())
    }

    let object = Object::new();
    set(
        &object,
        "format",
        JsValue::from_str(translate::texture_format(value.format).map_err(|error| {
            err(
                RhiErrorKind::Unsupported,
                "WebGPU depth-stencil state",
                error.what,
            )
        })?),
    )?;
    let depth = value.depth.unwrap_or(crate::api::pipeline::DepthState {
        write_enabled: false,
        compare: CompareFunction::Always,
    });
    set(
        &object,
        "depthWriteEnabled",
        JsValue::from_bool(depth.write_enabled),
    )?;
    set(
        &object,
        "depthCompare",
        JsValue::from_str(translate::compare_function(depth.compare).map_err(|error| {
            err(
                RhiErrorKind::Unsupported,
                "WebGPU depth-stencil state",
                error.what,
            )
        })?),
    )?;
    if let Some(stencil) = value.stencil {
        set(&object, "stencilFront", face(stencil.front)?)?;
        set(&object, "stencilBack", face(stencil.back)?)?;
        set(
            &object,
            "stencilReadMask",
            JsValue::from_f64(stencil.read_mask as f64),
        )?;
        set(
            &object,
            "stencilWriteMask",
            JsValue::from_f64(stencil.write_mask as f64),
        )?;
    }
    if let Some(bias) = bias {
        set(
            &object,
            "depthBias",
            JsValue::from_f64(bias.constant as f64),
        )?;
        set(
            &object,
            "depthBiasSlopeScale",
            JsValue::from_f64(bias.slope_scale as f64),
        )?;
        set(
            &object,
            "depthBiasClamp",
            JsValue::from_f64(bias.clamp as f64),
        )?;
    }
    Ok(object.into())
}

fn stencil_operation(value: crate::api::pipeline::StencilOperation) -> &'static str {
    use crate::api::pipeline::StencilOperation;
    match value {
        StencilOperation::Keep => "keep",
        StencilOperation::Zero => "zero",
        StencilOperation::Replace => "replace",
        StencilOperation::Invert => "invert",
        StencilOperation::IncrementClamp => "increment-clamp",
        StencilOperation::DecrementClamp => "decrement-clamp",
        StencilOperation::IncrementWrap => "increment-wrap",
        StencilOperation::DecrementWrap => "decrement-wrap",
    }
}
