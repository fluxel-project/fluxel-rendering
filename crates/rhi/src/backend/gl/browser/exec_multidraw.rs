//! Browser batch-draw execution and its deterministic single-draw fallback.
//!
//! WebGL2 core has no per-draw parameter batch, so the batch commands live on an
//! extension object rather than on the rendering context: they are reached
//! through raw JS, and they are only callable when the acquired object really
//! exposes them as functions. This module owns which commands such an object
//! must carry, how they are reflected, and how one batch is issued. Whether the
//! context proved the batch capability is discovery's decision, and the
//! single-draw path the fallback needs is the raster domain's.

use js_sys::{Array, Function, Int32Array};
use wasm_bindgen::{JsCast, JsValue};

use super::super::api::{
    GlCapability, GlError, GlFamilyApi as _, GlMultiDraw, GlMultiDrawApi, issue_single_draws,
};
use super::discovery::WebGl2BrowserDiscovery;
use super::exec_raster::{PreparedDraw, topology_mode};

/// The batch commands one acquired extension object must expose.
///
/// The object owns its commands, so it is retained as the receiver of every
/// call: a command invoked on the wrong receiver is not the interface the
/// browser published.
#[derive(Clone)]
pub(super) struct BrowserMultiDraw {
    object: JsValue,
    arrays: Function,
    arrays_instanced: Function,
    elements: Function,
    elements_instanced: Function,
}

impl BrowserMultiDraw {
    /// Reflects the batch commands off a reported extension object.
    ///
    /// Returns `None` unless every command is callable. An object that exists
    /// but cannot issue one of the four shapes would otherwise enable a
    /// capability whose command path throws at the next call, so the caller
    /// records the acquisition as failed and every batch falls back to single
    /// draws instead.
    pub(super) fn acquire(object: &JsValue) -> Option<Self> {
        Some(Self {
            object: object.clone(),
            arrays: command(object, "multiDrawArraysWEBGL")?,
            arrays_instanced: command(object, "multiDrawArraysInstancedWEBGL")?,
            elements: command(object, "multiDrawElementsWEBGL")?,
            elements_instanced: command(object, "multiDrawElementsInstancedWEBGL")?,
        })
    }

    /// The command for one draw shape, carrying one instance count per draw only
    /// when the batch needs it.
    fn select(&self, indexed: bool, per_draw_instances: bool) -> &Function {
        match (indexed, per_draw_instances) {
            (false, false) => &self.arrays,
            (false, true) => &self.arrays_instanced,
            (true, false) => &self.elements,
            (true, true) => &self.elements_instanced,
        }
    }
}

fn command(object: &JsValue, name: &str) -> Option<Function> {
    js_sys::Reflect::get(object, &JsValue::from_str(name))
        .ok()?
        .dyn_into::<Function>()
        .ok()
}

/// One per-draw parameter slice of a batch command.
fn int32_slice(values: &[i32]) -> JsValue {
    JsValue::from(Int32Array::new_from_slice(values))
}

impl GlMultiDrawApi for WebGl2BrowserDiscovery {
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError> {
        const OP: &str = "multi-draw";
        self.assert_provider_ready(OP)?;
        if self.pass.is_none() {
            return Err(Self::validation(OP, "no active render pass"));
        }
        // The capability row and the retained commands come from the same
        // acquisition, so both must hold before a combined command may run;
        // anything less takes the deterministic single-draw route.
        let proved = self
            .discovery()
            .capabilities()
            .supports(GlCapability::MultiDraw);
        let Some(commands) = self.multi_draw.clone().filter(|_| proved) else {
            return issue_single_draws(self, command);
        };
        let raster = self
            .raster
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
        let mode = topology_mode(raster.topology);
        // Every draw is readied before the first command is issued, so a batch
        // this provider cannot fully express leaves no partial submission.
        let prepared = command
            .draws()
            .iter()
            .map(|draw| self.prepare_draw(OP, *draw))
            .collect::<Result<Vec<_>, _>>()?;
        let indexed = matches!(prepared.first(), Some(PreparedDraw::Indexed { .. }));
        let per_draw_instances = command.needs_instance_counts();
        let function = commands.select(indexed, per_draw_instances);
        // One batch carries one shape, so only the slices of that shape are
        // filled and only they reach the call.
        let mut firsts = Vec::with_capacity(prepared.len());
        let mut counts = Vec::with_capacity(prepared.len());
        let mut offsets = Vec::with_capacity(prepared.len());
        let mut instances = Vec::with_capacity(prepared.len());
        let mut index_type = 0;
        for draw in &prepared {
            match *draw {
                PreparedDraw::NonIndexed {
                    first,
                    count,
                    instances: instance_count,
                } => {
                    firsts.push(first);
                    counts.push(count);
                    instances.push(instance_count);
                }
                PreparedDraw::Indexed {
                    count,
                    index_type: draw_type,
                    offset,
                    instances: instance_count,
                } => {
                    counts.push(count);
                    offsets.push(offset);
                    instances.push(instance_count);
                    index_type = draw_type;
                }
            }
        }
        let counts = int32_slice(&counts);
        let zero = JsValue::from_f64(0.0);
        // Every batch command takes the same parameter shape: the primitive
        // mode, one slice per draw, the optional per-draw instance counts, and
        // finally the draw count. The arguments are passed as an array so the
        // four shapes share one call path.
        let arguments = Array::new();
        arguments.push(&JsValue::from_f64(f64::from(mode)));
        if indexed {
            arguments.push(&counts);
            arguments.push(&zero);
            arguments.push(&JsValue::from_f64(f64::from(index_type)));
            arguments.push(&int32_slice(&offsets));
            arguments.push(&zero);
        } else {
            arguments.push(&int32_slice(&firsts));
            arguments.push(&zero);
            arguments.push(&counts);
            arguments.push(&zero);
        }
        if per_draw_instances {
            arguments.push(&int32_slice(&instances));
            arguments.push(&zero);
        }
        arguments.push(&JsValue::from_f64(prepared.len() as f64));
        function
            .apply(&commands.object, &arguments)
            .map_err(|value| Self::js_failure(OP, value))?;
        self.driver_error(OP)
    }
}
