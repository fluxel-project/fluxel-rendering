//! Explicit canvas-bound wasm bridge for the retained red/green/blue scene.
//!
//! The JavaScript adapter owns animation frames and DOM events.  This crate
//! owns neither a browser singleton nor a scene API; it only turns those
//! explicit lifecycle calls into the closed renderer/RHI execution path.

#![cfg(target_arch = "wasm32")]

mod webgpu;

pub use webgpu::WebGpuSession;

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_renderer::adapter::browser::WebGl2AssetResidency;
use fluxel_renderer::adapter::{PreparedBasicGraph, PreparedBasicScene};
use fluxel_renderer::{BasicMaterial, Camera, DrawList, Geometry, Mesh, MeshAsset, ModelTransform};
use fluxel_rhi::adapter::webgl2::{
    FixedResidentUnlitDraw, FixedUnlitGraph, WebGl2Session as RhiSession, WebGl2SessionError,
    WebGl2SessionState,
};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// Explicit one-canvas browser rendering session.
#[wasm_bindgen]
pub struct WebGl2Session {
    inner: Rc<RefCell<RhiSession>>,
    residency: WebGl2AssetResidency,
    /// Owns the logical identities every upload is keyed on.
    ///
    /// It is held rather than consulted per frame: which upload is due is the
    /// residency table's decision, and the table is keyed by snapshots this
    /// store minted. A content replacement, and the collection that follows
    /// it, would go through this value.
    #[expect(
        dead_code,
        reason = "held for the session's lifetime so the identities it minted stay owned"
    )]
    store: AssetStore<MeshAsset, Geometry, ()>,
    meshes: Vec<AssetSnapshot<MeshAsset, Geometry>>,
    scene: PreparedBasicScene,
    graph: Option<PreparedBasicGraph>,
}

#[wasm_bindgen]
impl WebGl2Session {
    /// Creates a session for this canvas only; no document/global lookup occurs.
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let scene = retained_scene().map_err(error)?;
        let store = AssetStore::new();
        let meshes = produce_scene_meshes(&store, &scene)?;
        let extent = [canvas.width(), canvas.height()];
        let graph = if extent[0] == 0 || extent[1] == 0 {
            None
        } else {
            Some(scene.compile_presentable_graph(extent).map_err(error)?)
        };
        let inner = Rc::new(RefCell::new(RhiSession::new(canvas.into()).map_err(error)?));
        let residency = WebGl2AssetResidency::new(Rc::clone(&inner));
        Ok(Self {
            inner,
            residency,
            store,
            meshes,
            scene,
            graph,
        })
    }
    /// Changes the exact drawing-buffer pixel extent.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let graph = if width == 0 || height == 0 {
            None
        } else {
            Some(
                self.scene
                    .compile_presentable_graph([width, height])
                    .map_err(error)?,
            )
        };
        self.inner
            .borrow_mut()
            .resize(width, height)
            .map_err(error)?;
        self.graph = graph;
        Ok(())
    }
    /// Renders once. The caller, not this session, owns RAF scheduling.
    pub fn render_once(&mut self) -> Result<JsValue, JsValue> {
        let started = js_sys::Date::now();
        if self.inner.borrow().state() != WebGl2SessionState::Active {
            return Err(structured_error(
                "invalid-state",
                "render",
                self.inner.borrow().generation(),
                "browser session is not active",
            ));
        }
        // One prepare per mesh per frame. The table, not this bridge, decides
        // whether an upload is due: it keys on the store's logical identity,
        // content generation, and context, so a repeated frame reuses the
        // resident buffers and a replaced generation re-uploads exactly once.
        let mut tokens = Vec::with_capacity(self.meshes.len());
        for mesh in &self.meshes {
            tokens.push(self.residency.prepare_mesh(mesh).map_err(error)?);
        }
        let draws: Vec<_> = self
            .scene
            .draws()
            .iter()
            .zip(&tokens)
            .map(|(draw, mesh)| FixedResidentUnlitDraw {
                mesh,
                pvm_and_color: draw.pvm_and_color(),
                insertion_index: draw.insertion_index(),
            })
            .collect();
        let graph = self.graph.as_ref().ok_or_else(|| {
            structured_error(
                "invalid-state",
                "render",
                self.inner.borrow().generation(),
                "presentable graph is suspended",
            )
        })?;
        let contract = FixedUnlitGraph::new(graph.compiled(), graph.draw_count(), graph.extent());
        let outcome = self.inner.borrow_mut().render_resident(&contract, &draws);
        let generation = self.inner.borrow().generation();
        let state = self.inner.borrow().state();
        let frame_marker = match outcome {
            Ok(marker) => marker,
            Err(WebGl2SessionError::Busy) => {
                return Ok(frame_report(
                    "backpressure",
                    None,
                    generation,
                    js_sys::Date::now() - started,
                    state,
                ));
            }
            Err(error_value) if error_value.code() == "context-lost" => {
                return Ok(frame_report(
                    "context-lost",
                    None,
                    generation,
                    js_sys::Date::now() - started,
                    state,
                ));
            }
            Err(error_value) => return Err(error(error_value)),
        };
        Ok(frame_report(
            "submitted",
            Some(frame_marker),
            generation,
            js_sys::Date::now() - started,
            state,
        ))
    }
    /// Stops new frames without claiming completion.
    pub fn suspend(&mut self) -> Result<(), JsValue> {
        self.inner.borrow_mut().suspend().map_err(error)
    }
    /// Resumes the fixed same-canvas session.
    pub fn resume(&mut self) -> Result<(), JsValue> {
        self.inner.borrow_mut().resume().map(|_| ()).map_err(error)
    }
    /// Applies an adapter-observed context-loss event.
    pub fn context_lost(&mut self) {
        self.inner.borrow_mut().context_lost();
    }
    /// Rebuilds resources after restoration of the same canvas only.
    pub fn context_restored(&mut self) -> Result<(), JsValue> {
        self.inner
            .borrow_mut()
            .context_restored()
            .map(|_| ())
            .map_err(error)?;
        // A restored context is a new device. Every token the old table holds
        // names buffers that no longer exist, so the table is replaced rather
        // than repaired; the logical identities and their content are the
        // store's, and they survive.
        self.residency = WebGl2AssetResidency::new(Rc::clone(&self.inner));
        Ok(())
    }
    /// Explicit shutdown with browser completion proof where available.
    pub fn dispose(&mut self) -> Result<(), JsValue> {
        self.inner.borrow_mut().dispose().map_err(error)
    }
    /// Returns a non-draining snapshot of RHI-generated structured diagnostics.
    pub fn diagnostics_snapshot(&self) -> Array {
        let result = Array::new();
        for item in self.inner.borrow().diagnostics() {
            let object = Object::new();
            let _ = Reflect::set(&object, &"code".into(), &item.code.into());
            let _ = Reflect::set(&object, &"severity".into(), &item.severity.into());
            let _ = Reflect::set(&object, &"operation".into(), &item.operation.into());
            let _ = Reflect::set(
                &object,
                &"generation".into(),
                &JsValue::from_f64(item.generation as f64),
            );
            let _ = Reflect::set(&object, &"message".into(), &item.message.clone().into());
            let context = Object::new();
            for (key, value) in &item.context {
                let _ = Reflect::set(&context, &(*key).into(), &value.clone().into());
            }
            let _ = Reflect::set(&object, &"context".into(), &context);
            result.push(&object);
        }
        result
    }
}

/// Produces the fixed scene's meshes into a store and returns their snapshots.
///
/// The scene is a fixed list rather than a loader, so this is where its three
/// meshes acquire the logical identity the residency table keys on. The store
/// mints it; nothing in this bridge invents an identity, and nothing here
/// decides when an upload is due. The geometries are rebuilt from the prepared
/// draws because the prepared scene is what owns them after preparation, and
/// they carry exactly the positions and indices it was prepared from.
fn produce_scene_meshes(
    store: &AssetStore<MeshAsset, Geometry, ()>,
    scene: &PreparedBasicScene,
) -> Result<Vec<AssetSnapshot<MeshAsset, Geometry>>, JsValue> {
    scene
        .draws()
        .iter()
        .map(|draw| {
            let geometry = Geometry::from_positions(draw.positions().to_vec())
                .with_indices(draw.indices().to_vec())
                .map_err(|error| {
                    structured_error(
                        "scene-preparation-failed",
                        "prepare-scene",
                        0,
                        &error.to_string(),
                    )
                })?;
            let bytes = (geometry.positions().len() * 12 + geometry.indices().len() * 4) as u64;
            let handle = store.create().map_err(store_error)?;
            let Acquire::Producer(producer) = store.acquire(&handle).map_err(store_error)? else {
                return Err(store_error_message(
                    "the fixed scene's own asset was already produced",
                ));
            };
            producer
                .commit(geometry, ResidentBytes::new(bytes))
                .map_err(store_error)
        })
        .collect()
}

fn store_error(error: fluxel_assets::AssetError) -> JsValue {
    store_error_message(&error.to_string())
}

fn store_error_message(message: &str) -> JsValue {
    structured_error("asset-store-failed", "prepare-scene", 0, message)
}

fn frame_report(
    outcome: &str,
    frame_marker: Option<u64>,
    generation: u64,
    cpu_submission_ms: f64,
    state: fluxel_rhi::adapter::webgl2::WebGl2SessionState,
) -> JsValue {
    let report = Object::new();
    let _ = Reflect::set(&report, &"outcome".into(), &outcome.into());
    if let Some(frame_marker) = frame_marker {
        let _ = Reflect::set(
            &report,
            &"frameMarker".into(),
            &JsValue::from_f64(frame_marker as f64),
        );
    }
    let _ = Reflect::set(&report, &"wasmMemoryBytes".into(), &wasm_memory_bytes());
    let _ = Reflect::set(
        &report,
        &"generation".into(),
        &JsValue::from_f64(generation as f64),
    );
    let _ = Reflect::set(
        &report,
        &"cpuSubmissionMs".into(),
        &JsValue::from_f64(cpu_submission_ms),
    );
    let _ = Reflect::set(&report, &"state".into(), &format!("{state:?}").into());
    report.into()
}

pub(crate) fn retained_scene()
-> Result<PreparedBasicScene, fluxel_renderer::adapter::PreparedBasicSceneError> {
    let triangle = |color| {
        Mesh::new(
            Geometry::from_positions(vec![
                [-0.16, -0.16, 0.5],
                [0.16, -0.16, 0.5],
                [0.0, 0.16, 0.5],
            ])
            .with_indices(vec![0, 1, 2])
            .expect("fixed valid triangle"),
            BasicMaterial::new(color).expect("fixed unit color"),
        )
    };
    let red = triangle([1.0, 0.0, 0.0, 1.0]);
    let green = triangle([0.0, 1.0, 0.0, 1.0]);
    let blue = triangle([0.0, 0.35, 1.0, 1.0]);
    let camera = Camera::default();
    let mut draws = DrawList::new(&camera);
    for (mesh, x, y) in [
        (&red, -0.45, -0.35),
        (&green, 0.45, -0.35),
        (&blue, 0.0, 0.42),
    ] {
        let mut matrix = ModelTransform::IDENTITY.world_from_model().to_owned();
        matrix[3][0] = x;
        matrix[3][1] = y;
        draws.push_transformed(
            mesh,
            ModelTransform::from_column_major(matrix).expect("finite affine placement"),
        );
    }
    PreparedBasicScene::prepare(&draws)
}
fn error(value: impl IntoBrowserError) -> JsValue {
    value.into_browser_error()
}

trait IntoBrowserError {
    fn into_browser_error(self) -> JsValue;
}

impl IntoBrowserError for WebGl2SessionError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            self.code(),
            self.operation(),
            self.generation(),
            &self.message(),
        )
    }
}

impl IntoBrowserError for fluxel_renderer::adapter::browser::WebGl2ResidencyError {
    fn into_browser_error(self) -> JsValue {
        match self {
            // A refused upload is the executor's own structured rejection, so
            // it keeps the code, operation and generation the executor chose.
            Self::UploadRejected(error) => error.into_browser_error(),
            Self::StaleMeshGeneration => structured_error(
                "stale-generation",
                "prepare-resident-mesh",
                0,
                "browser residency rejected a stale content generation",
            ),
            Self::NotResident => structured_error(
                "not-resident",
                "prepare-resident-mesh",
                0,
                &self.to_string(),
            ),
            // The error type is non-exhaustive, so a variant this bridge does
            // not know is reported as an unnamed residency failure rather than
            // being silently folded into one of the cases above.
            _ => structured_error(
                "residency-failed",
                "prepare-resident-mesh",
                0,
                &self.to_string(),
            ),
        }
    }
}

impl IntoBrowserError for fluxel_renderer::adapter::PreparedBasicSceneError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            "scene-preparation-failed",
            "prepare-scene",
            0,
            &self.to_string(),
        )
    }
}

impl IntoBrowserError for fluxel_renderer::adapter::PreparedBasicGraphError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            "graph-compile-failed",
            "compile-graph",
            0,
            &self.to_string(),
        )
    }
}

pub(crate) fn structured_error(
    code: &str,
    operation: &str,
    generation: u64,
    message: &str,
) -> JsValue {
    let object = Object::new();
    let _ = Reflect::set(&object, &"code".into(), &code.into());
    let _ = Reflect::set(&object, &"severity".into(), &"error".into());
    let _ = Reflect::set(&object, &"operation".into(), &operation.into());
    let _ = Reflect::set(
        &object,
        &"generation".into(),
        &JsValue::from_f64(generation as f64),
    );
    let _ = Reflect::set(&object, &"message".into(), &message.into());
    let _ = Reflect::set(&object, &"context".into(), &Object::new());
    object.into()
}

pub(crate) fn wasm_memory_bytes() -> JsValue {
    let memory = wasm_bindgen::memory();
    let buffer = js_sys::Reflect::get(&memory, &"buffer".into()).unwrap_or(JsValue::UNDEFINED);
    js_sys::Reflect::get(&buffer, &"byteLength".into()).unwrap_or(JsValue::from_f64(0.0))
}
