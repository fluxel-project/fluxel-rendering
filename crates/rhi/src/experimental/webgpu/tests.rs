//! Unit contracts for the closed browser executor types.

use js_sys::{Function, Object, Promise};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

use super::*;
use crate::experimental::webgpu::js::BrowserRequestProvider;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn resident_uploads_are_unconditional_and_a_new_device_retires_every_token() {
    let mut session = WebGpuSession::new(canvas()).await.expect("WebGPU session");
    let positions = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0, 1, 2];
    let mesh = session
        .upload_resident_mesh(&positions, &indices)
        .expect("upload mesh");
    let image = session
        .upload_resident_image([1, 1], &[9, 8, 7, 6])
        .expect("upload image");
    assert!(session.resident_mesh_current(&mesh));
    assert!(session.resident_image_current(&image));
    assert_eq!(mesh.device(), session.device_identity());
    // Asking twice is two uploads rather than a reuse: the session cannot know
    // that these bytes are a revision it already holds, because the revision is
    // the caller's fact and the caller is a keyed table.
    let second = session
        .upload_resident_mesh(&positions, &indices)
        .expect("second mesh");
    assert!(session.resident_mesh_current(&second));
    assert!(session.resident_mesh_current(&mesh));
    let device = session.device_identity();
    session.controlled_destroy_for_evidence().expect("destroy");
    lose(&mut session).await;
    JsFuture::from(session.recover().expect("recover"))
        .await
        .expect("recovered");
    assert_eq!(session.generation(), 2);
    // The recovery installs a different `GPUDevice`, and that -- not the
    // generation counter -- is what a token names.
    assert_ne!(session.device_identity(), device);
    assert!(!session.resident_mesh_current(&mesh));
    assert!(!session.resident_image_current(&image));
    let recreated = session
        .upload_resident_mesh(&positions, &indices)
        .expect("reupload");
    assert!(session.resident_mesh_current(&recreated));
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn fixed_resource_floor_and_compute_have_deterministic_map_read_oracles() {
    let mut session = WebGpuSession::new(canvas())
        .await
        .expect("real WebGPU session for fixed resource conformance");
    let frame = FixedResourceFrame {
        resources: WebGpuResourceKey::new(17),
        positions: &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]],
        indices: &[0, 1, 2],
        tint: [1.0, 1.0, 1.0, 1.0],
        sampled_rgba8: [12, 34, 56, 255],
    };
    assert!(matches!(
        session.render_fixed_resources(frame),
        Ok(WebGpuRenderOutcome::Submitted(_))
    ));
    session
        .resource_conformance()
        .await
        .expect("fixed resource floor must pass browser validation and exact MAP_READ oracles");
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn fixed_resource_registry_recreates_after_loss_and_disposal_joins_its_ticket() {
    let mut session = WebGpuSession::new(canvas())
        .await
        .expect("real WebGPU session for retained-resource lifecycle");
    let frame = || FixedResourceFrame {
        resources: WebGpuResourceKey::new(91),
        positions: &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]],
        indices: &[0, 1, 2],
        tint: [1.0, 1.0, 1.0, 1.0],
        sampled_rgba8: [12, 34, 56, 255],
    };
    assert!(matches!(
        session.render_fixed_resources(frame()),
        Ok(WebGpuRenderOutcome::Submitted(_))
    ));
    // This drops the registry's reference only: the first accepted submission
    // remains ticket-owned until completion, and this key receives a fresh
    // physical set on its next use.
    session.retire_fixed_resources(WebGpuResourceKey::new(91));
    assert!(matches!(
        session.render_fixed_resources(frame()),
        Ok(WebGpuRenderOutcome::Submitted(_))
    ));
    session
        .controlled_destroy_for_evidence()
        .expect("destroy active device");
    lose(&mut session).await;
    let recovery = session.recover().expect("recover after loss");
    JsFuture::from(recovery).await.expect("recovery succeeds");
    assert_eq!(session.generation(), 2);
    assert!(matches!(
        session.render_fixed_resources(frame()),
        Ok(WebGpuRenderOutcome::Submitted(_))
    ));
    dispose(&mut session).await;
}

struct PendingRequest {
    target: JsValue,
    kind: RequestKind,
    resolve: Function,
    reject: Function,
}

#[derive(Clone, Copy)]
enum RequestKind {
    Adapter,
    Device,
}

struct GatedBrowserRequests {
    production: ProductionBrowserRequests,
    defer_adapter: Cell<bool>,
    defer_device: Cell<bool>,
    adapter_calls: Cell<usize>,
    device_calls: Cell<usize>,
    adapter_pending: Rc<RefCell<Option<PendingRequest>>>,
    device_pending: Rc<RefCell<Option<PendingRequest>>>,
}

impl Default for GatedBrowserRequests {
    fn default() -> Self {
        Self {
            production: ProductionBrowserRequests,
            defer_adapter: Cell::new(false),
            defer_device: Cell::new(false),
            adapter_calls: Cell::new(0),
            device_calls: Cell::new(0),
            adapter_pending: Rc::new(RefCell::new(None)),
            device_pending: Rc::new(RefCell::new(None)),
        }
    }
}

impl GatedBrowserRequests {
    fn arm_adapter(&self) {
        self.defer_adapter.set(true);
    }

    fn arm_device(&self) {
        self.defer_device.set(true);
    }

    fn pending_adapter(&self) -> bool {
        self.adapter_pending.borrow().is_some()
    }

    fn pending_device(&self) -> bool {
        self.device_pending.borrow().is_some()
    }

    fn release_adapter(&self) {
        Self::release(&self.adapter_pending);
    }

    fn release_device(&self) {
        Self::release(&self.device_pending);
    }

    fn reject_adapter(&self, message: &str) {
        Self::reject(&self.adapter_pending, message);
    }

    fn reject_device(&self, message: &str) {
        Self::reject(&self.device_pending, message);
    }

    fn deferred(
        target: &JsValue,
        kind: RequestKind,
        pending: Rc<RefCell<Option<PendingRequest>>>,
    ) -> Promise {
        let target = target.clone();
        Promise::new(&mut move |resolve, reject| {
            assert!(
                pending
                    .borrow_mut()
                    .replace(PendingRequest {
                        target: target.clone(),
                        kind,
                        resolve,
                        reject,
                    })
                    .is_none(),
                "only one browser request may occupy a gate"
            );
        })
    }

    fn release(pending: &RefCell<Option<PendingRequest>>) {
        let pending = pending
            .borrow_mut()
            .take()
            .expect("a deferred browser request is pending");
        let production = ProductionBrowserRequests;
        let native = match pending.kind {
            RequestKind::Adapter => production.request_adapter(&pending.target),
            RequestKind::Device => production.request_device(&pending.target),
        };
        match native {
            Ok(native) => {
                let _ = call2(
                    &native,
                    "then",
                    pending.resolve.as_ref(),
                    pending.reject.as_ref(),
                )
                .expect("native browser request exposes Promise.then");
            }
            Err(error) => {
                pending
                    .reject
                    .call1(&JsValue::UNDEFINED, &error)
                    .expect("reject deferred browser request");
            }
        }
    }

    fn reject(pending: &RefCell<Option<PendingRequest>>, message: &str) {
        let pending = pending
            .borrow_mut()
            .take()
            .expect("a deferred browser request is pending");
        pending
            .reject
            .call1(&JsValue::UNDEFINED, &js_sys::Error::new(message))
            .expect("reject deferred browser request");
    }
}

impl BrowserRequestProvider for GatedBrowserRequests {
    fn request_adapter(&self, gpu: &JsValue) -> Result<Promise, JsValue> {
        self.adapter_calls.set(self.adapter_calls.get() + 1);
        if self.defer_adapter.replace(false) {
            Ok(Self::deferred(
                gpu,
                RequestKind::Adapter,
                Rc::clone(&self.adapter_pending),
            ))
        } else {
            self.production.request_adapter(gpu)
        }
    }

    fn request_device(&self, adapter: &JsValue) -> Result<Promise, JsValue> {
        self.device_calls.set(self.device_calls.get() + 1);
        if self.defer_device.replace(false) {
            Ok(Self::deferred(
                adapter,
                RequestKind::Device,
                Rc::clone(&self.device_pending),
            ))
        } else {
            self.production.request_device(adapter)
        }
    }
}

fn canvas() -> JsValue {
    js_sys::eval("document.createElement('canvas')")
        .expect("browser test creates an isolated canvas")
}

async fn turn() {
    let promise = js_sys::eval("new Promise(resolve => setTimeout(resolve, 0))")
        .expect("browser test can yield to the event loop");
    JsFuture::from(Promise::from(promise))
        .await
        .expect("event-loop turn resolves");
}

async fn wait_until(mut predicate: impl FnMut() -> bool, description: &str) {
    for _ in 0..200 {
        if predicate() {
            return;
        }
        turn().await;
    }
    panic!("timed out waiting for {description}");
}

async fn lose(session: &mut WebGpuSession) {
    session
        .controlled_destroy_for_evidence()
        .expect("destroy current real device");
    wait_until(
        || session.state() == WebGpuSessionState::Lost,
        "the production device.lost callback",
    )
    .await;
}

async fn dispose(session: &mut WebGpuSession) {
    let promise = session.dispose().expect("start terminal disposal");
    JsFuture::from(promise)
        .await
        .expect("terminal disposal completes");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
}

#[test]
fn formats_are_closed() {
    assert_eq!(
        WebGpuCanvasFormat::parse("rgba8unorm"),
        Some(WebGpuCanvasFormat::Rgba8Unorm)
    );
    assert!(WebGpuCanvasFormat::parse("rgba8unorm-srgb").is_none());
}

#[test]
fn lifecycle_has_distinct_loss_and_dispose() {
    assert_ne!(WebGpuSessionState::Lost, WebGpuSessionState::Disposed);
    assert_ne!(MAX_FRAMES_IN_FLIGHT, 0);
}

#[test]
fn recovery_publication_requires_its_original_token_and_state() {
    let mut shared = Shared {
        state: WebGpuSessionState::Recovering,
        generation: 7,
        token: 11,
        ..Shared::default()
    };
    assert!(recovery_attempt_current(shared.state, shared.token, 11));

    // `dispose` invalidates the token before it waits for the in-flight
    // recovery. The candidate must therefore never publish its facts.
    shared.token += 1;
    shared.state = WebGpuSessionState::Disposing;
    assert!(!recovery_attempt_current(shared.state, shared.token, 11));
    assert_eq!(shared.generation, 7);
}

#[test]
fn recovery_candidate_is_not_committable_from_a_terminal_state() {
    let shared = Shared {
        state: WebGpuSessionState::Disposed,
        token: 3,
        ..Shared::default()
    };
    assert!(!recovery_attempt_current(shared.state, shared.token, 3));
}

#[wasm_bindgen_test(async)]
async fn public_constructor_uses_the_production_browser_provider() {
    let mut session = WebGpuSession::new(canvas())
        .await
        .expect("the public constructor opens a real WebGPU session");
    assert_eq!(session.generation(), 1);
    assert_eq!(session.state(), WebGpuSessionState::Active);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_resolves_the_real_adapter_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    assert_eq!(requests.adapter_calls.get(), 1);
    assert_eq!(requests.device_calls.get(), 1);
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    assert_eq!(session.state(), WebGpuSessionState::Recovering);
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    assert_eq!(session.generation(), 1);
    requests.release_adapter();
    JsFuture::from(recovery)
        .await
        .expect("native adapter and device requests resolve");

    assert_eq!(session.state(), WebGpuSessionState::Active);
    assert_eq!(session.generation(), 2);
    assert_eq!(requests.adapter_calls.get(), 2);
    assert_eq!(requests.device_calls.get(), 2);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_rejects_the_adapter_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    requests.reject_adapter("injected requestAdapter rejection");
    JsFuture::from(recovery)
        .await
        .expect_err("recovery must reject");

    assert_eq!(session.state(), WebGpuSessionState::Poisoned);
    assert_eq!(session.generation(), 1);
    assert!(session.diagnostics().iter().any(|diagnostic| {
        diagnostic.code == "recovery-request-failed"
            && diagnostic.operation == "recover-request"
            && diagnostic.generation == 1
    }));
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_resolves_the_real_device_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_device();
    let recovery = session.recover().expect("start recovery");
    wait_until(|| requests.pending_device(), "deferred requestDevice call").await;
    assert_eq!(session.state(), WebGpuSessionState::Recovering);
    assert_eq!(session.generation(), 1);
    requests.release_device();
    JsFuture::from(recovery)
        .await
        .expect("native device request resolves");

    assert_eq!(session.state(), WebGpuSessionState::Active);
    assert_eq!(session.generation(), 2);
    assert_eq!(requests.adapter_calls.get(), 2);
    assert_eq!(requests.device_calls.get(), 2);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_rejects_the_device_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_device();
    let recovery = session.recover().expect("start recovery");
    wait_until(|| requests.pending_device(), "deferred requestDevice call").await;
    requests.reject_device("injected requestDevice rejection");
    JsFuture::from(recovery)
        .await
        .expect_err("recovery must reject");

    assert_eq!(session.state(), WebGpuSessionState::Poisoned);
    assert_eq!(session.generation(), 1);
    assert!(session.diagnostics().iter().any(|diagnostic| {
        diagnostic.code == "recovery-request-failed"
            && diagnostic.operation == "recover-request"
            && diagnostic.generation == 1
    }));
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn disposal_joins_a_deferred_adapter_resolution_without_publication() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    let disposal = session.dispose().expect("dispose pending recovery");
    assert_eq!(session.state(), WebGpuSessionState::Disposing);
    requests.release_adapter();

    JsFuture::from(recovery)
        .await
        .expect("stale recovery resolves without publication");
    JsFuture::from(disposal)
        .await
        .expect("disposal joins stale native requests");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
    assert_eq!(session.generation(), 1);
    assert!(
        session
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.code != "recovery-request-failed")
    );
}

#[wasm_bindgen_test(async)]
async fn disposal_joins_a_deferred_adapter_rejection_without_poisoning() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    let disposal = session.dispose().expect("dispose pending recovery");
    assert_eq!(session.state(), WebGpuSessionState::Disposing);
    requests.reject_adapter("stale requestAdapter rejection");

    JsFuture::from(recovery)
        .await
        .expect("stale rejection does not reject recovery");
    JsFuture::from(disposal)
        .await
        .expect("disposal joins stale rejection");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
    assert_eq!(session.generation(), 1);
    assert!(
        session
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.code != "recovery-request-failed")
    );
}

/// Resolves the production browser device, then instruments its
/// `createBuffer`, `createBindGroup`, and `queue.writeBuffer` calls with
/// usage-kind counters so a test can observe exactly which resources each
/// draw path creates. Counters are published on `globalThis.__fluxelDrawCounts`.
struct CountingBrowserRequests;

impl BrowserRequestProvider for CountingBrowserRequests {
    fn request_adapter(&self, gpu: &JsValue) -> Result<Promise, JsValue> {
        ProductionBrowserRequests.request_adapter(gpu)
    }

    fn request_device(&self, adapter: &JsValue) -> Result<Promise, JsValue> {
        let native = ProductionBrowserRequests.request_device(adapter)?;
        let instrument = js_sys::eval(INSTRUMENT_DEVICE_SOURCE)
            .expect("browser test evals the instrumentor")
            .dyn_into::<Function>()
            .expect("instrumentor is a function");
        let instrumented = instrument.call1(&JsValue::UNDEFINED, &native.into())?;
        Ok(Promise::from(instrumented))
    }
}

const INSTRUMENT_DEVICE_SOURCE: &str = r#"promise => promise.then(device => {
    const counts = {
        vertex: 0, index: 0, uniform: 0, bindGroups: 0,
        writeVertex: 0, writeIndex: 0, writeUniform: 0,
    };
    const kindOf = usage => (usage & 0x40) !== 0 ? 'uniform'
        : (usage & 0x10) !== 0 ? 'index'
        : (usage & 0x20) !== 0 ? 'vertex' : 'other';
    const realCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = descriptor => {
        const kind = kindOf(descriptor.usage);
        if (kind !== 'other') {
            counts[kind] += 1;
        }
        const buffer = realCreateBuffer(descriptor);
        try { buffer.__fluxelKind = kind; } catch (_error) {}
        return buffer;
    };
    const realCreateBindGroup = device.createBindGroup.bind(device);
    device.createBindGroup = (...args) => {
        counts.bindGroups += 1;
        return realCreateBindGroup(...args);
    };
    const queue = device.queue;
    const realWriteBuffer = queue.writeBuffer.bind(queue);
    queue.writeBuffer = (buffer, ...rest) => {
        const kind = buffer.__fluxelKind || 'other';
        if (kind === 'vertex') {
            counts.writeVertex += 1;
        } else if (kind === 'index') {
            counts.writeIndex += 1;
        } else if (kind === 'uniform') {
            counts.writeUniform += 1;
        }
        return realWriteBuffer(buffer, ...rest);
    };
    globalThis.__fluxelDrawCounts = counts;
    return device;
})"#;

fn draw_counter(name: &str) -> u32 {
    let counts = js_sys::Reflect::get(&js_sys::global(), &"__fluxelDrawCounts".into())
        .expect("instrumented session publishes draw counters")
        .dyn_into::<Object>()
        .expect("draw counters are an object");
    js_sys::Reflect::get(&counts, &JsValue::from_str(name))
        .expect("counter exists")
        .as_f64()
        .expect("counter is numeric") as u32
}

fn sized_canvas() -> JsValue {
    js_sys::eval(
        "(() => { const canvas = document.createElement('canvas');
           canvas.width = 8; canvas.height = 8; return canvas; })()",
    )
    .expect("browser test creates a sized canvas")
}

/// Compiles the fixed one-draw resident graph for the counting test.
fn resident_graph(
    format: WebGpuCanvasFormat,
    extent: [u32; 2],
) -> fluxel_rendergraph::CompiledGraph<()> {
    use fluxel_rendergraph::{
        AttachmentOps, BufferCapabilities, BufferDesc, BufferRange, BufferReadUse,
        ColorAttachmentDesc, DeviceCapabilities, DeviceLimits, Extent3d, ExternalOwnership,
        ImportBufferContract, InitialContents, LoadOp, PresentContract, QueueCapabilities,
        QueueDescriptor, QueueId, RecordingCapabilities, RecordingModel, RenderGraph,
        ResourceAccessState, StoreOp, SurfaceCapabilities, SurfaceTextureContract,
        SynchronizationCapabilities, TextureDesc, TextureDimension, TextureFormat,
        TextureFormatCapabilities, TextureRange, TimestampCapabilities,
        TransientResourceCapabilities, TransitionCapabilities, WriteCoverage,
    };
    let texture_format = match format {
        WebGpuCanvasFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        WebGpuCanvasFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
    };
    let imported = |size: u64| ImportBufferContract {
        descriptor: BufferDesc { size },
        initial_state: ResourceAccessState::CopyDestination,
        ownership: ExternalOwnership::Caller,
        initial_contents: InitialContents::Defined,
    };
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot("counting-positions", imported(36));
    let indices = graph.import_buffer_slot("counting-indices", imported(12));
    let uniform = graph.import_buffer_slot("counting-uniform", imported(256));
    let surface = graph.import_surface_texture_slot(
        "counting-presentable",
        SurfaceTextureContract {
            descriptor: TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: extent[0],
                    height: extent[1],
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: texture_format,
            },
        },
    );
    let pass = graph.add_raster_pass(
        "counting-resident-unlit",
        |pass| {
            let output = pass.color_attachment(
                surface.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let _ = pass.read_buffer(
                &positions.version,
                BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let _ = pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::Whole);
            let _ = pass.read_buffer(&uniform.version, BufferReadUse::Uniform, BufferRange::Whole);
            (output, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.present(pass.output, PresentContract::new());
    let presentable = TextureFormatCapabilities::builder(texture_format)
        .sampled(true, true)
        .storage(false, false)
        .attachments(true, false, vec![1])
        .copies(true, true)
        .build();
    let capabilities = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, false, true, true),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::ImmediateContext,
            false,
        ))
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(false, false, false))
        .texture_format(presentable)
        .surface(SurfaceCapabilities::new(vec![texture_format], true, false))
        .build();
    graph.compile(&capabilities).expect("counting graph").graph
}

/// Resident draws bind token-owned mesh buffers, so repeated resident drawing
/// must grow only the per-frame uniform pair. Vertex/index buffers and their
/// uploads come from the caller's single upload. This is the regression
/// contract for the removed per-draw dead transient buffers.
#[wasm_bindgen_test(async)]
async fn resident_draws_reuse_mesh_buffers_and_only_grow_per_frame_uniforms() {
    let mut session =
        WebGpuSession::new_with_requests(sized_canvas(), Rc::new(CountingBrowserRequests))
            .await
            .expect("instrumented WebGPU session");
    let positions = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0, 1, 2];
    let mesh = session
        .upload_resident_mesh(&positions, &indices)
        .expect("upload resident mesh");
    assert_eq!(
        draw_counter("vertex"),
        1,
        "upload creates one vertex buffer"
    );
    assert_eq!(draw_counter("index"), 1, "upload creates one index buffer");
    assert_eq!(draw_counter("uniform"), 0);
    assert_eq!(draw_counter("writeVertex"), 1);
    assert_eq!(draw_counter("writeIndex"), 1);

    let extent = [8, 8];
    let compiled = resident_graph(session.format(), extent);
    let contract = FixedUnlitGraph::new(&compiled, 1, extent, session.format());
    let draws = [FixedResidentUnlitDraw {
        mesh: &mesh,
        pvm_and_color: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            0.5, 0.25, 1.0,
        ],
        insertion_index: 0,
    }];

    let target_draws = 6;
    let mut submitted = 0;
    let mut turns = 0;
    while submitted < target_draws {
        match session.render_resident(&contract, &draws) {
            Ok(WebGpuRenderOutcome::Submitted(_)) => submitted += 1,
            Ok(WebGpuRenderOutcome::Backpressure) => {
                turn().await;
                turns += 1;
                assert!(turns < 100, "resident completions stalled");
            }
            other => panic!("unexpected resident outcome: {other:?}"),
        }
    }

    assert_eq!(
        draw_counter("vertex"),
        1,
        "resident draws must not create vertex buffers"
    );
    assert_eq!(
        draw_counter("index"),
        1,
        "resident draws must not create index buffers"
    );
    assert_eq!(
        draw_counter("writeVertex"),
        1,
        "positions are never re-uploaded"
    );
    assert_eq!(
        draw_counter("writeIndex"),
        1,
        "indices are never re-uploaded"
    );
    assert_eq!(
        draw_counter("uniform"),
        target_draws,
        "only per-frame uniforms grow"
    );
    assert_eq!(draw_counter("bindGroups"), target_draws);
    assert_eq!(draw_counter("writeUniform"), target_draws);
    dispose(&mut session).await;
}

/// Builds a no-GPU fake device for registry recipes. `fail_creation` and
/// `fail_write` are 1-based step numbers (zero disables injection); the fake
/// counts creations, writes, and destroys so tests can prove that a failed
/// recipe destroys exactly what it created.
fn fake_resident_device(fail_creation: u32, fail_write: u32) -> JsValue {
    js_sys::eval(&format!(
        r#"(() => {{
            const state = {{ creations: 0, writes: 0, destroys: 0 }};
            const inject = (count, limit) => {{
                if (limit > 0 && count === limit) {{
                    throw new Error("injected failure");
                }}
            }};
            const owned = () => ({{
                destroy: () => {{ state.destroys += 1; }},
            }});
            const device = {{
                createBuffer() {{
                    state.creations += 1;
                    inject(state.creations, {fail_creation});
                    return owned();
                }},
                createTexture() {{
                    state.creations += 1;
                    inject(state.creations, {fail_creation});
                    return owned();
                }},
                createBindGroup() {{
                    return {{}};
                }},
                queue: {{
                    writeBuffer() {{
                        state.writes += 1;
                        inject(state.writes, {fail_write});
                    }},
                    writeTexture() {{
                        state.writes += 1;
                        inject(state.writes, {fail_write});
                    }},
                }},
            }};
            device.__state = state;
            return device;
        }})()"#
    ))
    .expect("browser test evals the fake resident device")
}

fn fake_state(device: &JsValue) -> Object {
    let object = device
        .dyn_ref::<Object>()
        .expect("fake device is an object");
    js_sys::Reflect::get(object, &"__state".into())
        .expect("fake device exposes state")
        .dyn_into()
        .expect("state is an object")
}

fn fake_number(state: &Object, name: &str) -> u32 {
    js_sys::Reflect::get(state, &JsValue::from_str(name))
        .expect("state field exists")
        .as_f64()
        .expect("state field is numeric") as u32
}

fn fake_queue(device: &JsValue) -> JsValue {
    let object = device
        .dyn_ref::<Object>()
        .expect("fake device is an object");
    js_sys::Reflect::get(object, &"queue".into()).expect("fake device exposes queue")
}

/// Injected creation/write failures must destroy every created object and
/// never leave a half-updated registry entry: a healthy retry after any
/// injected failure creates a fresh complete set instead of observing a
/// partial entry.
#[wasm_bindgen_test]
fn injected_resident_failures_destroy_created_objects_without_half_updates() {
    let positions = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0, 1, 2];

    for fail_creation in [1, 2] {
        let device = fake_resident_device(fail_creation, 0);
        let queue = fake_queue(&device);
        assert!(
            super::resources::mesh(&device, &queue, &positions, &indices).is_err(),
            "injected creation failure {fail_creation} must reject the upload"
        );
        let state = fake_state(&device);
        assert_eq!(
            fake_number(&state, "destroys"),
            fail_creation - 1,
            "exactly the objects created before the failure are destroyed"
        );
        // The failed attempt left nothing behind: a healthy retry performs the
        // full recipe rather than writing into a half-created entry.
        let good = fake_resident_device(0, 0);
        assert!(super::resources::mesh(&good, &fake_queue(&good), &positions, &indices).is_ok());
        assert_eq!(fake_number(&fake_state(&good), "creations"), 2);
    }

    for fail_write in [1, 2] {
        let device = fake_resident_device(0, fail_write);
        let queue = fake_queue(&device);
        assert!(
            super::resources::mesh(&device, &queue, &positions, &indices).is_err(),
            "injected write failure {fail_write} must reject the upload"
        );
        assert_eq!(
            fake_number(&fake_state(&device), "destroys"),
            2,
            "both buffers created before the write failure are destroyed"
        );
        let good = fake_resident_device(0, 0);
        assert!(super::resources::mesh(&good, &fake_queue(&good), &positions, &indices).is_ok());
        assert_eq!(fake_number(&fake_state(&good), "creations"), 2);
    }

    // A rejected image upload destroys its texture; a healthy retry creates a
    // fresh complete recipe.
    let device = fake_resident_device(0, 1);
    let queue = fake_queue(&device);
    assert!(super::resources::image(&device, &queue, [1, 1], &[1, 2, 3, 4]).is_err());
    assert_eq!(fake_number(&fake_state(&device), "destroys"), 1);
    let good = fake_resident_device(0, 0);
    assert!(super::resources::image(&good, &fake_queue(&good), [1, 1], &[1, 2, 3, 4]).is_ok());
    assert_eq!(fake_number(&fake_state(&good), "creations"), 1);
    assert_eq!(fake_number(&fake_state(&good), "destroys"), 0);
}

/// Without injection the recipe creates exactly what each request asks for:
/// there is no table here to reuse a physical set, and the returned lease is
/// the only owner of what was created.
#[wasm_bindgen_test]
fn resident_recipe_creates_once_per_request_and_the_lease_owns_the_result() {
    let positions = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0, 1, 2];
    let device = fake_resident_device(0, 0);
    let queue = fake_queue(&device);
    let mesh = super::resources::mesh(&device, &queue, &positions, &indices)
        .expect("first upload succeeds");
    let second = super::resources::mesh(&device, &queue, &positions, &indices)
        .expect("second upload succeeds");
    let image = super::resources::image(&device, &queue, [2, 1], &[1, 2, 3, 4, 5, 6, 7, 8])
        .expect("image upload succeeds");
    assert_eq!(fake_number(&fake_state(&device), "creations"), 5);
    assert_eq!(fake_number(&fake_state(&device), "writes"), 3);
    assert_eq!(fake_number(&fake_state(&device), "destroys"), 0);
    assert!(mesh.mesh().is_some());
    assert!(second.mesh().is_some());
    drop(mesh);
    drop(second);
    drop(image);
    assert_eq!(
        fake_number(&fake_state(&device), "destroys"),
        5,
        "dropping the last references destroys both meshes' buffers and the image"
    );
}
