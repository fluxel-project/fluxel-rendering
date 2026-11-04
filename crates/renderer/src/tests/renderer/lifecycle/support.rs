//! Shared witness fixture and helpers for renderer lifecycle tests.

use super::backend::*;
use super::*;

pub(super) fn fixture(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |_| {})
}

pub(super) fn fixture_with_caps(
    script: Script,
    adjust: impl FnOnce(&mut DeviceCapabilities),
) -> (Device<WitnessBackend>, Arc<Native>) {
    let id = DeviceId::allocate().unwrap();
    let native = Arc::new(Native {
        script,
        fault_enabled: AtomicBool::new(true),
        finishes: AtomicUsize::new(0),
        next_completion: AtomicUsize::new(0),
        command_calls: AtomicUsize::new(0),
        copied_bytes: Mutex::new(Vec::new()),
        copy_destinations: Mutex::new(Vec::new()),
        uniform_offsets: Mutex::new(Vec::new()),
        texture_uploads: Mutex::new(Vec::new()),
        mapped_bytes: Mutex::new(Vec::new()),
        map_calls: AtomicUsize::new(0),
        readback_ranges: Mutex::new(Vec::new()),
        waits: AtomicUsize::new(0),
        idle_waits: AtomicUsize::new(0),
        waited_completions: Mutex::new(Vec::new()),
        created_buffers: AtomicUsize::new(0),
        dropped_buffers: AtomicUsize::new(0),
        outcome: Mutex::new(Weak::new()),
    });
    let mut caps = DeviceCapabilities {
        limits: DeviceLimits {
            max_buffer_bytes: 1 << 20,
            max_texture_dimension_2d: 4096,
            max_texture_array_layers: 1,
            max_color_attachments: 1,
            max_vertex_buffers: 1,
            max_vertex_attributes: 2,
            max_bind_groups: 1,
            max_bindings_per_group: 2,
            uniform_offset_alignment: 256,
            ..DeviceLimits::default()
        },
        lanes: Arc::from([LaneInfo {
            id: LaneId {
                device: id,
                ordinal: 0,
            },
            class: LaneClass::Graphics,
            routes: Vec::new(),
        }]),
        ..DeviceCapabilities::default()
    };
    adjust(&mut caps);
    // SAFETY: native state and capabilities were created together for this ID.
    (
        unsafe { Device::from_native(id, Arc::clone(&native), caps) },
        native,
    )
}

pub(super) fn snapshot() -> (AssetSnapshot<Bytes, Vec<u8>>, Weak<Vec<u8>>) {
    snapshot_bytes(vec![1, 2, 3, 4])
}

pub(super) fn snapshot_bytes(bytes: Vec<u8>) -> (AssetSnapshot<Bytes, Vec<u8>>, Weak<Vec<u8>>) {
    let store = AssetStore::<Bytes, Vec<u8>, ()>::new();
    let handle = store.create().unwrap();
    let size = u64::try_from(bytes.len()).unwrap();
    let snapshot = match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(bytes, ResidentBytes::new(size)).unwrap(),
        _ => unreachable!(),
    };
    let weak = Arc::downgrade(&snapshot.value_arc());
    (snapshot, weak)
}

pub(super) fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: AssetSnapshot<Bytes, Vec<u8>>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit(snapshot))
}

pub(super) fn block_on<F: Future>(future: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => panic!("the deterministic fixture must be ready"),
    }
}

pub(super) fn poll_once_pending<F: Future>(future: F) {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    assert!(matches!(
        future.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
}

pub(super) fn blank_surface_frame(
    device: &Device<WitnessBackend>,
) -> crate::residency::SurfaceForwardFrame<WitnessBackend> {
    let (resident, resident_upload) =
        crate::residency::ForwardResidentBuffers::allocate(device, Vec::new(), Vec::new()).unwrap();
    let uniform_bytes = vec![0; 16];
    let (frame, uniform_upload) = crate::residency::SurfaceForwardFrameBuffers::allocate(
        device,
        16,
        uniform_bytes.clone(),
        0..16,
    )
    .unwrap();
    crate::residency::SurfaceForwardFrame {
        resident,
        resident_upload,
        frame,
        uniform_upload,
        uniform_bytes,
        draws: Vec::new(),
        draw_batches: Vec::new(),
        extent: [1, 1],
        color_format: TextureFormat::Rgba8Unorm,
    }
}

pub(super) fn witness_surface(device: &Device<WitnessBackend>) -> Surface<WitnessBackend> {
    // SAFETY: the witness image has only an atomic generation contract.
    let mut surface = unsafe {
        Surface::from_native(
            device,
            SurfaceId::allocate().expect("surface identity"),
            WitnessSurface {
                generation: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            },
        )
    };
    let config = SurfaceConfig {
        extent: [1, 1],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    block_on(surface.configure(&config)).expect("configure witness surface");
    surface
}

impl Renderer<WitnessBackend> {
    async fn execute_forward<T: Send + Sync + 'static>(
        &mut self,
        retained: T,
        frame: crate::residency::ForwardFrame<WitnessBackend>,
        publication: crate::renderer::forward::ForwardPublication,
    ) -> std::result::Result<Submission, RendererError> {
        let batch_stats = crate::preparation::ForwardBatchStats {
            candidates: frame.draws.len(),
            batches: frame.draw_batches.len(),
        };
        self.execute_forward_with_contribution(
            retained,
            frame,
            publication,
            crate::FrameTopology::OpaqueForward,
            batch_stats,
            |context| context.opaque_forward(),
        )
        .await
    }
}
#[path = "../forward/cache.rs"]
mod forward_cache_tests;
#[path = "../forward/composed.rs"]
mod forward_composed_tests;
#[path = "../forward/contribution.rs"]
mod forward_contribution_tests;
#[path = "../forward/lifecycle.rs"]
mod forward_lifecycle_tests;
#[path = "../forward/pending.rs"]
mod forward_pending_tests;
#[path = "../forward/surface_contribution.rs"]
mod forward_surface_contribution_tests;
#[path = "../raster.rs"]
mod raster_tests;
#[path = "../read_set_raster.rs"]
mod read_set_raster_tests;
#[path = "../shutdown.rs"]
mod shutdown_tests;
#[path = "submission.rs"]
mod submission_tests;
#[path = "../surface.rs"]
mod surface_tests;
