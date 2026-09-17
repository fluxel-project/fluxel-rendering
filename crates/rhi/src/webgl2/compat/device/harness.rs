//! Driving a real DesktopGl4 context through the compatibility adapter, and
//! reporting what the run cost.
//!
//! # Why this exists, and why it is not [`crate::webgl2::conformance`]
//!
//! `conformance` answers *what is this context*.  It opens one, reads the
//! discovery snapshot, drops it, and returns a projection -- observe and drop in
//! one call, which is the whole reason it could exist before the question "may a
//! caller open a GL-family device?" was answered.  That shape cannot measure
//! anything a *frame* does: a run has to keep the provider stack alive across a
//! compiled graph, an executor, a submission and a completion, and it has to
//! report what the run emitted rather than what the driver claimed about itself.
//!
//! This is the second entry of that family.  It takes the same narrow host
//! protocol (a raw drawable through the standard handle traits, plus the extent
//! the caller says it has) and returns plain data, but its data is a cost: the
//! per-domain tallies Layer 2 keeps, the cache traffic behind them, and the two
//! durations the layer deliberately does not record.  Nothing here is public API
//! and nothing here becomes one -- see the note on `mode` below.
//!
//! # The workload is one pass with a draw loop, and that is a choice
//!
//! A frame's cost is dominated by what it *repeats*: the state a draw re-installs
//! because the previous draw left it alone.  Two thousand separate passes would
//! measure the per-pass setup, which no candidate in the experiment funnel
//! targets, and it would also conflate the adapter with the executor's per-pass
//! scheduling.  So the workload is one raster pass whose recording closure issues
//! `draws` indexed draws over the same pipeline, the same binding set and the
//! same two buffers -- the steady state the grouped-dirty-state and
//! bind-group-identity candidates are about -- and the counters it reports are
//! the ones that say whether that steady state was actually reached.
//!
//! It is not a *representative* workload yet, and it does not claim to be.  What
//! makes a workload representative is variety in the state a frame dirties, and
//! that variety is added when a candidate needs it, by the funnel's own screening
//! step, rather than guessed at here.
//!
//! # The mode is a string, and that is not a style preference
//!
//! The differential this entry exists for is the same frame driven twice, once
//! with Layer 2 allowed to skip a redundant call and once with it forced to emit
//! every one.  That knob is [`ExecutionMode`], and it is `pub(crate)`: which mode
//! a *renderer* runs is not a choice the common contract offers, so it must not
//! appear in a signature an out-of-workspace caller can name.  The caller
//! therefore spells the mode, this module parses it, and the crate-private type
//! stays crate-private.  An unrecognized spelling is refused rather than
//! defaulted, because a differential whose two halves silently ran the same mode
//! is worse than one that failed.

use std::time::Instant;

use fluxel_rendergraph::Extent3d;
use fluxel_rendergraph::{
    AttachmentOps, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferDesc,
    BufferRange, BufferReadUse, BufferUsage, BufferUsageKind, ColorAttachmentDesc,
    ExecutionBackend, ExportTextureContract, ExternalOwnership, FrameBindingError,
    FrameBindingErrorKind, FrameExecutor, FrameInputs, FrameResourceProvider, ImportBufferContract,
    ImportedBuffer, IndexFormat, InitialContents, LoadOp, RasterPipelineId, RenderGraph,
    ResourceAccessState, StoreOp, TextureBindingId, TextureDesc, TextureDimension, TextureFormat,
    TextureRange, WriteCoverage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::object::Recipe;
use super::retention::GlRetentionLease;
use super::{GlCompatibilityDevice, GlObjectRegistry};
use crate::resource::RasterKernel;
use crate::webgl2::api::{
    BufferId, ContextEpoch, ContextStamp, DeviceIdentity, NativeGlProvider, TextureId,
    WglContextSurface,
};
use crate::webgl2::conformance::{self, DesktopGl4ContextReport};
use crate::webgl2::state::{ExecutionMode, GlStateBackend, StateCounters};

/// One domain's tallies, as a report row.
///
/// The four numbers are the ones `DomainCounters` carries, restated as plain
/// data so an out-of-workspace gate can read them without naming a crate-private
/// type.  `requests` is what the frame *asked* for and `emitted` what the layer
/// *did*; the gap between them is skipped work, which is the quantity every
/// optimization candidate is judged on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainTally {
    /// The domain's stable name.
    pub domain: String,
    /// How many times the frame named this domain.
    pub requests: u64,
    /// How many calls the layer emitted for it.
    pub emitted: u64,
    /// How many it skipped as already applied.
    pub skipped: u64,
    /// How many times it had to re-derive state it could not vouch for.
    pub unknown_recoveries: u64,
}

/// What one driven run of the workload cost, together with the context it ran on.
///
/// The context reading is carried rather than left to a second call, and the
/// reason is a driver fact this entry learned by being run: `SetPixelFormat` may
/// be called **once** per window, so a process cannot open two WGL contexts over
/// the same drawable -- the second is refused with `PixelFormatAlreadyConfigured`
/// before anything is measured.  What a caller would otherwise do is observe the
/// context, drop it, then reopen it to drive the frame, and that second open is
/// exactly the call the driver refuses.  So the reading rides along with the run
/// that produced it, and [`crate::test_support::observe_desktop_gl4_context`]
/// stays what it is: the entry for callers who want the reading and no run.
///
/// # Three of the submission tallies are absent, and why
///
/// [`StateCounters::submissions`] carries twelve tallies because the other
/// backends need all twelve.  This family writes **three** of them -- `passes`,
/// `pass_loads` and `pass_stores`, all from its session domain -- and never
/// increments `draws`, `clears`, `presents` or the rest.  A report that carried
/// those would be reporting a field that is zero because nothing writes it, which
/// is a worse answer than no field: it reads as "no draws were emitted".  So they
/// are absent here, and the emitted work is read where it is actually recorded,
/// per domain, as `emitted` against `requests`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesktopGl4DrawReport {
    /// What the context this run drove answered when it was asked.
    pub context: DesktopGl4ContextReport,
    /// The execution mode the run actually used, as it was parsed.
    pub mode: String,
    /// How many draws the caller asked for.
    ///
    /// What was *asked*, not what was counted: this family has no tally of draws
    /// issued, so pairing this with a "draws emitted" would be an invention.  The
    /// load-bearing number next to it is the per-domain `emitted`.
    pub draws_requested: u32,
    /// Raster passes begun.
    pub passes: u64,
    /// Attachments loaded rather than cleared.
    pub pass_loads: u64,
    /// Attachments stored at pass end.
    pub pass_stores: u64,
    /// Derived-cache hits.
    pub cache_hits: u64,
    /// Derived-cache misses.
    pub cache_misses: u64,
    /// Derived objects the cache created.
    pub cache_created: u64,
    /// Derived objects it evicted.
    pub cache_evicted: u64,
    /// Derived objects it still holds.
    pub cache_live_entries: u64,
    /// Bytes those entries account for.
    pub cache_live_bytes: u64,
    /// Steady-state heap allocations made while applying state.
    pub steady_state_allocations: u64,
    /// Bytes copied while applying state.
    pub binding_bytes_copied: u64,
    /// Every domain's tallies, in the layer's own stable order.
    pub domains: Vec<DomainTally>,
    /// How long the executor call took, excluding context creation.
    pub submit_nanos: u64,
    /// How long the whole in-context run took, including object creation.
    pub total_nanos: u64,
    /// The extent the context was opened for.
    pub drawable_extent: [u32; 2],
}

/// Opens a real desktop GL context over `host`'s drawable and drives `draws`
/// indexed draws through the compatibility adapter.
///
/// `mode` is `"optimized"` or `"oracle"`; `draws` must be nonzero.  The context
/// is made current on the calling thread, driven, and destroyed before this
/// returns, so nothing borrowed from `host` outlives it.
///
/// # Errors
///
/// The `Err` is a rendered description, for the reason
/// [`crate::test_support::observe_desktop_gl4_context`] gives: the typed errors
/// here are crate-private, and what a hardware gate needs from a failure is the
/// message.
pub fn drive_desktop_gl4_draws<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
) -> Result<DesktopGl4DrawReport, String>
where
    H: HasWindowHandle + HasDisplayHandle,
{
    let (mode, stamp) = parse_request(mode, identity, draws)?;
    let window = host
        .window_handle()
        .map_err(|error| format!("the host's window handle is not available: {error:?}"))?;
    let display = host
        .display_handle()
        .map_err(|error| format!("the host's display handle is not available: {error:?}"))?;

    // Both handles borrow `host`, and the context that adopts them is dropped
    // before this function returns, so neither outlives the window it names.
    let context = WglContextSurface::open(stamp, window, display, extent)
        .map_err(|error| format!("the WGL context did not open: {error:?}"))?;
    let snapshot = context
        .discover()
        .map_err(|error| format!("the context opened but discovery refused it: {error:?}"))?
        .clone();
    // The context reading is taken here because here is the only place it can be:
    // this context is the only one this process gets over this drawable, so a
    // caller who wants the reading and the run gets both from this call or
    // neither.
    let reading = conformance::report(&snapshot, extent, &context.owner_thread());

    // The whole run happens inside one `with_current`, and it has to: the
    // provider borrows the `glow::Context` for its entire lifetime, so the
    // context has to be current for every call the executor makes, not just for
    // the one that built it.  The callback's own error channel is the context's,
    // so the run's message-shaped failures ride out through the inner `Result`
    // rather than being forced into a driver error they are not.
    let outcome = context.with_current("drive the representative workload", |gl| {
        // SAFETY: `with_current` made this context current on this thread and
        // owns it for the whole call, and `snapshot` is the evidence this exact
        // context produced during its own discovery -- which is the pair of
        // conditions `from_discovered` requires.
        let backend = unsafe { NativeGlProvider::from_discovered(gl, snapshot.clone()) };
        Ok(run(backend, mode, draws, extent, reading))
    });
    match outcome {
        Err(error) => Err(format!(
            "the context refused to run the workload: {error:?}"
        )),
        Ok(inner) => inner,
    }
}

/// Parses the caller's spelling of an execution mode.
/// Everything [`drive_desktop_gl4_draws`] can refuse before it opens anything.
///
/// Split out for the same reason the context is opened last: a request that is
/// going to be refused should be refused while it is still a request.  It also
/// makes the refusals testable without a window, which matters because the rest
/// of this module cannot be.
fn parse_request(
    mode: &str,
    identity: u64,
    draws: u32,
) -> Result<(ExecutionMode, ContextStamp), String> {
    let mode = parse_mode(mode)?;
    if draws == 0 {
        return Err(
            "a workload of zero draws measures only the setup, so it is refused".to_owned(),
        );
    }
    let device = DeviceIdentity::new(identity)
        .ok_or_else(|| "a device identity has to be nonzero".to_owned())?;
    Ok((mode, ContextStamp::new(device, ContextEpoch::INITIAL)))
}

fn parse_mode(mode: &str) -> Result<ExecutionMode, String> {
    match mode {
        "optimized" => Ok(ExecutionMode::Optimized),
        "oracle" => Ok(ExecutionMode::Oracle),
        other => Err(format!(
            "an execution mode is `optimized` or `oracle`, and `{other}` is neither"
        )),
    }
}

/// The two streams one indexed triangle draw reads, as host bytes.
///
/// The same geometry the crate's own frame suites draw, restated here because
/// this module is not `#[cfg(test)]` and therefore cannot reach their fixtures.
/// Three vertices rather than a mesh, because what the workload varies is how
/// many times the draw happens and not how much it covers.
fn geometry() -> (Vec<u8>, Vec<u8>) {
    const POSITIONS: [[f32; 3]; 3] = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
    const INDICES: [u32; 3] = [0, 1, 2];
    let positions = POSITIONS
        .iter()
        .flatten()
        .flat_map(|component| component.to_le_bytes())
        .collect();
    let indices = INDICES
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    (positions, indices)
}

/// The colour attachment the workload renders into: index zero, cleared, stored.
fn cleared() -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        index: 0,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        },
    }
}

/// The off-screen target, one two-dimensional layer of RGBA8.
fn target() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

/// The provider that answers this frame's two imports with objects the adapter
/// created before the frame existed.
///
/// A provider has to answer *per binding*, which is why the two are held as
/// pairs rather than as a list: a frame with two imports whose provider returned
/// whichever object it happened to hold would pass this workload and be wrong the
/// moment the two differ in usage.
struct Geometry {
    vertices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
    indices: (BufferBindingId, BoundBuffer<BufferId, GlRetentionLease>),
}

/// The same resolved buffer, restated so the provider can hand it out.
///
/// The lease is cloned rather than moved, and the clone shares the one inner
/// record: the contract lets a provider answer the same object more than once in
/// a frame, so the object is released when the last holder drops.
fn reissue(
    buffer: &BoundBuffer<BufferId, GlRetentionLease>,
) -> BoundBuffer<BufferId, GlRetentionLease> {
    BoundBuffer {
        device: buffer.device,
        identity: buffer.identity,
        physical: buffer.physical,
        descriptor: buffer.descriptor,
        usage: buffer.usage,
        initial_state: buffer.initial_state,
        lease: buffer.lease.clone(),
    }
}

impl<B: GlStateBackend> FrameResourceProvider<GlCompatibilityDevice<B, super::compute::NoCompute>>
    for Geometry
{
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TextureId, GlRetentionLease>, FrameBindingError> {
        Err(missing(
            FrameBindingErrorKind::MissingTexture,
            format!("this frame imports two buffers and no texture, so {id:?} has no answer"),
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<BufferId, GlRetentionLease>, FrameBindingError> {
        if id == self.vertices.0 {
            return Ok(reissue(&self.vertices.1));
        }
        if id == self.indices.0 {
            return Ok(reissue(&self.indices.1));
        }
        Err(missing(
            FrameBindingErrorKind::MissingBuffer,
            format!("this frame binds two buffers, so {id:?} has no answer"),
        ))
    }
}

/// The refusal this provider answers with for a binding it was not given.
fn missing(kind: FrameBindingErrorKind, detail: String) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail,
    }
}

/// Creates one caller-owned buffer and fills it, before any frame exists.
fn own<B: GlStateBackend>(
    device: &mut GlCompatibilityDevice<B, super::compute::NoCompute>,
    size: u64,
    usage: BufferUsageKind,
    bytes: &[u8],
) -> Result<BoundBuffer<BufferId, GlRetentionLease>, String> {
    let buffer = device
        .create_transient_buffer(BufferDesc { size }, BufferUsage::from_kinds([usage]))
        .map_err(|error| format!("a caller-owned buffer was refused: {error:?}"))?;
    device
        .upload_buffer(buffer.physical, 0, bytes)
        .map_err(|error| format!("the bytes did not reach the buffer: {error:?}"))?;
    Ok(buffer)
}

/// Builds the graph, drives it once, and reads the tallies back.
///
/// Generic over the backend on purpose: the workload is a fact about the graph
/// and the verbs, not about WGL, so the same code drives the mock in this
/// crate's own tests and a real context in the hardware gate.
fn run<B: GlStateBackend>(
    backend: B,
    mode: ExecutionMode,
    draws: u32,
    extent: [u32; 2],
    context: DesktopGl4ContextReport,
) -> Result<DesktopGl4DrawReport, String> {
    let started = Instant::now();
    let (positions, elements) = geometry();
    let (vertex_bytes, index_bytes) = (positions.len() as u64, elements.len() as u64);

    let mut device = GlCompatibilityDevice::with_mode(backend, mode);
    let vertex_buffer = own(
        &mut device,
        vertex_bytes,
        BufferUsageKind::Vertex,
        &positions,
    )?;
    let index_buffer = own(&mut device, index_bytes, BufferUsageKind::Index, &elements)?;

    let mut graph = RenderGraph::new();
    let vertices = imported(&mut graph, "triangle-vertices", vertex_bytes);
    let indices = imported(&mut graph, "triangle-indices", index_bytes);
    let colour = graph.create_texture("colour", target());
    let raster = graph.add_raster_pass(
        "triangle",
        |pass| {
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let indices =
                pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::whole());
            let colour = pass.color_attachment(colour, cleared());
            (colour, (vertices, indices))
        },
        // `move` because the recording closure is required to be `'static`, so
        // the draw count has to be owned by it rather than borrowed from here.
        move |commands, resolver, (vertices, indices), _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            commands.set_vertex_buffer(0, vertices)?;
            commands.set_index_buffer(indices, IndexFormat::Uint32)?;
            let bindings = resolver.resolve_bindings(BindingSetId::new(0), &[], &[])?;
            commands.set_bindings(&bindings)?;
            // The workload: the same draw, repeated.  Nothing between two
            // iterations dirties a domain, so every call after the first is a
            // request an optimized layer is entitled to skip and an oracle layer
            // must emit -- which is exactly the difference the report has to show.
            for _ in 0..draws {
                commands.draw_indexed(0..3, 0, 0..1)?;
            }
            Ok(())
        },
    );
    graph.export_texture(
        raster.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );

    let compiled = graph
        .compile(device.capabilities())
        .map_err(|error| format!("the graph did not compile for this context: {error:?}"))?
        .graph;
    let mut objects: GlObjectRegistry<B, super::compute::NoCompute> = device.object_registry();
    let kernel = RasterKernel::IndexedPositionFloat32x3;
    objects
        .register_raster_pipeline(RasterPipelineId::new(1), kernel)
        .map_err(|error| format!("the pipeline was refused: {error:?}"))?;
    objects.register_bindings(BindingSetId::new(0), Recipe::Raster(kernel));

    let provider = Geometry {
        vertices: (BufferBindingId::new(1), vertex_buffer),
        indices: (BufferBindingId::new(2), index_buffer),
    };
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(vertices.slot, provider.vertices.0);
    inputs.bind_buffer(indices.slot, provider.indices.0);

    let executor = FrameExecutor::new(device);
    let submit_started = Instant::now();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &provider,
            &objects,
        )
        .map_err(|error| format!("the frame did not run: {error:?}"))?;
    let submit_nanos = submit_started.elapsed().as_nanos() as u64;

    let backend = executor.try_backend().ok_or_else(|| {
        "the executor still holds the backend after the frame returned".to_owned()
    })?;
    let report = project(
        backend.counters(),
        context,
        mode,
        draws,
        extent,
        submit_nanos,
    );
    let total_nanos = started.elapsed().as_nanos() as u64;
    drop(backend);
    Ok(DesktopGl4DrawReport {
        total_nanos,
        ..report
    })
}

/// One imported buffer slot, declared the way this adapter's creation verb answers.
fn imported(graph: &mut RenderGraph, name: &str, size: u64) -> ImportedBuffer {
    graph.import_buffer_slot(
        name,
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::Undefined,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    )
}

/// Projects the layer's counters into the report, total time aside.
fn project(
    counters: &StateCounters,
    context: DesktopGl4ContextReport,
    mode: ExecutionMode,
    draws: u32,
    extent: [u32; 2],
    submit_nanos: u64,
) -> DesktopGl4DrawReport {
    let domains = counters
        .report()
        .map(|(domain, counts)| DomainTally {
            domain: domain.name().to_owned(),
            requests: counts.requests,
            emitted: counts.emitted,
            skipped: counts.skipped,
            unknown_recoveries: counts.unknown_recoveries,
        })
        .collect();
    DesktopGl4DrawReport {
        context,
        mode: match mode {
            ExecutionMode::Optimized => "optimized".to_owned(),
            ExecutionMode::Oracle => "oracle".to_owned(),
        },
        draws_requested: draws,
        passes: counters.submissions.passes,
        pass_loads: counters.submissions.pass_loads,
        pass_stores: counters.submissions.pass_stores,
        cache_hits: counters.caches.hits,
        cache_misses: counters.caches.misses,
        cache_created: counters.caches.created,
        cache_evicted: counters.caches.evicted,
        cache_live_entries: counters.caches.live_entries,
        cache_live_bytes: counters.caches.live_bytes,
        steady_state_allocations: counters.steady_state_allocations,
        binding_bytes_copied: counters.binding_bytes_copied,
        domains,
        submit_nanos,
        total_nanos: 0,
        drawable_extent: extent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgl2::api::tests::snapshot;
    use crate::webgl2::api::{GlFamilyProfile, MockGlFamilyApi, OwnerThreadIdentity};
    use crate::webgl2::state::StateDomain;

    /// One run of the workload over the mock backend this crate's own suites use.
    ///
    /// The mock is the right vehicle for exactly the claim under test: what this
    /// module has to get right is *which number goes where*, and that is a fact
    /// about the projection rather than about a driver.  A real context would make
    /// the same assertions harder to read and no more true.
    ///
    /// The reading is projected from the same snapshot the mock is built from,
    /// through the same function the real entry calls, so the mock run exercises
    /// the whole of `run` rather than a reduced version of it.
    fn drive(mode: ExecutionMode, draws: u32) -> DesktopGl4DrawReport {
        let discovery = snapshot(GlFamilyProfile::WebGl2);
        // The test thread owns the mock, so the identity it reports is this
        // thread's -- which is the same fact the real entry records.
        let reading = conformance::report(&discovery, [4, 4], &OwnerThreadIdentity::current());
        run(
            MockGlFamilyApi::from_discovery(discovery),
            mode,
            draws,
            [4, 4],
            reading,
        )
        .unwrap_or_else(|error| {
            panic!("the workload runs over the mock context in {mode:?}: {error}")
        })
    }

    /// A report's per-domain rows, as `(name, requests, emitted, skipped)`.
    fn rows(report: &DesktopGl4DrawReport) -> Vec<(String, u64, u64, u64)> {
        report
            .domains
            .iter()
            .map(|domain| {
                (
                    domain.domain.clone(),
                    domain.requests,
                    domain.emitted,
                    domain.skipped,
                )
            })
            .collect()
    }

    fn sum(rows: &[(String, u64, u64, u64)], pick: fn(&(String, u64, u64, u64)) -> u64) -> u64 {
        rows.iter().map(pick).sum()
    }

    #[test]
    fn the_workload_runs_as_one_pass_whatever_the_mode() {
        for mode in [ExecutionMode::Optimized, ExecutionMode::Oracle] {
            let report = drive(mode, 8);
            assert_eq!(report.passes, 1, "the workload is one pass, in {mode:?}");
            assert_eq!(report.pass_loads, 1, "with one attachment loaded");
            assert_eq!(report.pass_stores, 1, "and stored at the end");
            assert_eq!(report.draws_requested, 8);
            assert_eq!(report.drawable_extent, [4, 4]);
            assert_eq!(
                report.domains.len(),
                StateDomain::COUNT,
                "the report names every domain, not only the ones this frame touched"
            );
        }
    }

    /// A differential is only a differential if both halves complete.
    ///
    /// Found on hardware, not here: the optimized path was the only one that
    /// preserved an invariant the draw verb asserts.  A pipeline install binds
    /// the pipeline's vertex array and records it; under the oracle the next
    /// geometry request re-derives that array and destroys the one it replaced,
    /// so the recorded id is dead by the time the draw re-resolves it, and the
    /// run refuses with `draw-raster / vertex array is not live` -- at *one*
    /// draw, not at some count.  The recorder has to model that refusal or this
    /// test passes for the wrong reason, which is what it did before the
    /// recorder was taught the rule.
    #[test]
    fn the_uncached_path_completes_a_frame_with_more_than_one_draw() {
        for draws in [1, 2, 8] {
            let oracle = drive(ExecutionMode::Oracle, draws);
            assert_eq!(oracle.draws_requested, draws);
            assert_eq!(oracle.passes, 1);
        }
    }

    /// The differential this whole entry exists for.
    ///
    /// The identity a reader might expect -- `oracle.emitted ==
    /// optimized.emitted + optimized.skipped` -- does **not** hold, and this test
    /// says so rather than asserting it.  Measured with eight draws: `optimized`
    /// emits 8 and skips 21, `oracle` emits 51.  The reason is a property of the
    /// layer and not of this harness -- a domain may emit for a reason that is not
    /// a request, so `requests` is not `emitted + skipped` in either mode:
    /// `pipeline` is asked eight times and answers with two emits and seven skips
    /// under one mode and sixteen emits under the other.
    ///
    /// What *is* mode-independent is `requests` itself, and that is the fact
    /// pinned here: the mode decides what the layer does about a request, never
    /// what the frame asks for.  A differential whose two halves were asked
    /// different things would be measuring two workloads.
    #[test]
    fn the_two_modes_are_asked_the_same_thing_and_answer_differently() {
        let optimized = drive(ExecutionMode::Optimized, 8);
        let oracle = drive(ExecutionMode::Oracle, 8);
        let (optimized_rows, oracle_rows) = (rows(&optimized), rows(&oracle));

        let requests = |rows: &[(String, u64, u64, u64)]| -> Vec<(String, u64)> {
            rows.iter()
                .map(|(name, requests, _, _)| (name.clone(), *requests))
                .collect()
        };
        assert_eq!(
            requests(&optimized_rows),
            requests(&oracle_rows),
            "the mode decides what the layer does about a request, never what the frame asks for"
        );

        let skipped = |rows: &[(String, u64, u64, u64)]| sum(rows, |row| row.3);
        let emitted = |rows: &[(String, u64, u64, u64)]| sum(rows, |row| row.2);

        assert!(
            skipped(&optimized_rows) > 0,
            "a repeated draw loop dirties nothing between iterations, so an optimized layer \
             proves some calls redundant -- a zero here would mean the workload had stopped \
             being the steady state the funnel screens against"
        );
        assert_eq!(
            skipped(&oracle_rows),
            0,
            "and the oracle proves nothing redundant, by definition"
        );
        assert!(
            emitted(&oracle_rows) > emitted(&optimized_rows),
            "so the oracle emits strictly more: {} against {}",
            emitted(&oracle_rows),
            emitted(&optimized_rows)
        );
        for (name, _, optimized_emitted, _) in &optimized_rows {
            let oracle_emitted = oracle_rows
                .iter()
                .find(|row| row.0 == *name)
                .map(|row| row.2)
                .expect("both modes report the same domains");
            assert!(
                oracle_emitted >= *optimized_emitted,
                "{name}: a layer that skips nothing cannot emit fewer calls than one that does"
            );
        }
    }

    /// The oracle keeps no derived-cache traffic, which bounds what it can
    /// baseline.
    ///
    /// Pinned rather than discovered later: a candidate judged on cache hits or
    /// created entries has no oracle number to compare against, because the
    /// uncached path never consults those caches.  Such a candidate is screened
    /// against the optimized run's own earlier revision instead, and the funnel
    /// has to say so before it screens one.
    #[test]
    fn the_oracle_reports_no_derived_cache_traffic() {
        let oracle = drive(ExecutionMode::Oracle, 8);
        assert_eq!(
            (
                oracle.cache_hits,
                oracle.cache_misses,
                oracle.cache_created,
                oracle.cache_live_entries
            ),
            (0, 0, 0, 0),
            "the uncached path consults no derived cache, so it has nothing to report"
        );
        let optimized = drive(ExecutionMode::Optimized, 8);
        assert!(
            optimized.cache_hits + optimized.cache_misses + optimized.cache_created > 0,
            "while the optimized path does consult them, which is the asymmetry"
        );
    }

    #[test]
    fn a_mode_is_parsed_or_refused_rather_than_defaulted() {
        assert!(matches!(
            parse_mode("optimized"),
            Ok(ExecutionMode::Optimized)
        ));
        assert!(matches!(parse_mode("oracle"), Ok(ExecutionMode::Oracle)));
        let refused = parse_mode("fast").expect_err("a spelling that is neither is refused");
        assert!(
            refused.contains("fast"),
            "and the refusal names what it got"
        );
    }

    /// Both refusals happen while the request is still a request.
    ///
    /// Checked on `parse_request` rather than the entry because the entry's next
    /// act is to open a context over a real drawable, and a window is the one
    /// thing these two refusals exist to avoid needing: they are decided before
    /// the host is consulted at all.
    #[test]
    fn a_workload_is_refused_before_a_context_is_opened() {
        assert_eq!(
            parse_request("optimized", 1, 0).expect_err("a zero-draw workload measures only setup"),
            "a workload of zero draws measures only the setup, so it is refused"
        );
        assert_eq!(
            parse_request("optimized", 0, 8).expect_err("a zero identity names no device"),
            "a device identity has to be nonzero"
        );
        assert!(
            parse_request("oracle", 7, 8).is_ok(),
            "and a well-formed request is not refused"
        );
    }
}
