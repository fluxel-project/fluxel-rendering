//! The one measured workload, and what driving it costs.
//!
//! # Why this is not inside either entry that calls it
//!
//! `Checkpoint G` asks the same question of two surfaces -- a desktop GL4
//! context on hardware and a live WebGL2 context in a browser -- and the only
//! thing that differs between them is *what opened the context*.  The graph, the
//! verbs, the executor, the counters and the durations are the same on both, and
//! they are the same because they are facts about the adapter rather than about
//! WGL or a canvas.  So the workload lives here, generic over the backend, and
//! `super::harness` and the crate's browser draw test are two callers of one
//! implementation rather than two implementations of one measurement.
//!
//! The alternative -- each surface building its own graph -- would make the two
//! halves of the differential differ in the frame as well as in the context,
//! which is exactly the confound the differential exists to remove.
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
//! # What a caller supplies, and what it gets back
//!
//! A caller supplies a backend that already satisfies
//! [`GlStateBackend`] -- which is to say a provider over a *current* context,
//! because Layer 1's verbs are context-bound and this module never opens one --
//! plus the mode, the draw count, the extent it wants reported, and a clock.
//!
//! # Why the clock is the caller's, and not this module's
//!
//! The two durations this module reports are the only readings a run produces
//! that are not Layer 2 counters, and there is no clock in the layers to read
//! them from: `state` is barred from naming a browser crate, so a clock declared
//! there could not be implemented by the browser provider, and `api/browser` is
//! barred from naming `state`, so it could not implement one anyway.  On the
//! native side there is also nothing to declare -- `std::time::Instant` is a fact
//! about the host, not about a GL context.
//!
//! It is the caller's for the same reason the context reading below is nobody's:
//! a clock is a per-surface fact.  `Instant` is unsupported on
//! `wasm32-unknown-unknown` -- it panics, which is how this was found -- and the
//! browser's monotonic clock is `performance.now()`.  A module that reached for
//! either one itself would be a module that only runs on one of its two surfaces.
//!
//! Back comes [`DrawCost`]: the per-domain tallies, the cache traffic behind
//! them, and the two durations.  It carries no context reading, because the
//! surfaces do not have the same reading to give: the native entry projects a
//! desktop GL4 discovery report and the browser test projects nothing of the
//! kind.  A shared struct with an optional context in it would have been a type
//! that is a different shape on each surface, which is the one thing this module
//! exists to avoid.

use fluxel_rendergraph::{
    AttachmentOps, BindingSetId, BoundBuffer, BoundTexture, BufferBindingId, BufferDesc,
    BufferRange, BufferReadUse, BufferUsage, BufferUsageKind, ColorAttachmentDesc,
    ExecutionBackend, ExportTextureContract, Extent3d, ExternalOwnership, FrameBindingError,
    FrameBindingErrorKind, FrameExecutor, FrameInputs, FrameResourceProvider, ImportBufferContract,
    ImportedBuffer, IndexFormat, InitialContents, LoadOp, RasterPipelineId, RenderGraph,
    ResourceAccessState, StoreOp, TextureBindingId, TextureDesc, TextureDimension, TextureFormat,
    TextureRange, WriteCoverage,
};

use super::object::Recipe;
use super::retention::GlRetentionLease;
use super::{GlCompatibilityDevice, GlObjectRegistry};
use crate::resource::RasterKernel;
use crate::webgl2::api::{BufferId, TextureId};
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

/// What one driven run of [`drive`] cost, apart from what it ran on.
///
/// Every field here is a reading the run itself produced.  The counters are read
/// off Layer 2 after the frame returns rather than sampled during it, because the
/// layer's own report is the durable statement of what it did and a sampler would
/// be a second, weaker one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DrawCost {
    /// The execution mode the run actually used.
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
    /// How long the whole run took, including object creation.
    pub total_nanos: u64,
    /// The extent the caller said the target has.
    pub drawable_extent: [u32; 2],
}

/// Drives the workload over `backend` and reports what it cost.
///
/// `draws` is how many indexed draws the one raster pass issues.  `extent` is
/// reported rather than used: the target this workload renders into is a fixed
/// four-by-four texture, and the extent is the caller's statement about the
/// drawable behind the context, which is a fact this module has no way to read.
///
/// `now_nanos` is the caller's monotonic clock, in nanoseconds, and the module
/// doc says why it is the caller's.  It is called twice for the whole run and
/// twice more around the executor call, so a clock that is expensive to read is
/// read four times and not once per draw.  The two differences are taken with
/// `saturating_sub` rather than `-`: the contract is that the clock is monotonic,
/// and a subtraction that can only be right when the contract holds would turn a
/// violating clock into a panic inside a measurement.
///
/// # Errors
///
/// The `Err` is a rendered description rather than a typed error, for the reason
/// `super::harness` gives: the errors along this path are crate-private types,
/// and what a gate needs from a failure is the message.  A failure here is always
/// a defect or an unsupported context -- never a wrong number, because every
/// field of [`DrawCost`] is read from a counter the layer keeps.
pub(crate) fn drive<B: GlStateBackend>(
    backend: B,
    mode: ExecutionMode,
    draws: u32,
    extent: [u32; 2],
    now_nanos: impl Fn() -> u64,
) -> Result<DrawCost, String> {
    let started = now_nanos();
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
    let submit_started = now_nanos();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &provider,
            &objects,
        )
        .map_err(|error| format!("the frame did not run: {error:?}"))?;
    let submit_nanos = now_nanos().saturating_sub(submit_started);

    let backend = executor.try_backend().ok_or_else(|| {
        "the executor still holds the backend after the frame returned".to_owned()
    })?;
    let cost = project(backend.counters(), mode, draws, extent, submit_nanos);
    let total_nanos = now_nanos().saturating_sub(started);
    // The backend is dropped before the cost is returned so the provider stack
    // the counters came from does not outlive the call: a caller reads numbers,
    // never a device.
    drop(backend);
    Ok(DrawCost {
        total_nanos,
        ..cost
    })
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

/// Projects the layer's counters into the cost, total time aside.
fn project(
    counters: &StateCounters,
    mode: ExecutionMode,
    draws: u32,
    extent: [u32; 2],
    submit_nanos: u64,
) -> DrawCost {
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
    DrawCost {
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
