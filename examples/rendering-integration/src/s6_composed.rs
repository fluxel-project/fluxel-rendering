//! S6 composed offscreen workload. Surface is deliberately a sibling case.
use crate::block_on;
use external_renderer::{ExternalComposedPipeline, QuadFeature, QuadInput};
use fluxel_renderer::{
    CpuPhase, FeatureBinding, ForwardAssets, ForwardResources, RenderBounds, RenderChange,
    RenderChangeSet, RenderFeature, RenderScene, RenderView, Renderer, RendererError,
};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use std::{error::Error, fs, path::Path};
mod evidence;
mod fixtures;
mod observed;
mod oracle;
mod surface;
use evidence::write_frame_inputs;
use fixtures::*;
use observed::ObservedQuadFeature;
#[cfg(all(windows, feature = "dx12"))]
pub(crate) use surface::run_dx12 as run_surface_dx12;
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) use surface::run_wgl as run_surface_wgl;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<()>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;
    let a = Assets::new()?;
    a.write(output)?;
    let mut scene = RenderScene::new()?;
    let issuer = scene.feature_issuer()?;
    let mut feature = ObservedQuadFeature::new(QuadFeature::new(&issuer)?);
    let mut storage = issuer.storage::<QuadInput>();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(a.record(false)));
    changes.push(RenderChange::InsertView(RenderView::new(
        fluxel_renderer::RenderTransform::identity(),
        oracle::EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let id = receipt.inserted()[0];
    let view = receipt.inserted_views()[0];
    let p = ExternalComposedPipeline::new();
    let mut r = Renderer::new(device)?;
    let vol = RenderBounds::new([-1.; 3], [1.; 3])?;
    // Original same-frame order and warm-cache proof.
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, -10),
        "frame-0",
        Some(MESH),
        Some(QUAD),
        false,
        output,
        true,
    )?;
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "frame-1",
        Some(MESH),
        Some(QUAD),
        true,
        output,
        true,
    )?;
    let plans = r.compilations();
    let warm_before = r.forward_cache_stats();
    let callbacks_before = feature.callback_counts();
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "frame-2",
        Some(MESH),
        Some(QUAD),
        true,
        output,
        true,
    )?;
    if r.compilations() != plans {
        return Err("S6 warm frame did not reuse Graph plan".into());
    }
    let warm_after = r.forward_cache_stats();
    let callbacks_after = feature.callback_counts();
    fs::write(
        output.join("s6-composed-cache.json"),
        format!(
            "{{\"warmGeometryHitsBefore\":{},\"warmGeometryHitsAfter\":{},\"warmTextureHitsBefore\":{},\"warmTextureHitsAfter\":{},\"warmPlanCompilationsBefore\":{},\"warmPlanCompilationsAfter\":{},\"warmPrepareCallbacksBefore\":{},\"warmPrepareCallbacksAfter\":{},\"warmContributeCallbacksBefore\":{},\"warmContributeCallbacksAfter\":{}}}\n",
            warm_before.geometry_hits,
            warm_after.geometry_hits,
            warm_before.texture_hits,
            warm_after.texture_hits,
            plans,
            r.compilations(),
            callbacks_before.prepare,
            callbacks_after.prepare,
            callbacks_before.contribute,
            callbacks_after.contribute
        ),
    )?;
    // Runtime state changes retain the same typed shape and still declare feature reads.
    feature.set_enabled(true);
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "toggle-enabled",
        Some(MESH),
        Some(QUAD),
        true,
        output,
        true,
    )?;
    feature.set_enabled(false);
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "toggle-disabled",
        Some(MESH),
        None,
        true,
        output,
        true,
    )?;
    feature.set_enabled(true);
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "toggle-reenabled",
        Some(MESH),
        Some(QUAD),
        true,
        output,
        true,
    )?;
    // Built-in BoundsVisibility and QuadFeature's own footprint filter disagree deliberately.
    let mut c = RenderChangeSet::new();
    c.push(RenderChange::Update(id, a.record(true)));
    scene.apply(&c)?;
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(false, 10),
        "provider-marker-only",
        None,
        Some(QUAD),
        true,
        output,
        true,
    )?;
    let mut c = RenderChangeSet::new();
    c.push(RenderChange::Update(id, a.record(false)));
    scene.apply(&c)?;
    frame(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        view,
        vol,
        &a.forward(),
        a.input(true, 10),
        "provider-mesh-only",
        Some(MESH),
        None,
        true,
        output,
        true,
    )?;
    pending(
        &mut r,
        &p,
        &issuer,
        &feature,
        &mut storage,
        &mut scene,
        id,
        view,
        vol,
        a,
        output,
    )?;
    let stats = r.forward_cache_stats();
    fs::write(
        output.join("s6-composed-measurements.json"),
        format!(
            "{{\"schemaVersion\":1,\"preparationElapsedNs\":{},\"graphCompileElapsedNs\":{},\"acceptedBufferAllocations\":{},\"acceptedBufferAllocationBytes\":{},\"acceptedTextureAllocations\":{},\"residentCopyBytes\":{},\"frameUploadBytes\":{},\"geometryHits\":{},\"textureHits\":{},\"frameReuses\":{},\"opaqueCandidates\":{},\"opaqueBatches\":{},\"opaqueDraws\":{},\"pendingPeak\":{},\"acceptedSubmissions\":11,\"compilations\":{},\"preparationBoundary\":\"accepted preparation excludes Graph execution, GPU completion and readback\"}}\n",
            stats.preparation_elapsed_ns,
            stats.graph_compile_elapsed_ns,
            stats.accepted_buffer_allocations,
            stats.accepted_buffer_allocation_bytes,
            stats.accepted_texture_allocations,
            stats.resident_copy_bytes,
            stats.frame_upload_bytes,
            stats.geometry_hits,
            stats.texture_hits,
            stats.frame_reuses,
            stats.opaque_candidates,
            stats.opaque_batches,
            stats.opaque_draws,
            stats.pending_peak,
            r.compilations()
        ),
    )?;
    fs::write(output.join("s6-composed-workload.json"), manifest())?;
    Ok(())
}
fn frame<B>(
    r: &mut Renderer<B>,
    p: &ExternalComposedPipeline,
    i: &fluxel_renderer::FeatureIssuer,
    f: &ObservedQuadFeature,
    s: &mut fluxel_renderer::TypedFeatureStorage<QuadInput>,
    snap: fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    vol: RenderBounds,
    a: &ForwardAssets,
    input: QuadInput,
    name: &str,
    mesh: Option<[u8; 4]>,
    quad: Option<[u8; 4]>,
    after: bool,
    o: &Path,
    readback: bool,
) -> Result<Option<fluxel_renderer::Submission>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let e = if name.starts_with("pending-new") {
        oracle::image_with_quad_half(mesh, quad, after, 0.275)
    } else {
        oracle::image(mesh, quad, after)
    };
    let ctx = i.context(&snap, CpuPhase::Reference, f.handle())?;
    let typed = s.publish(&ctx, input)?;
    write_frame_inputs(
        o,
        name,
        &snap,
        typed.scene_version(),
        typed.data(),
        f.is_enabled(),
        vol,
        mesh.is_some(),
        quad.is_some(),
        &e,
    )?;
    let before = f.callback_counts();
    let sub = block_on(r.submit_forward_with_feature(
        p,
        snap,
        view,
        vol,
        a,
        FeatureBinding::new(i, f, &typed),
    ))?;
    let now = f.callback_counts();
    if now.prepare != before.prepare + 1 || now.contribute != before.contribute + 1 {
        return Err("S6 current feature callbacks were not invoked once".into());
    }
    if readback {
        read(r, &sub, &e, name, o)?;
        Ok(None)
    } else {
        Ok(Some(sub))
    }
}
fn pending<B>(
    r: &mut Renderer<B>,
    p: &ExternalComposedPipeline,
    i: &fluxel_renderer::FeatureIssuer,
    f: &ObservedQuadFeature,
    s: &mut fluxel_renderer::TypedFeatureStorage<QuadInput>,
    scene: &mut RenderScene<ForwardResources>,
    id: fluxel_renderer::RenderId,
    view: fluxel_renderer::RenderViewId,
    vol: RenderBounds,
    a: Assets,
    o: &Path,
) -> Result<()>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let old = a.forward();
    let oe = oracle::image(Some(MESH), Some(QUAD), true);
    let one = frame(
        r,
        p,
        i,
        f,
        s,
        scene.snapshot(),
        view,
        vol,
        &old,
        a.input(false, 10),
        "pending-old",
        Some(MESH),
        Some(QUAD),
        true,
        o,
        false,
    )?
    .ok_or("missing pending old")?;
    let g = replace_geometry(&a.geometry, &a.quad_g, small_quad())?;
    let m = replace_material(&a.material, &a.quad_m, replacement_quad_material())?;
    let t = replace_texture(
        &a.texture,
        &a.quad_t,
        fluxel_renderer::ColorTextureData {
            extent: [1, 1],
            rgba8: [255, 255, 0, 255].to_vec(),
        },
    )?;
    if g.id() != a.quad_g.id()
        || m.id() != a.quad_m.id()
        || t.id() != a.quad_t.id()
        || g.generation() <= a.quad_g.generation()
        || m.generation() <= a.quad_m.generation()
        || t.generation() <= a.quad_t.generation()
    {
        return Err("S6 replacement identity/generation mismatch".into());
    }
    fs::write(
        o.join("s6-pending-assets.json"),
        format!(
            "{{\"old\":{{\"geometry\":{{\"id\":{},\"generation\":{:?}}},\"material\":{{\"id\":{},\"generation\":{:?}}},\"texture\":{{\"id\":{},\"generation\":{:?}}}}},\"new\":{{\"geometry\":{{\"id\":{},\"generation\":{:?}}},\"material\":{{\"id\":{},\"generation\":{:?}}},\"texture\":{{\"id\":{},\"generation\":{:?}}}}}}}\n",
            a.quad_g.id().slot(),
            a.quad_g.generation().get(),
            a.quad_m.id().slot(),
            a.quad_m.generation().get(),
            a.quad_t.id().slot(),
            a.quad_t.generation().get(),
            g.id().slot(),
            g.generation().get(),
            m.id().slot(),
            m.generation().get(),
            t.id().slot(),
            t.generation().get()
        ),
    )?;
    for (index, bytes) in g.value().streams.iter().enumerate() {
        fs::write(
            o.join(format!("s6-pending-new-quad-geometry-stream-{index}.bin")),
            bytes,
        )?;
    }
    fs::write(
        o.join("s6-pending-new-quad-geometry.txt"),
        format!("{:#?}\n", g.value()),
    )?;
    fs::write(
        o.join("s6-pending-new-quad-material.txt"),
        format!("{:#?}\n", m.value()),
    )?;
    fs::write(o.join("s6-pending-new-quad-texture.rgba"), &t.value().rgba8)?;
    let new = ForwardAssets {
        geometry: vec![a.mesh_g.clone(), g.clone()],
        materials: vec![a.mesh_m.clone(), m.clone()],
        textures: vec![a.mesh_t.clone(), t.clone()],
    };
    let mut c = RenderChangeSet::new();
    c.push(RenderChange::Update(
        id,
        fluxel_renderer::RenderRecord {
            resources: ForwardResources {
                // Advance the scene version without turning the built-in mesh
                // into the feature's replacement quad product.
                geometry: fluxel_renderer::SceneAssetRef::from_snapshot(&a.mesh_g),
                material: fluxel_renderer::SceneAssetRef::from_snapshot(&a.mesh_m),
                texture: fluxel_renderer::SceneAssetRef::from_snapshot(&a.mesh_t),
                tint: [1.; 4],
            },
            transform: fluxel_renderer::RenderTransform::identity(),
            bounds: RenderBounds::new([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0])?,
        },
    ));
    scene.apply(&c)?;
    let ne = oracle::image_with_quad_half(Some(MESH), Some([191, 64, 0, 255]), true, 0.275);
    let two = frame(
        r,
        p,
        i,
        f,
        s,
        scene.snapshot(),
        view,
        vol,
        &new,
        a.input_with(&g, &m, &t, 10),
        "pending-new",
        Some(MESH),
        Some([191, 64, 0, 255]),
        true,
        o,
        false,
    )?
    .ok_or("missing pending new")?;
    let three = frame(
        r,
        p,
        i,
        f,
        s,
        scene.snapshot(),
        view,
        vol,
        &new,
        a.input_with(&g, &m, &t, 10),
        "pending-new-warm",
        Some(MESH),
        Some([191, 64, 0, 255]),
        true,
        o,
        false,
    )?
    .ok_or("missing pending warm")?;
    if r.pending() != 3 {
        return Err("S6 did not retain three submissions".into());
    }
    // Drop application-owned old feature generations before retirement; the
    // submitted frames must retain every pending owner they still need.
    drop(old);
    drop(a);
    let callbacks_before_backpressure = f.callback_counts();
    if !matches!(
        block_on(r.submit_forward_with_feature(
            p,
            scene.snapshot(),
            view,
            vol,
            &new,
            FeatureBinding::new(i, f, s.current().ok_or("missing typed feature")?)
        )),
        Err(RendererError::Backpressure)
    ) {
        return Err("S6 fourth submission was not Backpressure".into());
    }
    if f.callback_counts().prepare != callbacks_before_backpressure.prepare
        || f.callback_counts().contribute != callbacks_before_backpressure.contribute
    {
        return Err("S6 Backpressure submission invoked feature callbacks".into());
    }
    read(r, &one, &oe, "pending-old", o)?;
    let pending_after_old = r.pending();
    if pending_after_old != 2 {
        return Err("S6 old pending retirement count was not two".into());
    }
    read(r, &two, &ne, "pending-new", o)?;
    let pending_after_new = r.pending();
    if pending_after_new != 1 {
        return Err("S6 new pending retirement count was not one".into());
    }
    read(r, &three, &ne, "pending-new-warm", o)?;
    let pending_after_warm = r.pending();
    if pending_after_warm != 0 {
        return Err("S6 pending submissions did not retire".into());
    }
    fs::write(
        o.join("s6-pending-lifecycle.json"),
        format!("{{\"schemaVersion\":1,\"pendingAfterThree\":{},\"fourthSubmission\":\"Backpressure\",\"callbacksUnchangedOnBackpressure\":true,\"retirements\":[{{\"case\":\"pending-old\",\"pendingAfter\":{}}},{{\"case\":\"pending-new\",\"pendingAfter\":{}}},{{\"case\":\"pending-new-warm\",\"pendingAfter\":{}}}]}}\n", 3, pending_after_old, pending_after_new, pending_after_warm),
    )?;
    Ok(())
}
fn read<B>(
    r: &mut Renderer<B>,
    s: &fluxel_renderer::Submission,
    e: &[u8],
    n: &str,
    o: &Path,
) -> Result<()>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let a = block_on(r.readback(s))?;
    fs::write(o.join(format!("s6-{n}-actual.rgba")), &a)?;
    fs::write(o.join(format!("s6-{n}-actual.ppm")), oracle::ppm(&a)?)?;
    oracle::verify(&a, e, &oracle::mask(), n)
}
fn manifest() -> String {
    let artifacts = |names: &[&str]| -> String {
        names
            .iter()
            .flat_map(|name| {
                [
                    format!("\"s6-{name}-input.json\""),
                    format!("\"s6-{name}-expected.rgba\""),
                    format!("\"s6-{name}-expected.ppm\""),
                    format!("\"s6-{name}-actual.rgba\""),
                    format!("\"s6-{name}-actual.ppm\""),
                ]
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let original = format!(
        "\"s6-initial-mesh-geometry.txt\",\"s6-initial-mesh-geometry-stream-0.bin\",\"s6-initial-mesh-material.txt\",\"s6-initial-mesh-texture.rgba\",\"s6-initial-quad-geometry.txt\",\"s6-initial-quad-geometry-stream-0.bin\",\"s6-initial-quad-material.txt\",\"s6-initial-quad-texture.rgba\",\"s6-stable-mask.bin\",\"s6-composed-cache.json\",\"s6-composed-measurements.json\",{}",
        artifacts(&["frame-0", "frame-1", "frame-2"])
    );
    format!(
        "{{\"schemaVersion\":1,\"workload\":\"s6-composed-offscreen-typed-quad\",\"cases\":[{{\"name\":\"original-three-frame-cache\",\"artifacts\":[{original}]}},{{\"name\":\"enabled-disabled-enabled\",\"artifacts\":[{}]}},{{\"name\":\"independent-providers\",\"artifacts\":[{}]}},{{\"name\":\"three-pending-replacements\",\"artifacts\":[\"s6-pending-assets.json\",\"s6-pending-lifecycle.json\",\"s6-pending-new-quad-geometry-stream-0.bin\",\"s6-pending-new-quad-geometry.txt\",\"s6-pending-new-quad-material.txt\",\"s6-pending-new-quad-texture.rgba\",{}]}}]}}\n",
        artifacts(&["toggle-enabled", "toggle-disabled", "toggle-reenabled"]),
        artifacts(&["provider-marker-only", "provider-mesh-only"]),
        artifacts(&["pending-old", "pending-new", "pending-new-warm"])
    )
}
