//! Native Surface evidence for the typed external Deferred position producer.

use std::{error::Error, fs, path::Path};

use external_renderer::{BEND_SOURCE, BendFeature, BendInput, ExternalDeferredPipeline};
use fluxel_renderer::{
    CpuPhase, DeferredPositionDeformer, DeformationBinding, ForwardAssets, ForwardResources,
    FramePipeline, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer, prepare_forward,
};
use fluxel_rhi::{
    api::Surface,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};
use fluxel_shader::ShaderComposer;

use super::{COMPANION_EXTENT, asset_json, fixture::Assets};
use crate::{block_on, surface as surface_lifecycle};

const FIRST: [f32; 4] = [0.2, -0.12, 0.0, 0.0];
const SECOND: [f32; 4] = [-0.2, 0.12, 0.0, 0.0];

/// Runs two typed bend inputs through the same Surface and offscreen routes.
#[allow(clippy::too_many_arguments)]
pub(super) fn run<B, Capture, Pump>(
    renderer: &mut Renderer<B>,
    target: &mut Surface<B>,
    surface_extent: [u32; 2],
    volume: RenderBounds,
    output: &Path,
    capture: &Capture,
    pump: &Pump,
    start_frame: usize,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
    Capture: Fn(usize, [u32; 2], [u8; 4]) -> Result<(), Box<dyn Error>>,
    Pump: Fn() -> Result<(), Box<dyn Error>>,
{
    if renderer.pending() != 0 {
        return Err("Surface deformation began with retained Renderer work".into());
    }
    let assets = Assets::new()?;
    let forward_assets = assets.forward();
    let record = assets.record();
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let issuer = scene.feature_issuer()?;
    let feature = BendFeature::new(&issuer)?;
    let mut storage = issuer.storage::<BendInput>();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        surface_extent,
    )?));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        COMPANION_EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let surface_view = receipt.inserted_views()[0];
    let companion_view = receipt.inserted_views()[1];
    let snapshot = scene.snapshot();
    let pipeline = ExternalDeferredPipeline::new();
    let context = issuer.context(&snapshot, CpuPhase::Reference, feature.handle())?;
    let positions = positions(&forward_assets, &record)?;
    let texture = texture(&forward_assets, &record)?;
    let material_evidence = material_evidence(
        &pipeline,
        &snapshot,
        companion_view,
        volume,
        &forward_assets,
    )?;
    fs::write(
        output.join("s7-deferred-surface-deformation-feature.wgsl"),
        BEND_SOURCE,
    )?;
    fs::write(
        output.join("s7-deferred-surface-deformation-positions.bin"),
        geometry_bytes(&forward_assets, &record)?,
    )?;
    fs::write(
        output.join("s7-deferred-surface-deformation-texture.rgba"),
        texture,
    )?;

    for (offset, (name, parameters)) in [("first", FIRST), ("second", SECOND)]
        .into_iter()
        .enumerate()
    {
        let frame = start_frame
            .checked_add(offset)
            .ok_or("Surface deformation frame identifier overflow")?;
        let deformed = deform(positions, parameters);
        let (expected, mask) = image(deformed, texture)?;
        let interior = surface_color(texture, positions, parameters)?;
        write_inputs(
            output,
            name,
            &snapshot,
            &record,
            surface_extent,
            parameters,
            positions,
            texture,
            interior,
            &expected,
            &mask,
            &material_evidence,
        )?;
        let typed = storage.publish(&context, BendInput { parameters })?;
        let image = block_on(target.acquire())?;
        if image.extent() != surface_extent {
            return Err(format!("Surface deformation {name} acquired wrong extent").into());
        }
        let surface = block_on(renderer.submit_deferred_surface_with_deformation(
            &pipeline,
            snapshot.clone(),
            surface_view,
            volume,
            &forward_assets,
            DeformationBinding::new(&issuer, &feature, &typed),
            image,
        ))?;
        let facts = renderer
            .deferred_probe_facts()
            .ok_or("Surface deformation accepted without Graph facts")?;
        if facts.passes != 4 {
            return Err(
                "Surface deformation must order upload, compute, geometry, lighting".into(),
            );
        }
        if renderer.pending() != 1 {
            return Err("Surface deformation did not retain its pending present owner".into());
        }
        fs::write(
            output.join(format!("s7-deferred-surface-deformation-{name}-owners.txt")),
            renderer
                .deferred_probe_diagnostic(&surface)?
                .ok_or("Surface deformation lacks retained Graph owners")?,
        )?;
        block_on(renderer.complete(&surface))?;
        if renderer.pending() != 0 {
            return Err("Surface deformation owner survived successful completion".into());
        }
        capture(frame, surface_extent, interior)?;
        surface_lifecycle::await_visual_release(output, frame, surface_extent, pump)?;

        let offscreen = block_on(renderer.submit_deferred_with_deformation(
            &pipeline,
            snapshot.clone(),
            companion_view,
            volume,
            &forward_assets,
            DeformationBinding::new(&issuer, &feature, &typed),
        ))?;
        if renderer.pending() != 1 {
            return Err("Surface deformation companion did not retain its readback owner".into());
        }
        let raw = block_on(renderer.readback(&offscreen))?;
        if renderer.pending() != 0 {
            return Err("Surface deformation companion owner survived readback".into());
        }
        fs::write(
            output.join(format!(
                "s7-deferred-surface-deformation-{name}-readback.rgba"
            )),
            &raw,
        )?;
        fs::write(
            output.join(format!(
                "s7-deferred-surface-deformation-{name}-readback.ppm"
            )),
            ppm(&raw)?,
        )?;
        verify(&raw, &expected, &mask, name)?;
        fs::write(
            output.join(format!("s7-deferred-surface-deformation-{name}-facts.json")),
            format!(
                concat!(
                    "{{\"schemaVersion\":1,\"graphPasses\":{},\"computeBeforeGeometry\":true,",
                    "\"pending\":[0,1,0,1,0],\"surfaceCapture\":\"surface-client-frame-{}.bmp\",",
                    "\"companionRaw\":\"s7-deferred-surface-deformation-{}-readback.rgba\"}}\n"
                ),
                facts.passes, frame, name,
            ),
        )?;
    }
    Ok(())
}

struct MaterialEvidence {
    source: String,
    source_map: String,
}

fn material_evidence(
    pipeline: &ExternalDeferredPipeline,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    volume: RenderBounds,
    assets: &ForwardAssets,
) -> Result<MaterialEvidence, Box<dyn Error>> {
    let frame = pipeline.prepare(snapshot, view, volume)?;
    let draws = prepare_forward(&frame, assets, &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("Surface deformation expects one prepared material draw".into());
    };
    Ok(MaterialEvidence {
        source: draw.prepared().shader().source().to_owned(),
        source_map: format!(
            "{:#?}\nsource units:\n{:#?}\n",
            draw.prepared().shader().source_map(),
            draw.prepared().material().contribution().source_units()
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn write_inputs(
    output: &Path,
    name: &str,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    surface_extent: [u32; 2],
    parameters: [f32; 4],
    positions: [[f32; 3]; 3],
    texture: [u8; 4],
    surface_interior: [u8; 4],
    expected: &[u8],
    mask: &[u8],
    material: &MaterialEvidence,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-deferred-surface-deformation-{name}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"surfaceExtent\":{:?},",
                "\"companionExtent\":{:?},\"route\":\"deferred-color-position-deformation\",",
                "\"geometryPositions\":{:?},\"textureRgba\":{:?},\"tint\":{:?},",
                "\"parameters\":{:?},\"bendSource\":\"s7-deferred-surface-deformation-feature.wgsl\",",
                "\"surfaceFactor\":0.9,\"expectedSurfaceInteriorRgba\":{:?},",
                "\"triangleRgbTolerance\":1,\"backgroundRgbTolerance\":0,",
                "\"materialSource\":\"s7-deferred-surface-deformation-{}-composed.wgsl\",",
                "\"materialSourceMap\":\"s7-deferred-surface-deformation-{}-source-map.txt\",",
                "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            surface_extent,
            COMPANION_EXTENT,
            positions,
            texture,
            record.resources.tint,
            parameters,
            surface_interior,
            name,
            name,
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture),
        ),
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-deformation-{name}-expected.rgba"
        )),
        expected,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-deformation-{name}-expected.ppm"
        )),
        ppm(expected)?,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-deformation-{name}-stable-mask.bin"
        )),
        mask,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-deformation-{name}-composed.wgsl"
        )),
        &material.source,
    )?;
    fs::write(
        output.join(format!(
            "s7-deferred-surface-deformation-{name}-source-map.txt"
        )),
        &material.source_map,
    )?;
    Ok(())
}

fn geometry_bytes<'a>(
    assets: &'a ForwardAssets,
    record: &RenderRecord<ForwardResources>,
) -> Result<&'a [u8], Box<dyn Error>> {
    assets
        .geometry
        .iter()
        .find(|snapshot| snapshot.id() == record.resources.geometry.id())
        .and_then(|snapshot| snapshot.value().streams.first())
        .map(Vec::as_slice)
        .ok_or_else(|| "Surface deformation geometry snapshot missing".into())
}

fn positions(
    assets: &ForwardAssets,
    record: &RenderRecord<ForwardResources>,
) -> Result<[[f32; 3]; 3], Box<dyn Error>> {
    let bytes = geometry_bytes(assets, record)?;
    if bytes.len() != 36 {
        return Err("Surface deformation requires exactly three packed positions".into());
    }
    let mut result = [[0.0; 3]; 3];
    for (vertex, values) in bytes.chunks_exact(12).enumerate() {
        for (component, bytes) in values.chunks_exact(4).enumerate() {
            result[vertex][component] = f32::from_le_bytes(bytes.try_into()?);
        }
    }
    Ok(result)
}

fn texture(
    assets: &ForwardAssets,
    record: &RenderRecord<ForwardResources>,
) -> Result<[u8; 4], Box<dyn Error>> {
    assets
        .textures
        .iter()
        .find(|snapshot| snapshot.id() == record.resources.texture.id())
        .and_then(|snapshot| <[u8; 4]>::try_from(snapshot.value().rgba8.as_slice()).ok())
        .ok_or_else(|| "Surface deformation requires a 1x1 RGBA texture".into())
}

fn deform(positions: [[f32; 3]; 3], parameters: [f32; 4]) -> [[f32; 3]; 3] {
    positions.map(|[x, y, z]| {
        [
            x + parameters[0] * y * y + parameters[1],
            y,
            z + parameters[2] * x + parameters[3],
        ]
    })
}

fn surface_color(
    texture: [u8; 4],
    positions: [[f32; 3]; 3],
    parameters: [f32; 4],
) -> Result<[u8; 4], Box<dyn Error>> {
    if parameters[2] != 0.0
        || parameters[3] != 0.0
        || positions.iter().any(|position| position[2] != 0.5)
    {
        return Err("Surface deformation Color oracle requires a flat z=0.5 plane".into());
    }
    Ok([
        quantize(f32::from(texture[0]) / 255.0 * 0.9),
        quantize(f32::from(texture[1]) / 255.0 * 0.9),
        quantize(f32::from(texture[2]) / 255.0 * 0.9),
        texture[3],
    ])
}

fn image(positions: [[f32; 3]; 3], texture: [u8; 4]) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let interior = surface_color(texture, positions, [0.0; 4])?;
    let mut expected = vec![0; (COMPANION_EXTENT[0] * COMPANION_EXTENT[1] * 4) as usize];
    let mut mask = vec![0; (COMPANION_EXTENT[0] * COMPANION_EXTENT[1]) as usize];
    for y in 0..COMPANION_EXTENT[1] {
        for x in 0..COMPANION_EXTENT[0] {
            let point = [
                2.0 * (x as f32 + 0.5) / COMPANION_EXTENT[0] as f32 - 1.0,
                1.0 - 2.0 * (y as f32 + 0.5) / COMPANION_EXTENT[1] as f32,
            ];
            let index = (y * COMPANION_EXTENT[0] + x) as usize;
            let edges = edge_distances(positions, point);
            mask[index] = u8::from(edges.iter().all(|distance| *distance > 0.035));
            if barycentric(positions, point)
                .is_some_and(|weights| weights.iter().all(|v| *v >= 0.0))
            {
                expected[index * 4..index * 4 + 4].copy_from_slice(&interior);
            }
        }
    }
    Ok((expected, mask))
}

fn verify(actual: &[u8], expected: &[u8], mask: &[u8], name: &str) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() || expected.len() != mask.len() * 4 {
        return Err("Surface deformation oracle dimensions differ".into());
    }
    let mut interior = false;
    let mut exterior = false;
    for (index, stable) in mask.iter().enumerate() {
        if *stable == 0 {
            continue;
        }
        let wanted = &expected[index * 4..index * 4 + 4];
        let observed = &actual[index * 4..index * 4 + 4];
        interior |= wanted != &[0; 4];
        exterior |= wanted == &[0; 4];
        let rgb_matches = if wanted == &[0; 4] {
            observed[..3] == wanted[..3]
        } else {
            observed[..3]
                .iter()
                .zip(&wanted[..3])
                .all(|(actual, expected)| actual.abs_diff(*expected) <= 1)
        };
        if !rgb_matches || observed[3] != wanted[3] {
            return Err(format!("Surface deformation {name} differs at pixel {index}").into());
        }
    }
    if !interior || !exterior {
        return Err("Surface deformation stable mask lacks interior/exterior coverage".into());
    }
    Ok(())
}

fn ppm(rgba: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    if rgba.len() != (COMPANION_EXTENT[0] * COMPANION_EXTENT[1] * 4) as usize {
        return Err("Surface deformation malformed RGBA".into());
    }
    let mut bytes =
        format!("P6\n{} {}\n255\n", COMPANION_EXTENT[0], COMPANION_EXTENT[1]).into_bytes();
    for pixel in rgba.chunks_exact(4) {
        bytes.extend_from_slice(&pixel[..3]);
    }
    Ok(bytes)
}

fn barycentric(vertices: [[f32; 3]; 3], point: [f32; 2]) -> Option<[f32; 3]> {
    let area = edge(vertices[0], vertices[1], [vertices[2][0], vertices[2][1]]);
    (area != 0.0).then(|| {
        [
            edge(vertices[1], vertices[2], point) / area,
            edge(vertices[2], vertices[0], point) / area,
            edge(vertices[0], vertices[1], point) / area,
        ]
    })
}

fn edge_distances(vertices: [[f32; 3]; 3], point: [f32; 2]) -> [f32; 3] {
    std::array::from_fn(|index| {
        let a = vertices[index];
        let b = vertices[(index + 1) % 3];
        edge(a, b, point).abs() / ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    })
}

fn edge(a: [f32; 3], b: [f32; 3], point: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}

fn quantize(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
