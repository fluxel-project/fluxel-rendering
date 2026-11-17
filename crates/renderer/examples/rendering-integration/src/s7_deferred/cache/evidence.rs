//! Source inputs, shader products and cache-result serialization.

use super::*;

pub(super) struct ShaderEvidence {
    variant_key: String,
    source_units: String,
}

pub(super) fn write_shader_route(
    output: &Path,
    name: &str,
    pipeline: &ExternalDeferredPipeline,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    visibility: RenderBounds,
    assets: &ForwardAssets,
) -> Result<ShaderEvidence, Box<dyn Error>> {
    let prepared = pipeline.prepare(snapshot, view, visibility)?;
    let draws = prepare_forward(&prepared, assets, &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("D1 cache source evidence requires exactly one prepared draw".into());
    };
    let shader = draw.prepared().shader();
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-composed.wgsl")),
        shader.source(),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-source-map.txt")),
        format!("{:#?}\n", shader.source_map()),
    )?;
    let material = draw.prepared().material();
    Ok(ShaderEvidence {
        variant_key: material.key().as_str().into(),
        source_units: format!("{:#?}", material.contribution().source_units()),
    })
}

pub(super) fn write_inputs(
    output: &Path,
    name: &str,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    assets: &ForwardAssets,
    shader: &ShaderEvidence,
    expected: &[u8],
    mask: &[u8],
    color: [u8; 4],
) -> Result<(), Box<dyn Error>> {
    let geometry = assets
        .geometry
        .first()
        .ok_or("D1 cache input lacks submitted geometry")?;
    let material = assets
        .materials
        .first()
        .ok_or("D1 cache input lacks submitted material")?;
    let texture = assets
        .textures
        .first()
        .ok_or("D1 cache input lacks submitted texture")?;
    let stream = geometry
        .value()
        .streams
        .first()
        .ok_or("D1 cache geometry lacks its submitted stream")?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"route\":\"deferred\",",
                "\"factorPolicy\":\"analytic geometric normal/depth, independent of WGSL\",",
                "\"expectedAlbedo\":{:?},\"tint\":{:?},\"geometryStreamBytes\":{:?},",
                "\"textureRgba8\":{:?},\"materialVariantKey\":{},\"materialSourceUnits\":{},",
                "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            color,
            record.resources.tint,
            stream,
            texture.value().rgba8,
            json_string(&shader.variant_key),
            json_string(&shader.source_units),
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(crate::named_texture::material_texture_reference(
                &record.resources
            )?),
        ),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-material.txt")),
        format!("{:#?}\n", material.value()),
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-stable-mask.bin")),
        mask,
    )?;
    Ok(())
}

fn json_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

pub(super) fn write_facts(
    output: &Path,
    name: &str,
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    compilations: usize,
    facts: fluxel_renderer::DeferredProbeFacts,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-deferred-cache-{name}-facts.json")),
        format!(
            "{{\"schemaVersion\":1,\"before\":{},\"after\":{},\"compilations\":{},\"graphPasses\":{}}}\n",
            stats_json(before),
            stats_json(after),
            compilations,
            facts.passes,
        ),
    )?;
    Ok(())
}

pub(super) fn write_summary(
    output: &Path,
    before: DeferredCacheStats,
    after: DeferredCacheStats,
    before_compilations: usize,
    after_compilations: usize,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("s7-deferred-cache-facts.json"),
        format!(
            "{{\"schemaVersion\":1,\"start\":{},\"end\":{},\"compilationsBefore\":{},\"compilationsAfter\":{},\"frames\":[\"cold\",\"warm-state\",\"warm-settled\",\"tint\",\"texture-replacement\",\"source-replacement\"]}}\n",
            stats_json(before),
            stats_json(after),
            before_compilations,
            after_compilations,
        ),
    )?;
    Ok(())
}

fn stats_json(stats: DeferredCacheStats) -> String {
    format!(
        "{{\"geometryHits\":{},\"textureHits\":{},\"gbufferProgramHits\":{},\"lightingProgramHits\":{},\"graphHits\":{},\"frameReuses\":{},\"bufferAllocations\":{},\"textureAllocations\":{},\"residentUploadBytes\":{},\"frameUploadBytes\":{}}}",
        stats.geometry_hits,
        stats.texture_hits,
        stats.gbuffer_program_hits,
        stats.lighting_program_hits,
        stats.graph_hits,
        stats.frame_reuses,
        stats.buffer_allocations,
        stats.texture_allocations,
        stats.resident_upload_bytes,
        stats.frame_upload_bytes,
    )
}
