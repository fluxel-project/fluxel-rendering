use std::{error::Error, fs, path::Path};

use fluxel_renderer::DeferredCacheStats;

const EXTENT: [u32; 2] = [96, 96];

pub(super) fn stats_json(stats: DeferredCacheStats) -> String {
    format!(
        "{{\"geometryHits\":{},\"textureHits\":{},\"gbufferProgramHits\":{},\"lightingProgramHits\":{},\"graphHits\":{},\"frameReuses\":{},\"residentUploadBytes\":{}}}",
        stats.geometry_hits,
        stats.texture_hits,
        stats.gbuffer_program_hits,
        stats.lighting_program_hits,
        stats.graph_hits,
        stats.frame_reuses,
        stats.resident_upload_bytes,
    )
}

pub(super) fn expected(color: [u8; 4]) -> (Vec<u8>, Vec<u8>) {
    let mut image = vec![0; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let mut mask = vec![0; (EXTENT[0] * EXTENT[1]) as usize];
    let shaded = [
        (f32::from(color[0]) * 0.9).round() as u8,
        (f32::from(color[1]) * 0.9).round() as u8,
        (f32::from(color[2]) * 0.9).round() as u8,
        color[3],
    ];
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let point = [
                2.0 * (x as f32 + 0.5) / EXTENT[0] as f32 - 1.0,
                1.0 - 2.0 * (y as f32 + 0.5) / EXTENT[1] as f32,
            ];
            let vertices = [[-0.62, -0.56], [0.58, -0.42], [-0.12, 0.66]];
            let edge = |a: [f32; 2], b: [f32; 2]| {
                (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
            };
            let inside =
                (0..3).all(|index| edge(vertices[index], vertices[(index + 1) % 3]) >= 0.0);
            let index = (y * EXTENT[0] + x) as usize;
            if inside {
                image[index * 4..index * 4 + 4].copy_from_slice(&shaded);
            }
            if inside && point[0] > -0.5 && point[0] < 0.4 && point[1] > -0.35 && point[1] < 0.45 {
                mask[index] = 1;
            }
        }
    }
    (image, mask)
}

pub(super) fn verify(actual: &[u8], color: [u8; 4]) -> Result<(), Box<dyn Error>> {
    let (expected, mask) = expected(color);
    if actual.len() != expected.len() {
        return Err("Deferred readback size differs from oracle".into());
    }
    for (index, stable) in mask.iter().enumerate() {
        if *stable != 0
            && actual[index * 4..index * 4 + 3]
                .iter()
                .zip(&expected[index * 4..index * 4 + 3])
                .any(|(actual, expected)| actual.abs_diff(*expected) > 1)
        {
            return Err("Deferred readback differs from independent interior oracle".into());
        }
    }
    Ok(())
}

pub(super) fn write_manifest(output: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("s7-deferred-device-isolation-workload.json"),
        "{\"schemaVersion\":1,\"workload\":\"s7-deferred-device-isolation\",\"sharedArtifacts\":[\"source-position.bin\",\"source-geometry.txt\",\"source-material.txt\",\"source-first.rgba\",\"source-replaced.rgba\",\"source-identities.json\",\"old-pending-owners.txt\",\"old-pending-completed.rgba\",\"device-isolation-facts.json\"],\"cases\":[{\"name\":\"old-cold\",\"artifacts\":[\"old-cold-input.json\",\"old-cold-expected.rgba\",\"old-cold-mask.bin\",\"old-cold-owners.txt\",\"old-cold-actual.rgba\",\"old-cold-facts.json\"]},{\"name\":\"old-warm\",\"artifacts\":[\"old-warm-input.json\",\"old-warm-expected.rgba\",\"old-warm-mask.bin\",\"old-warm-owners.txt\",\"old-warm-actual.rgba\",\"old-warm-facts.json\"]},{\"name\":\"new-cold\",\"artifacts\":[\"new-cold-input.json\",\"new-cold-expected.rgba\",\"new-cold-mask.bin\",\"new-cold-owners.txt\",\"new-cold-actual.rgba\",\"new-cold-facts.json\"]},{\"name\":\"new-warm\",\"artifacts\":[\"new-warm-input.json\",\"new-warm-expected.rgba\",\"new-warm-mask.bin\",\"new-warm-owners.txt\",\"new-warm-actual.rgba\",\"new-warm-facts.json\"]},{\"name\":\"old-return\",\"artifacts\":[\"old-return-input.json\",\"old-return-expected.rgba\",\"old-return-mask.bin\",\"old-return-owners.txt\",\"old-return-actual.rgba\",\"old-return-facts.json\"]},{\"name\":\"foreign-submission-rejected\",\"artifacts\":[\"foreign-submission-rejected-input.json\",\"foreign-submission-rejected.json\",\"foreign-submission-rejected-facts.json\"]},{\"name\":\"new-replacement\",\"artifacts\":[\"new-replacement-input.json\",\"new-replacement-expected.rgba\",\"new-replacement-mask.bin\",\"new-replacement-owners.txt\",\"new-replacement-actual.rgba\",\"new-replacement-facts.json\"]},{\"name\":\"old-replacement\",\"artifacts\":[\"old-replacement-input.json\",\"old-replacement-expected.rgba\",\"old-replacement-mask.bin\",\"old-replacement-owners.txt\",\"old-replacement-actual.rgba\",\"old-replacement-facts.json\"]}]}\n",
    )?;
    Ok(())
}
