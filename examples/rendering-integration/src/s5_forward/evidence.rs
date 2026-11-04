//! Artifact writing and image checks for the multi-frame Forward workload.

use std::{error::Error, fs, path::Path};

use fluxel_renderer::Renderer;
use fluxel_rhi::backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend};

use super::super::{CLEAR, EXTENT, paint_triangle, ppm, triangle_pixels, verify_stable_image};

pub(super) fn frame_expected(visible: [bool; 4]) -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(EXTENT[0] * EXTENT[1] * 4).expect("fixed extent")];
    for pixel in bytes.chunks_exact_mut(4) {
        pixel.copy_from_slice(&CLEAR);
    }
    for ((triangle, color), visible) in triangle_pixels()
        .into_iter()
        .zip([
            [255, 0, 0, 255],
            [255, 0, 255, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ])
        .zip(visible)
    {
        if visible {
            paint_triangle(&mut bytes, triangle, color);
        }
    }
    bytes
}

pub(super) fn write_frame_inputs(
    output: &Path,
    frame: usize,
    scene_version: u64,
    first_clipped: bool,
    second_tint: [f32; 4],
    expected: &[u8],
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s5-forward-frame-{frame}-input.json")),
        format!(
            concat!(
                "{{\n",
                "  \"sceneVersion\": {},\n",
                "  \"firstMeshClipTranslationZ\": {},\n",
                "  \"secondMeshTint\": [{}, {}, {}, {}]\n",
                "}}\n"
            ),
            scene_version,
            if first_clipped { 2 } else { 0 },
            second_tint[0],
            second_tint[1],
            second_tint[2],
            second_tint[3],
        ),
    )?;
    fs::write(
        output.join(format!("s5-forward-frame-{frame}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s5-forward-frame-{frame}-expected.ppm")),
        ppm(expected)?,
    )?;
    Ok(())
}

pub(super) fn read_and_verify<B>(
    renderer: &mut Renderer<B>,
    submission: &fluxel_renderer::Submission,
    output: &Path,
    frame: usize,
    expected: &[u8],
    mask: &[u8],
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    let actual = crate::block_on(renderer.readback(submission))?;
    fs::write(
        output.join(format!("s5-forward-frame-{frame}-actual.rgba")),
        &actual,
    )?;
    fs::write(
        output.join(format!("s5-forward-frame-{frame}-actual.ppm")),
        ppm(&actual)?,
    )?;
    verify_stable_image(&actual, expected, mask)
}

pub(super) fn read_replacement<B>(
    renderer: &mut Renderer<B>,
    submission: &fluxel_renderer::Submission,
    output: &Path,
    name: &str,
    expected: &[u8],
    mask: &[u8],
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    let actual = crate::block_on(renderer.readback(submission))?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-actual.rgba")),
        &actual,
    )?;
    fs::write(
        output.join(format!("s5-forward-replacement-{name}-actual.ppm")),
        ppm(&actual)?,
    )?;
    verify_stable_image(&actual, expected, mask)
}
