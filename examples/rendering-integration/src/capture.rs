//! Consumer-owned HWND capture for surface fixtures.
//!
//! Pixels come from Windows Graphics Capture, not RHI readback. This module
//! never changes z-order, focus, visibility, or a window style.

use std::{error::Error, fs, path::Path};

use fluxel_host::Window;

#[path = "capture/windows.rs"]
mod windows_capture;

/// Process-scoped WinRT ownership for native window capture.
///
/// The integration entry point creates this before native GPU objects and
/// keeps it alive until they are all dropped. Individual frame captures must
/// not initialize or tear down the apartment mid-renderer lifetime.
pub(crate) struct CaptureRuntime {
    _inner: windows_capture::CaptureRuntime,
}

impl CaptureRuntime {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            _inner: windows_capture::CaptureRuntime::new()?,
        })
    }
}

pub(crate) fn capture_client(
    window: &Window,
    output: &Path,
    frame: usize,
    expected_extent: [u32; 2],
    expected_rgba: [u8; 4],
) -> Result<(), Box<dyn Error>> {
    capture_client_with_tolerance(window, output, frame, expected_extent, expected_rgba, 0)
}

pub(crate) fn capture_client_with_tolerance(
    window: &Window,
    output: &Path,
    frame: usize,
    expected_extent: [u32; 2],
    expected_rgba: [u8; 4],
    triangle_rgb_tolerance: u8,
) -> Result<(), Box<dyn Error>> {
    let captured = windows_capture::capture_window_client(window, expected_extent)?;
    let bmp_name = write_snapshot(output, frame, &captured, Some(triangle_rgb_tolerance))?;
    let verdict = validate_capture_pixels(
        &captured.pixels,
        captured.extent,
        expected_rgba,
        triangle_rgb_tolerance,
    );
    fs::write(
        output.join(format!("surface-client-frame-{frame}.verdict.json")),
        format!(
            "{{\"schemaVersion\":2,\"frame\":{},\"image\":{},\"oracle\":\"window-graphics-capture-corners-exact-and-stable-triangle-interior\",\"triangleRgbTolerance\":{},\"status\":{}}}\n",
            frame,
            crate::json_string(&bmp_name),
            triangle_rgb_tolerance,
            if verdict.is_ok() {
                "\"passed\""
            } else {
                "\"failed\""
            }
        ),
    )?;
    verdict.map_err(|reason| {
        format!("Windows Graphics Capture visual oracle failed; retained {bmp_name}: {reason}")
    })?;
    Ok(())
}

/// Saves a native HWND snapshot for an animated visible demo.
///
/// This records the same independent Windows Graphics Capture evidence as the
/// fixture path, but intentionally has no fixed-color fixture oracle or
/// verdict: the demo's content changes over time.
pub(crate) fn snapshot_client(
    window: &Window,
    output: &Path,
    frame: usize,
    expected_extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    let captured = windows_capture::capture_window_client(window, expected_extent)?;
    let _ = write_snapshot(output, frame, &captured, None)?;
    Ok(())
}

fn write_snapshot(
    output: &Path,
    frame: usize,
    captured: &windows_capture::CapturedClient,
    triangle_rgb_tolerance: Option<u8>,
) -> Result<String, Box<dyn Error>> {
    let bmp_name = format!("surface-client-frame-{frame}.bmp");
    write_bmp(output, &bmp_name, captured.extent, &captured.pixels)?;
    let fixture_oracle = triangle_rgb_tolerance
        .map(|tolerance| format!("  \"triangleRgbTolerance\": {tolerance},\n"))
        .unwrap_or_default();
    fs::write(
        output.join(format!("surface-client-frame-{frame}.capture.json")),
        format!(
            concat!(
                "{{\n  \"schemaVersion\": 2,\n  \"frame\": {},\n  \"clientExtent\": [{}, {}],\n",
                "  \"method\": \"windows-graphics-capture-window\",\n  \"pixelFormat\": \"BGRA8-top-down\",\n  \"image\": {:?},\n",
                "  \"captureContentExtent\": [{}, {}],\n  \"clientCrop\": [{}, {}, {}, {}],\n  \"captureTimeoutMs\": {},\n",
                "  \"windowStateBefore\": {},\n  \"windowStateAfter\": {},\n  \"windowStateUnchanged\": {},\n",
                "{}  \"automaticVisualPass\": false\n}}\n"
            ),
            frame,
            captured.extent[0],
            captured.extent[1],
            bmp_name,
            captured.content_extent[0],
            captured.content_extent[1],
            captured.client_crop[0],
            captured.client_crop[1],
            captured.client_crop[2],
            captured.client_crop[3],
            captured.timeout_ms,
            captured.before.json(),
            captured.after.json(),
            captured.before == captured.after,
            fixture_oracle,
        ),
    )?;
    Ok(bmp_name)
}

fn write_bmp(
    output: &Path,
    name: &str,
    extent: [u32; 2],
    pixels: &[u8],
) -> Result<(), Box<dyn Error>> {
    let width = i32::try_from(extent[0])?;
    let height = i32::try_from(extent[1])?;
    let mut bmp = Vec::with_capacity(54 + pixels.len());
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&u32::try_from(54 + pixels.len())?.to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54_u32.to_le_bytes());
    bmp.extend_from_slice(&40_u32.to_le_bytes());
    bmp.extend_from_slice(&width.to_le_bytes());
    bmp.extend_from_slice(&(-height).to_le_bytes());
    bmp.extend_from_slice(&1_u16.to_le_bytes());
    bmp.extend_from_slice(&32_u16.to_le_bytes());
    bmp.extend_from_slice(&0_u32.to_le_bytes());
    bmp.extend_from_slice(&u32::try_from(pixels.len())?.to_le_bytes());
    bmp.extend_from_slice(&[0; 16]);
    bmp.extend_from_slice(pixels);
    fs::write(output.join(name), bmp)?;
    Ok(())
}

fn validate_capture_pixels(
    pixels: &[u8],
    extent: [u32; 2],
    expected_rgba: [u8; 4],
    triangle_rgb_tolerance: u8,
) -> Result<(), String> {
    let width = usize::try_from(extent[0]).map_err(|error| error.to_string())?;
    let height = usize::try_from(extent[1]).map_err(|error| error.to_string())?;
    if width == 0 || height == 0 {
        return Err("capture extent must be nonzero".into());
    }
    if pixels.len() != width.saturating_mul(height).saturating_mul(4) {
        return Err("capture byte length does not match its client extent".into());
    }
    let rgb_at = |x: usize, y: usize| {
        let offset = (y * width + x) * 4;
        [pixels[offset + 2], pixels[offset + 1], pixels[offset]]
    };
    for (x, y) in [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ] {
        if rgb_at(x, y) != [0, 0, 0] {
            return Err(format!(
                "background corner ({x},{y}) was {:?}, expected black",
                rgb_at(x, y)
            ));
        }
    }
    let x = width * 8 / 20;
    let y = height * 57 / 100;
    let expected = [expected_rgba[0], expected_rgba[1], expected_rgba[2]];
    let actual = rgb_at(x, y);
    if !actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| actual.abs_diff(expected) <= triangle_rgb_tolerance)
    {
        return Err(format!(
            "triangle interior ({x},{y}) was {actual:?}, expected {expected:?} within {triangle_rgb_tolerance}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/capture.rs"]
mod tests;
