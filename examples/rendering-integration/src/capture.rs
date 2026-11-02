//! Consumer-owned native client captures for the surface fixture.
//!
//! This is intentionally a Win32 GDI capture of the live HWND client area,
//! not an RHI readback. It records what the native window exposes after a
//! completed presentation; callers must still inspect the resulting pixels.

use std::{error::Error, fs, mem::size_of, path::Path};

use fluxel_host::Window;

pub(crate) fn capture_client(
    window: &Window,
    output: &Path,
    frame: usize,
    expected_extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::{
        Foundation::{HWND, POINT, RECT},
        Graphics::Gdi::{
            BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, ClientToScreen, CreateCompatibleBitmap,
            CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits,
            ReleaseDC, SRCCOPY, SelectObject,
        },
        UI::WindowsAndMessaging::{
            GetClientRect, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos,
        },
    };

    let raw = window
        .window_handle()
        .map_err(|error| format!("capture window handle: {error:?}"))?
        .as_raw();
    let RawWindowHandle::Win32(handle) = raw else {
        return Err("surface capture expected a Win32 window".into());
    };
    let hwnd = HWND(handle.hwnd.get() as *mut _);
    let mut rect = RECT::default();
    // SAFETY: `hwnd` belongs to the live Host window and `rect` is writable.
    unsafe { GetClientRect(hwnd, &mut rect)? };
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 {
        return Err(format!("cannot capture empty client extent {width}x{height}").into());
    }
    let actual_extent = [u32::try_from(width)?, u32::try_from(height)?];
    if actual_extent != expected_extent {
        return Err(format!(
            "capture client extent {actual_extent:?} differs from completed frame {expected_extent:?}"
        )
        .into());
    }
    let bytes = usize::try_from(actual_extent[0])?
        .checked_mul(usize::try_from(actual_extent[1])?)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("capture byte size overflow")?;

    // SAFETY: all GDI handles are created from the live client DC and are
    // released before return. The selected bitmap is restored before delete.
    // Only this test application's window is raised, without taking focus.
    // Capturing an occluded client DC can return another application's pixels.
    unsafe {
        SetWindowPos(
            hwnd,
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )?;
    }
    let _raised = RaisedWindow(hwnd);
    // A square client is the fixture's visual target. Do not mask rounded
    // compositor corners and accidentally classify desktop pixels as output.
    let corner_preference = windows::Win32::Graphics::Dwm::DWMWCP_DONOTROUND;
    let square_corners = unsafe {
        windows::Win32::Graphics::Dwm::DwmSetWindowAttribute(
            hwnd,
            windows::Win32::Graphics::Dwm::DWMWA_WINDOW_CORNER_PREFERENCE,
            (&corner_preference
                as *const windows::Win32::Graphics::Dwm::DWM_WINDOW_CORNER_PREFERENCE)
                .cast(),
            u32::try_from(size_of::<
                windows::Win32::Graphics::Dwm::DWM_WINDOW_CORNER_PREFERENCE,
            >())?,
        )
    }
    .is_ok();
    // Let the native window opening/stacking animation settle before observing
    // the composed desktop. No GPU retirement decision uses this display wait.
    std::thread::sleep(std::time::Duration::from_millis(250));
    let mut origin = POINT { x: 0, y: 0 };
    if !unsafe { ClientToScreen(hwnd, &mut origin) }.as_bool() {
        return Err("ClientToScreen failed".into());
    }
    // Wait for a compositor boundary as well as the Renderer GPU completion.
    // This is a display observation; it is not used to retire GPU owners.
    unsafe {
        windows::Win32::Graphics::Dwm::DwmFlush()?;
    }
    let pixels = unsafe {
        let client_dc = GetDC(None);
        if client_dc.0.is_null() {
            return Err("GetDC returned a null client DC".into());
        }
        let memory_dc = CreateCompatibleDC(Some(client_dc));
        if memory_dc.0.is_null() {
            let _ = ReleaseDC(None, client_dc);
            return Err("CreateCompatibleDC returned null".into());
        }
        let bitmap = CreateCompatibleBitmap(client_dc, width, height);
        if bitmap.0.is_null() {
            let _ = DeleteDC(memory_dc);
            let _ = ReleaseDC(None, client_dc);
            return Err("CreateCompatibleBitmap returned null".into());
        }
        let previous = SelectObject(memory_dc, bitmap.into());
        let result = (|| -> Result<Vec<u8>, Box<dyn Error>> {
            BitBlt(
                memory_dc,
                0,
                0,
                width,
                height,
                Some(client_dc),
                origin.x,
                origin.y,
                SRCCOPY,
            )?;
            // GetDIBits requires that the bitmap is not selected into a DC.
            let _ = SelectObject(memory_dc, previous);
            let mut info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: u32::try_from(size_of::<BITMAPINFOHEADER>())?,
                    biWidth: width,
                    // A negative DIB height asks GDI for top-down scanlines.
                    biHeight: -height,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    biSizeImage: u32::try_from(bytes)?,
                    ..BITMAPINFOHEADER::default()
                },
                ..BITMAPINFO::default()
            };
            let mut pixels = vec![0_u8; bytes];
            let rows = GetDIBits(
                client_dc,
                bitmap,
                0,
                actual_extent[1],
                Some(pixels.as_mut_ptr().cast()),
                &mut info,
                DIB_RGB_COLORS,
            );
            if rows != height {
                return Err(format!("GetDIBits returned {rows} rows, expected {height}").into());
            }
            Ok(pixels)
        })();
        let _ = SelectObject(memory_dc, previous);
        let _ = DeleteObject(bitmap.into());
        let _ = DeleteDC(memory_dc);
        let _ = ReleaseDC(None, client_dc);
        result?
    };

    let bmp_name = format!("surface-client-frame-{frame}.bmp");
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
    bmp.extend_from_slice(&pixels);
    fs::write(output.join(&bmp_name), bmp)?;
    fs::write(
        output.join(format!("surface-client-frame-{frame}.capture.json")),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"frame\": {},\n",
                "  \"clientExtent\": [{}, {}],\n",
                "  \"method\": \"DwmFlush+raised-own-HWND+desktop-client-crop+BitBlt+GetDIBits\",\n",
                "  \"pixelFormat\": \"BGRA8-top-down\",\n",
                "  \"image\": {:?},\n",
                "  \"squareCornerPreferenceApplied\": {},\n",
                "  \"automaticVisualPass\": false\n",
                "}}\n"
            ),
            frame, actual_extent[0], actual_extent[1], bmp_name, square_corners,
        ),
    )?;
    Ok(())
}

/// Restore the fixture's ordinary stacking order even if capture fails.
struct RaisedWindow(windows::Win32::Foundation::HWND);
impl Drop for RaisedWindow {
    fn drop(&mut self) {
        use windows::Win32::UI::WindowsAndMessaging::{
            HWND_NOTOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos,
        };
        // SAFETY: this guard is scoped inside capture while the Host is alive.
        let _ = unsafe {
            SetWindowPos(
                self.0,
                Some(HWND_NOTOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        };
    }
}
