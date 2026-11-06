//! The private Windows Graphics Capture consumer and CPU readback path.

use std::{error::Error, fmt, mem::size_of, sync::mpsc, time::Duration};

use fluxel_host::Window;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::{
    Foundation::TypedEventHandler,
    Graphics::{
        Capture::{
            Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem,
            GraphicsCaptureSession,
        },
        DirectX::{
            Direct3D11::{IDirect3DDevice, IDirect3DSurface},
            DirectXPixelFormat,
        },
        SizeInt32,
    },
    Win32::{
        Foundation::{HMODULE, HWND, POINT, RECT},
        Graphics::{
            Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL},
            Direct3D11::{
                D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
                D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
                ID3D11Texture2D,
            },
            Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute},
            Dxgi::{Common::DXGI_FORMAT_B8G8R8A8_UNORM, IDXGIDevice},
            Gdi::ClientToScreen,
        },
        System::WinRT::{
            Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess},
            Graphics::Capture::IGraphicsCaptureItemInterop,
            RO_INIT_MULTITHREADED, RoInitialize, RoUninitialize,
        },
        UI::WindowsAndMessaging::{
            GWL_EXSTYLE, GetClientRect, GetForegroundWindow, GetWindowLongPtrW, GetWindowRect,
            IsIconic, IsWindowVisible, WS_EX_TOPMOST,
        },
    },
    core::{Interface, factory},
};

const CAPTURE_TIMEOUT: Duration = Duration::from_secs(2);

pub(super) struct CapturedClient {
    pub(super) pixels: Vec<u8>,
    pub(super) extent: [u32; 2],
    pub(super) content_extent: [u32; 2],
    pub(super) client_crop: [u32; 4],
    pub(super) timeout_ms: u128,
    pub(super) before: WindowState,
    pub(super) after: WindowState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct WindowState {
    foreground: isize,
    visible: bool,
    minimized: bool,
    topmost: bool,
}
impl WindowState {
    pub(super) fn json(self) -> String {
        format!(
            "{{\"foregroundHwnd\":{},\"visible\":{},\"minimized\":{},\"topmost\":{}}}",
            self.foreground, self.visible, self.minimized, self.topmost
        )
    }
}

#[derive(Debug)]
struct CaptureError {
    stage: &'static str,
    diagnostics: String,
}
impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "windows-graphics-capture-window {{\"stage\":\"{}\",\"diagnostics\":{}}}",
            self.stage, self.diagnostics
        )
    }
}
impl Error for CaptureError {}

pub(super) fn capture_window_client(
    window: &Window,
    expected: [u32; 2],
) -> Result<CapturedClient, Box<dyn Error>> {
    let hwnd = hwnd(window)?;
    let before = state(hwnd);
    if !before.visible || before.minimized {
        return Err(err(
            "window-state",
            before,
            if before.minimized {
                "HWND is minimized"
            } else {
                "HWND is not visible"
            },
        ));
    }
    let geometry = geometry(hwnd, expected, before)?;
    if !GraphicsCaptureSession::IsSupported().map_err(|e| winerr("is-supported", before, e))? {
        return Err(err(
            "is-supported",
            before,
            "Windows Graphics Capture is unavailable",
        ));
    }
    let item = item(hwnd, before)?;
    let size = item.Size().map_err(|e| winerr("item-size", before, e))?;
    let content = size_of_capture(size, "item-size", before)?;
    let crop = geometry.crop(content, before)?;
    let (device, context, winrt_device) = device(before)?;
    let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &winrt_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        size,
    )
    .map_err(|e| winerr("frame-pool", before, e))?;
    let session = pool
        .CreateCaptureSession(&item)
        .map_err(|e| winerr("capture-session", before, e))?;
    let (send, receive) = mpsc::sync_channel(1);
    let token = pool
        .FrameArrived(&TypedEventHandler::new(move |_, _| {
            let _ = send.try_send(());
            Ok(())
        }))
        .map_err(|e| winerr("frame-handler", before, e))?;
    let capture = CaptureLifetime {
        pool,
        session,
        token,
    };
    capture
        .session
        .StartCapture()
        .map_err(|e| winerr("start-capture", before, e))?;
    receive.recv_timeout(CAPTURE_TIMEOUT).map_err(|e| {
        err(
            "frame-timeout",
            before,
            &format!("timeoutMs={}; event={e}", CAPTURE_TIMEOUT.as_millis()),
        )
    })?;
    let frame = FrameLifetime(
        capture
            .pool
            .TryGetNextFrame()
            .map_err(|e| winerr("next-frame", before, e))?,
    );
    if size_of_capture(
        frame
            .0
            .ContentSize()
            .map_err(|e| winerr("frame-size", before, e))?,
        "frame-size",
        before,
    )? != content
    {
        return Err(err(
            "frame-size",
            before,
            "content size changed during capture",
        ));
    }
    let surface = frame
        .0
        .Surface()
        .map_err(|e| winerr("frame-surface", before, e))?;
    let pixels = read(&device, &context, &surface, content, crop, before)?;
    let after = state(hwnd);
    if after != before {
        return Err(err(
            "window-state-changed",
            after,
            &format!("before={}; after={}", before.json(), after.json()),
        ));
    }
    Ok(CapturedClient {
        pixels,
        extent: expected,
        content_extent: content,
        client_crop: crop,
        timeout_ms: CAPTURE_TIMEOUT.as_millis(),
        before,
        after,
    })
}

pub(super) struct CaptureRuntime;
impl CaptureRuntime {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        // SAFETY: this thread balances every successful initialization below.
        unsafe {
            RoInitialize(RO_INIT_MULTITHREADED).map_err(|error| {
                format!("Windows Graphics Capture WinRT initialization: {error}")
            })?;
        }
        Ok(Self)
    }
}
impl Drop for CaptureRuntime {
    fn drop(&mut self) {
        unsafe { RoUninitialize() };
    }
}

struct CaptureLifetime {
    pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    token: i64,
}
impl Drop for CaptureLifetime {
    fn drop(&mut self) {
        let _ = self.pool.RemoveFrameArrived(self.token);
        let _ = self.session.Close();
        let _ = self.pool.Close();
    }
}
struct FrameLifetime(Direct3D11CaptureFrame);
impl Drop for FrameLifetime {
    fn drop(&mut self) {
        let _ = self.0.Close();
    }
}

fn hwnd(window: &Window) -> Result<HWND, Box<dyn Error>> {
    match window
        .window_handle()
        .map_err(|e| format!("capture window handle: {e:?}"))?
        .as_raw()
    {
        RawWindowHandle::Win32(handle) => Ok(HWND(handle.hwnd.get() as *mut _)),
        _ => Err("surface capture expected a Win32 window".into()),
    }
}
fn state(hwnd: HWND) -> WindowState {
    unsafe {
        WindowState {
            foreground: GetForegroundWindow().0 as isize,
            visible: IsWindowVisible(hwnd).as_bool(),
            minimized: IsIconic(hwnd).as_bool(),
            topmost: (GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32 & WS_EX_TOPMOST.0) != 0,
        }
    }
}

struct Geometry {
    origin: POINT,
    client: [u32; 2],
    window: RECT,
    extended: Option<RECT>,
}
impl Geometry {
    fn crop(&self, content: [u32; 2], state: WindowState) -> Result<[u32; 4], Box<dyn Error>> {
        // WGC reports whole-window content. Accept only an actual WindowRect
        // or DWM extended-frame rectangle with exactly that size: no inferred
        // DPI scale or fabricated desktop crop is permitted.
        let capture = (rect_extent(self.window) == content)
            .then_some(self.window)
            .or_else(|| self.extended.filter(|rect| rect_extent(*rect) == content))
            .ok_or_else(|| {
                err(
                    "capture-bounds",
                    state,
                    &format!(
                        "content={content:?}; window={:?}; extended={:?}",
                        rect_extent(self.window),
                        self.extended.map(rect_extent)
                    ),
                )
            })?;
        let x = self.origin.x - capture.left;
        let y = self.origin.y - capture.top;
        if x < 0
            || y < 0
            || x as u32 + self.client[0] > content[0]
            || y as u32 + self.client[1] > content[1]
        {
            return Err(err(
                "client-crop",
                state,
                &format!(
                    "origin=({},{}), client={:?}, captureOrigin=({},{}), content={content:?}",
                    self.origin.x, self.origin.y, self.client, capture.left, capture.top
                ),
            ));
        }
        Ok([x as u32, y as u32, self.client[0], self.client[1]])
    }
}
fn geometry(
    hwnd: HWND,
    expected: [u32; 2],
    state: WindowState,
) -> Result<Geometry, Box<dyn Error>> {
    let mut rect = RECT::default();
    let mut origin = POINT::default();
    let mut window = RECT::default();
    let mut extended = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut rect).map_err(|e| winerr("client-rect", state, e))?;
        if !ClientToScreen(hwnd, &mut origin).as_bool() {
            return Err(err("client-origin", state, "ClientToScreen failed"));
        }
        GetWindowRect(hwnd, &mut window).map_err(|e| winerr("window-rect", state, e))?;
    }
    let client = [
        u32::try_from(rect.right - rect.left)
            .map_err(|_| err("client-rect", state, "nonpositive width"))?,
        u32::try_from(rect.bottom - rect.top)
            .map_err(|_| err("client-rect", state, "nonpositive height"))?,
    ];
    if client != expected {
        return Err(err(
            "client-extent",
            state,
            &format!("actual={client:?}; expected={expected:?}"),
        ));
    }
    let extended = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            (&mut extended as *mut RECT).cast(),
            size_of::<RECT>() as u32,
        )
    }
    .ok()
    .map(|_| extended);
    Ok(Geometry {
        origin,
        client,
        window,
        extended,
    })
}

fn rect_extent(rect: RECT) -> [u32; 2] {
    [
        u32::try_from(rect.right - rect.left).unwrap_or(0),
        u32::try_from(rect.bottom - rect.top).unwrap_or(0),
    ]
}

fn item(hwnd: HWND, state: WindowState) -> Result<GraphicsCaptureItem, Box<dyn Error>> {
    let interop: IGraphicsCaptureItemInterop =
        factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
            .map_err(|e| winerr("capture-interop", state, e))?;
    unsafe {
        interop
            .CreateForWindow(hwnd)
            .map_err(|e| winerr("create-for-window", state, e))
            .map_err(Into::into)
    }
}
fn device(
    state: WindowState,
) -> Result<(ID3D11Device, ID3D11DeviceContext, IDirect3DDevice), Box<dyn Error>> {
    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None::<&[D3D_FEATURE_LEVEL]>,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(|e| winerr("d3d11-device", state, e))?;
    }
    let device = device.ok_or_else(|| err("d3d11-device", state, "API returned no device"))?;
    let context = context.ok_or_else(|| err("d3d11-device", state, "API returned no context"))?;
    let dxgi: IDXGIDevice = device.cast().map_err(|e| winerr("dxgi-device", state, e))?;
    let inspectable = unsafe {
        CreateDirect3D11DeviceFromDXGIDevice(&dxgi).map_err(|e| winerr("winrt-device", state, e))?
    };
    Ok((
        device,
        context,
        inspectable
            .cast()
            .map_err(|e| winerr("winrt-device-cast", state, e))?,
    ))
}
fn read(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    surface: &IDirect3DSurface,
    content: [u32; 2],
    crop: [u32; 4],
    state: WindowState,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let access: IDirect3DDxgiInterfaceAccess = surface
        .cast()
        .map_err(|e| winerr("surface-access", state, e))?;
    let texture: ID3D11Texture2D = unsafe {
        access
            .GetInterface()
            .map_err(|e| winerr("surface-texture", state, e))?
    };
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: `texture` is the frame's live DXGI surface and `desc` is writable.
    unsafe {
        texture.GetDesc(&mut desc);
    }
    if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM
        || desc.ArraySize != 1
        || desc.MipLevels != 1
        || desc.SampleDesc.Count != 1
        || desc.SampleDesc.Quality != 0
        || desc.Width < content[0]
        || desc.Height < content[1]
    {
        return Err(err(
            "frame-texture",
            state,
            &format!(
                "format={:?}; dimensions={}x{}; mips={}; array={}; samples={}:{}; content={content:?}",
                desc.Format,
                desc.Width,
                desc.Height,
                desc.MipLevels,
                desc.ArraySize,
                desc.SampleDesc.Count,
                desc.SampleDesc.Quality
            ),
        ));
    }
    // CopyResource requires exactly compatible descriptors. Preserve the frame
    // allocation extent (which may exceed ContentSize), then make it CPU-only.
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    desc.MiscFlags = 0;
    let mut staging = None;
    unsafe {
        device
            .CreateTexture2D(&desc, None, Some(&mut staging))
            .map_err(|e| winerr("staging-texture", state, e))?;
    }
    let staging =
        staging.ok_or_else(|| err("staging-texture", state, "API returned no texture"))?;
    unsafe {
        context.CopyResource(&staging, &texture);
    }
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        context
            .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|e| winerr("map-staging", state, e))?;
    }
    let copied = copy(mapped, crop, state);
    unsafe {
        context.Unmap(&staging, 0);
    }
    copied
}
fn copy(
    mapped: D3D11_MAPPED_SUBRESOURCE,
    crop: [u32; 4],
    state: WindowState,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let row = usize::try_from(crop[2])?
        .checked_mul(4)
        .ok_or_else(|| err("copy-crop", state, "row overflow"))?;
    let required_pitch = usize::try_from(crop[0])?
        .checked_mul(4)
        .and_then(|offset| offset.checked_add(row))
        .ok_or_else(|| err("copy-crop", state, "row footprint overflow"))?;
    if mapped.pData.is_null() || (mapped.RowPitch as usize) < required_pitch {
        return Err(err("copy-crop", state, "invalid mapped row pitch"));
    }
    let mut output = vec![
        0;
        row.checked_mul(crop[3] as usize).ok_or_else(|| err(
            "copy-crop",
            state,
            "output overflow"
        ))?
    ];
    for y in 0..crop[3] as usize {
        let offset = (crop[1] as usize + y) * mapped.RowPitch as usize + crop[0] as usize * 4;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (mapped.pData as *const u8).add(offset),
                output[y * row..].as_mut_ptr(),
                row,
            );
        }
    }
    Ok(output)
}
fn size_of_capture(
    size: SizeInt32,
    stage: &'static str,
    state: WindowState,
) -> Result<[u32; 2], Box<dyn Error>> {
    match (u32::try_from(size.Width), u32::try_from(size.Height)) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Ok([w, h]),
        _ => Err(err(
            stage,
            state,
            &format!("invalid size {}x{}", size.Width, size.Height),
        )),
    }
}
fn err(stage: &'static str, state: WindowState, detail: &str) -> Box<dyn Error> {
    Box::new(CaptureError {
        stage,
        diagnostics: format!(
            "{{\"detail\":{:?},\"windowState\":{}}}",
            detail,
            state.json()
        ),
    })
}
fn winerr(stage: &'static str, state: WindowState, error: windows::core::Error) -> Box<dyn Error> {
    err(stage, state, &error.to_string())
}
