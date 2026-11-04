//! Surface configuration and host resize observation.
use super::*;

#[cfg(windows)]
struct ResizeWaitState {
    observed_events: Vec<String>,
    selected_event: Option<&'static str>,
    last_extent: [u32; 2],
}

#[cfg(windows)]
enum ResizeWaitOutcome {
    Confirmed {
        fact: ResizeFact,
        state: ResizeWaitState,
    },
    TimedOut(ResizeWaitState),
}

const RESIZE_EVENT_TIMEOUT: Duration = Duration::from_secs(1);
const RESIZE_EVENT_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[cfg(windows)]
fn requested_resize_event(
    events: &[fluxel_host::WindowEvent],
    target: [u32; 2],
) -> Option<&'static str> {
    events.iter().find_map(|event| match event {
        fluxel_host::WindowEvent::Resized { width, height } if [*width, *height] == target => {
            Some("resized")
        }
        fluxel_host::WindowEvent::Restored { width, height } if [*width, *height] == target => {
            Some("restored")
        }
        _ => None,
    })
}

#[cfg(windows)]
fn resize_is_confirmed(
    event: Option<&'static str>,
    client_extent: [u32; 2],
    target: [u32; 2],
) -> bool {
    event.is_some() && client_extent == target
}

#[cfg(windows)]
fn await_resize_confirmation<Observe, Expired, Pause>(
    target: [u32; 2],
    mut observe: Observe,
    mut expired: Expired,
    mut pause: Pause,
) -> Result<ResizeWaitOutcome, Box<dyn Error>>
where
    Observe: FnMut() -> Result<(Vec<fluxel_host::WindowEvent>, [u32; 2]), Box<dyn Error>>,
    Expired: FnMut() -> bool,
    Pause: FnMut(),
{
    let mut observed_events = Vec::new();
    let mut selected_event = None;
    loop {
        let (events, observed_extent) = observe()?;
        observed_events.extend(events.iter().map(|event| format!("{event:?}")));
        if selected_event.is_none() {
            selected_event = requested_resize_event(&events, target);
        }
        if resize_is_confirmed(selected_event, observed_extent, target) {
            return Ok(ResizeWaitOutcome::Confirmed {
                fact: ResizeFact {
                    extent: observed_extent,
                    event: selected_event.expect("resize confirmation requires an event"),
                },
                state: ResizeWaitState {
                    observed_events,
                    selected_event,
                    last_extent: observed_extent,
                },
            });
        }
        if expired() {
            return Ok(ResizeWaitOutcome::TimedOut(ResizeWaitState {
                observed_events,
                selected_event,
                last_extent: observed_extent,
            }));
        }
        pause();
    }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::from("\"");
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => escaped.push_str(&format!("\\u{:04x}", value as u32)),
            value => escaped.push(value),
        }
    }
    escaped.push('\"');
    escaped
}

fn append_resize_observation(
    output: &Path,
    phase: &str,
    requested: [u32; 2],
    events: &[String],
    selected_event: Option<&str>,
    client_extent: Option<[u32; 2]>,
) -> Result<(), Box<dyn Error>> {
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("surface-events.jsonl"))?;
    let events = events
        .iter()
        .map(|event| json_string(event))
        .collect::<Vec<_>>()
        .join(", ");
    let selected_event = selected_event.map_or_else(|| "null".to_owned(), json_string);
    let client_extent = client_extent.map_or_else(
        || "null".to_owned(),
        |[width, height]| format!("[{width}, {height}]"),
    );
    writeln!(
        log,
        "{{\"phase\":{},\"requestedExtent\":[{},{}],\"observedEvents\":[{}],\"selectedEvent\":{},\"clientExtent\":{}}}",
        json_string(phase),
        requested[0],
        requested[1],
        events,
        selected_event,
        client_extent,
    )?;
    Ok(())
}

pub(super) fn initial(
    store: &AssetStore<SurfaceBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<SurfaceBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<SurfaceBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial surface asset state".into()),
    }
}

pub(super) fn replacement(
    store: &AssetStore<SurfaceBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<SurfaceBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<SurfaceBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected surface replacement waiter".into()),
    }
}

/// Configures one acquired-surface generation after checking the advertised
/// capabilities and recording the exact configuration decision.
///
/// A caller must have completed every submission that retained an image from
/// the previous configuration before calling this function.
pub(crate) fn configure<B: SurfaceBackend>(
    surface: &mut Surface<B>,
    extent: [u32; 2],
    output: &Path,
) -> Result<SurfaceConfig, Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let format = if caps.formats.contains(&TextureFormat::Rgba8Unorm) {
        TextureFormat::Rgba8Unorm
    } else {
        caps.formats
            .first()
            .copied()
            .ok_or("surface exposed no format")?
    };
    let present_mode = if caps.present_modes.contains(&PresentMode::Fifo) {
        PresentMode::Fifo
    } else {
        caps.present_modes
            .first()
            .copied()
            .ok_or("surface exposed no present mode")?
    };
    let alpha_mode = if caps.alpha_modes.contains(&CompositeAlphaMode::Opaque) {
        CompositeAlphaMode::Opaque
    } else {
        caps.alpha_modes
            .first()
            .copied()
            .ok_or("surface exposed no composite alpha mode")?
    };
    let color_space = if caps.color_spaces.contains(&ColorSpace::Srgb) {
        ColorSpace::Srgb
    } else {
        caps.color_spaces
            .first()
            .copied()
            .ok_or("surface exposed no color space")?
    };
    if extent[0] < caps.min_extent[0]
        || extent[1] < caps.min_extent[1]
        || extent[0] > caps.max_extent[0]
        || extent[1] > caps.max_extent[1]
    {
        return Err(format!(
            "surface extent {extent:?} is outside its reported range {:?}..={:?}",
            caps.min_extent, caps.max_extent
        )
        .into());
    }
    if caps.usages.0 & SurfaceUsage::RENDER_ATTACHMENT.0 == 0 {
        return Err("surface does not advertise render-attachment usage".into());
    }
    let config = SurfaceConfig {
        extent,
        format,
        present_mode,
        alpha_mode,
        color_space,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    if let Support::Unsupported(reason) = surface.configuration_support(&config) {
        append_configuration_observation(output, &caps, &config, false, Some(reason))?;
        return Err(
            format!("surface rejected the selected supported configuration: {reason}").into(),
        );
    }
    append_configuration_observation(output, &caps, &config, true, None)?;
    block_on(surface.configure(&config))
        .map_err(|error| format!("surface configure {extent:?}: {error:?}"))?;
    Ok(config)
}

fn append_configuration_observation(
    output: &Path,
    caps: &SurfaceCapabilities,
    config: &SurfaceConfig,
    supported: bool,
    reason: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let SurfaceCapabilities {
        formats,
        present_modes,
        alpha_modes,
        color_spaces,
        usages,
        min_extent,
        max_extent,
    } = caps;
    let SurfaceConfig {
        extent,
        format,
        present_mode,
        alpha_mode,
        color_space,
        usage,
        maximum_frame_latency,
    } = config;
    fn enum_values<T: std::fmt::Debug>(values: &[T]) -> String {
        values
            .iter()
            .map(|value| json_string(&format!("{value:?}")))
            .collect::<Vec<_>>()
            .join(",")
    }
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("surface-configuration.jsonl"))?;
    writeln!(
        log,
        concat!(
            "{{\"extent\":[{},{}],\"requiredUsage\":\"render_attachment\",\"configurationSupport\":{},\"reason\":{},",
            "\"capabilities\":{{\"formats\":[{}],\"presentModes\":[{}],\"alphaModes\":[{}],\"colorSpaces\":[{}],\"usages\":{},\"minExtent\":{:?},\"maxExtent\":{:?}}},",
            "\"configuration\":{{\"extent\":{:?},\"format\":{},\"presentMode\":{},\"alphaMode\":{},\"colorSpace\":{},\"usage\":{},\"maximumFrameLatency\":{}}}}}"
        ),
        extent[0],
        extent[1],
        supported,
        reason.map_or_else(|| "null".to_owned(), json_string),
        enum_values(formats),
        enum_values(present_modes),
        enum_values(alpha_modes),
        enum_values(color_spaces),
        usages.0,
        min_extent,
        max_extent,
        extent,
        json_string(&format!("{format:?}")),
        json_string(&format!("{present_mode:?}")),
        json_string(&format!("{alpha_mode:?}")),
        json_string(&format!("{color_space:?}")),
        usage.0,
        maximum_frame_latency,
    )?;
    Ok(())
}

#[cfg(windows)]
/// Requests a native client resize and waits for both Host evidence and the
/// queried client extent before a caller reconfigures its Surface.
pub(crate) fn resize_window(
    window: &Window,
    target: [u32; 2],
    output: &Path,
    phase: &str,
) -> Result<ResizeFact, Box<dyn Error>> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::{
        Foundation::{HWND, RECT},
        UI::WindowsAndMessaging::{
            GetClientRect, GetWindowRect, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
        },
    };

    let raw = window
        .window_handle()
        .map_err(|error| format!("resize window handle: {error:?}"))?
        .as_raw();
    let RawWindowHandle::Win32(handle) = raw else {
        return Err("surface fixture expected a Win32 window".into());
    };
    let hwnd = HWND(handle.hwnd.get() as *mut _);
    let mut outer = RECT::default();
    let mut client = RECT::default();
    // SAFETY: `hwnd` was borrowed from the live host window on its owner thread;
    // the RECT values are local writable storage.
    unsafe {
        GetWindowRect(hwnd, &mut outer)?;
        GetClientRect(hwnd, &mut client)?;
    }
    let border_width = (outer.right - outer.left) - (client.right - client.left);
    let border_height = (outer.bottom - outer.top) - (client.bottom - client.top);
    let outer_width = i32::try_from(target[0])?
        .checked_add(border_width)
        .ok_or("resize width overflow")?;
    let outer_height = i32::try_from(target[1])?
        .checked_add(border_height)
        .ok_or("resize height overflow")?;
    // SAFETY: this changes only the live fixture HWND and retains its z-order
    // and location. The previous renderer submission was completion-drained.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            outer_width,
            outer_height,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )?;
    }
    // SetWindowPos posts Win32 messages asynchronously. Pump the actual Host
    // queue for a bounded period, retaining every observed event and the last
    // queried client extent. A matching extent alone is not confirmation: the
    // Host must also report the matching lifecycle transition.
    let deadline = Instant::now() + RESIZE_EVENT_TIMEOUT;
    match await_resize_confirmation(
        target,
        || -> Result<_, Box<dyn Error>> {
            let events = window.poll_events()?;
            let mut observed = RECT::default();
            // Query the actual client extent after each real event pump. The
            // resulting log remains useful if the Host never reports the resize.
            // SAFETY: the same live HWND and local RECT remain valid for this query.
            unsafe {
                GetClientRect(hwnd, &mut observed)?;
            }
            Ok((
                events,
                [
                    u32::try_from(observed.right - observed.left)?,
                    u32::try_from(observed.bottom - observed.top)?,
                ],
            ))
        },
        || Instant::now() >= deadline,
        || std::thread::sleep(RESIZE_EVENT_POLL_INTERVAL),
    )? {
        ResizeWaitOutcome::Confirmed { fact, state } => {
            append_resize_observation(
                output,
                phase,
                target,
                &state.observed_events,
                state.selected_event,
                Some(state.last_extent),
            )?;
            Ok(fact)
        }
        ResizeWaitOutcome::TimedOut(state) => {
            append_resize_observation(
                output,
                phase,
                target,
                &state.observed_events,
                state.selected_event,
                Some(state.last_extent),
            )?;
            return Err(format!(
                "timed out waiting for host resize {target:?}; matching event: {:?}; last client extent: {:?}; observed events: {:?}",
                state.selected_event, state.last_extent, state.observed_events,
            )
            .into());
        }
    }
}

#[cfg(all(test, windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "tests/surface.rs"]
mod tests;
