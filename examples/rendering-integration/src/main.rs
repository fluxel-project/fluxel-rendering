//! S0 copy and completion consumer for the rebuilt Fluxel Renderer.
//!
//! This executable intentionally has no renderer-private access. It creates
//! three immutable Asset snapshots for one logical identity, submits all three
//! before the first completion wait, then checks the complete GPU readbacks.

#![cfg_attr(not(windows), allow(dead_code))]

mod capabilities;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod capture;
mod compute;
mod diagnostics;
mod raster;
mod read_set;
mod surface;

use std::{
    error::Error,
    fs,
    future::Future,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_renderer::{Renderer, RendererError};
use fluxel_rhi::{
    api::Device,
    backend::{ExecutionBackend, MappingBackend, ReadbackBackend, ResourceBackend},
};

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

fn write_capabilities<B>(
    device: &Device<B>,
    backend: &str,
    adapter: &str,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: fluxel_rhi::backend::Backend,
{
    capabilities::write(device, backend, adapter, output)
}

struct Bytes;
impl AssetKind for Bytes {}

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn output_directory() -> Result<PathBuf, Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut output = None;
    while let Some(argument) = args.next() {
        if argument == "--output" {
            output = Some(
                args.next()
                    .ok_or("--output requires a directory argument")?
                    .into(),
            );
        } else {
            return Err(format!("unsupported argument: {}", argument.to_string_lossy()).into());
        }
    }
    Ok(output.unwrap_or_else(|| PathBuf::from("target/integration-copy")))
}

fn producer(
    store: &AssetStore<Bytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<Bytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<Bytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial asset state".into()),
    }
}

fn replacement(
    store: &AssetStore<Bytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<Bytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<Bytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected replacement waiter".into()),
    }
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    fs::write(path, bytes)?;
    Ok(())
}

fn framed_copy_source(
    prefix: &[u8],
    payload_byte: u8,
    payload_len: usize,
    suffix: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(prefix.len() + payload_len + suffix.len());
    bytes.extend_from_slice(prefix);
    bytes.extend(std::iter::repeat_n(payload_byte, payload_len));
    bytes.extend_from_slice(suffix);
    bytes
}

fn run_copy<B>(
    device: Device<B>,
    adapter: &str,
    output: &Path,
) -> Result<fluxel_renderer::Submission, Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fs::create_dir_all(output)?;
    write_bytes(&output.join("selected-adapter.txt"), adapter.as_bytes())?;

    let store = AssetStore::<Bytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    // The payloads deliberately start at nonzero, changing offsets in source
    // buffers with asymmetric sentinel prefixes and suffixes. Compatible Copy
    // uses must refresh frame bindings while retaining one compiled topology.
    let inputs = [
        framed_copy_source(&[0xa1, 0xa2, 0xa3, 0xa4], 17, 16, &[0xb1, 0xb2, 0xb3]),
        framed_copy_source(
            &[0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8],
            18,
            24,
            &[0xd1, 0xd2, 0xd3, 0xd4, 0xd5],
        ),
        framed_copy_source(
            &[0xe1, 0xe2, 0xe3, 0xe4],
            19,
            12,
            &[0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6],
        ),
    ];
    let source_ranges: [Range<u64>; 3] = [4..20, 8..32, 4..16];
    let expected = [vec![17; 16], vec![18; 24], vec![19; 12]];
    for index in 0..inputs.len() {
        write_bytes(&output.join(format!("input-{index}.bin")), &inputs[index])?;
        write_bytes(
            &output.join(format!("expected-{index}.bin")),
            &expected[index],
        )?;
    }
    let first = producer(&store, &handle, inputs[0].clone())?;
    let second = replacement(&store, &handle, inputs[1].clone())?;
    let third = replacement(&store, &handle, inputs[2].clone())?;
    let generations = [
        first.generation().get(),
        second.generation().get(),
        third.generation().get(),
    ];
    let snapshots = [
        Arc::downgrade(&first.value_arc()),
        Arc::downgrade(&second.value_arc()),
        Arc::downgrade(&third.value_arc()),
    ];
    if first.id() != second.id()
        || second.id() != third.id()
        || !(first.generation() < second.generation() && second.generation() < third.generation())
    {
        return Err("asset replacement did not preserve identity and advance generations".into());
    }

    let mut renderer = Renderer::new(device.clone())?;
    let before = device.stats().snapshot();
    let empty_store = AssetStore::<Bytes, Vec<u8>, String>::new();
    let empty_handle = empty_store.create()?;
    let empty = producer(&empty_store, &empty_handle, Vec::new())?;
    if !matches!(
        block_on(renderer.submit(empty)),
        Err(RendererError::EmptyAsset)
    ) {
        return Err("empty asset did not receive the structured EmptyAsset refusal".into());
    }
    if device.stats().snapshot() != before {
        return Err("empty asset changed RHI request counters".into());
    }

    let submit_started = Instant::now();
    let first_submission = block_on(renderer.submit_copy_range(first, source_ranges[0].clone()))?;
    let plan = renderer
        .plan_id()
        .ok_or("copy plan identity was not retained")?;
    let submissions = [
        first_submission,
        block_on(renderer.submit_copy_range(second, source_ranges[1].clone()))?,
        block_on(renderer.submit_copy_range(third, source_ranges[2].clone()))?,
    ];
    let submit_elapsed = submit_started.elapsed();
    if renderer.pending() != 3
        || renderer.compilations() != 1
        || renderer.plan_id() != Some(plan)
        || renderer.binding_refreshes() != 3
    {
        return Err("three outstanding copy submissions did not retain one compiled plan".into());
    }

    let fourth = replacement(&store, &handle, vec![20; 20])?;
    let saturated_before = device.stats().snapshot();
    if !matches!(
        block_on(renderer.submit(fourth)),
        Err(RendererError::Backpressure)
    ) {
        return Err("slot saturation did not receive the structured Backpressure refusal".into());
    }
    if device.stats().snapshot() != saturated_before {
        return Err("backpressure refusal changed RHI request counters".into());
    }

    let mut other_renderer = Renderer::new(device.clone())?;
    if !matches!(
        block_on(other_renderer.readback(&submissions[0])),
        Err(RendererError::ForeignSubmission)
    ) {
        return Err("foreign renderer did not reject a submission before waiting".into());
    }

    drop(handle);
    drop(store);
    if snapshots
        .iter()
        .any(|snapshot| snapshot.upgrade().is_none())
    {
        return Err(
            "renderer did not retain every submitted source snapshot before completion".into(),
        );
    }
    let snapshot_retention_verified = true;
    let readback_started = Instant::now();
    for (index, submission) in submissions.iter().enumerate() {
        let readback = block_on(renderer.readback(submission))?;
        write_bytes(&output.join(format!("readback-{index}.bin")), &readback)?;
        if readback != expected[index] {
            return Err(format!(
                "readback {index} differs from its selected immutable payload range"
            )
            .into());
        }
        if snapshots[index].upgrade().is_some() {
            return Err(
                format!("snapshot {index} remained retained after its covering readback").into(),
            );
        }
        if snapshots[index + 1..]
            .iter()
            .any(|snapshot| snapshot.upgrade().is_none())
        {
            return Err(format!(
                "a later copy snapshot retired before its covering readback at frame {index}"
            )
            .into());
        }
    }
    let readback_elapsed = readback_started.elapsed();
    if renderer.pending() != 0 {
        return Err("all readbacks completed but renderer still retains submissions".into());
    }
    if !matches!(
        block_on(renderer.readback(&submissions[0])),
        Err(RendererError::StaleSubmission)
    ) {
        return Err(
            "retired submission did not receive the structured StaleSubmission refusal".into(),
        );
    }

    let after = device.stats().snapshot();
    let payload = format!(
        concat!(
            "{{\n",
            "  \"schemaVersion\": 1,\n",
            "  \"workload\": \"s0-copy-three-generations\",\n",
            "  \"adapter\": {:?},\n",
            "  \"assetGenerations\": [{}, {}, {}],\n",
            "  \"sourceByteLengths\": [{}, {}, {}],\n",
            "  \"sourceRanges\": [[{}, {}], [{}, {}], [{}, {}]],\n",
            "  \"outputByteLengths\": [{}, {}, {}],\n",
            "  \"submissions\": 3,\n",
            "  \"compiledPlans\": {},\n",
            "  \"frameBindingRefreshes\": {},\n",
            "  \"dynamicRangePlanReuseVerified\": true,\n",
            "  \"snapshotRetentionVerified\": {},\n",
            "  \"perFrameRetirementVerified\": true,\n",
            "  \"timingsNanoseconds\": {{\"submitThree\": {}, \"readbackThree\": {}}},\n",
            "  \"rhiCounters\": {{\"copies\": {}, \"submits\": {}}}\n",
            "}}\n"
        ),
        adapter,
        generations[0],
        generations[1],
        generations[2],
        inputs[0].len(),
        inputs[1].len(),
        inputs[2].len(),
        source_ranges[0].start,
        source_ranges[0].end,
        source_ranges[1].start,
        source_ranges[1].end,
        source_ranges[2].start,
        source_ranges[2].end,
        expected[0].len(),
        expected[1].len(),
        expected[2].len(),
        renderer.compilations(),
        renderer.binding_refreshes(),
        snapshot_retention_verified,
        submit_elapsed.as_nanos(),
        readback_elapsed.as_nanos(),
        after.copies.saturating_sub(before.copies),
        after.submits.saturating_sub(before.submits),
    );
    write_bytes(&output.join("workload.json"), payload.as_bytes())?;
    Ok(submissions[0].clone())
}

#[cfg(all(windows, feature = "dx12"))]
fn run_dx12(output: &Path) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{api::DeviceRequest, dx12::Dx12Instance};

    let native = Dx12Instance::new()?;
    let instance = native.instance()?;
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no DX12 adapter was enumerated")?;
    let adapter_name = adapter.name.clone();
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    diagnostics::record_dx12_device_open(output)?;
    write_capabilities(&device, "dx12", &adapter_name, output)?;
    let old_submission = run_copy(device.clone(), &adapter_name, output)?;
    raster::run(device.clone(), &output.join("raster"))?;
    compute::run(device.clone(), &output.join("compute"))?;
    read_set::run(device.clone(), &output.join("read-set"))?;
    surface::run_dx12(device.clone(), &output.join("surface"))?;
    let old_pipeline = raster::make_pipeline(&device, fluxel_rhi::api::TextureFormat::Rgba8Unorm)?;
    let replacement = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    raster::replacement_rejects_old_pipeline(old_pipeline, replacement, &old_submission, output)
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn run_wgl(output: &Path) -> Result<(), Box<dyn Error>> {
    use fluxel_host::{Window, WindowConfig};
    use fluxel_rhi::{
        api::{AdapterId, DeviceRequest, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };

    let window = Window::new(WindowConfig::new("Fluxel S1 Surface", 320, 180)?)?;
    // SAFETY: the window remains alive for the instance, device, and all work
    // in `run_copy`, and this function remains on its context owner thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [320, 180]) }?;
    let id = InstanceId::allocate()?;
    // SAFETY: this Core instance owns the adopted native context while it opens.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL adapter was enumerated")?;
    let adapter_name = adapter.name.clone();
    let device = block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    write_capabilities(&device, "native-gl-wgl", &adapter_name, output)?;
    let mut observer = diagnostics::WglObserver::install(&device)?;
    let mut old_submission = None;
    let result = run_copy(device.clone(), &adapter_name, output)
        .map(|submission| old_submission = Some(submission))
        .and_then(|()| raster::run(device.clone(), &output.join("raster")))
        .and_then(|()| compute::run(device.clone(), &output.join("compute")))
        .and_then(|()| read_set::run(device.clone(), &output.join("read-set")))
        .and_then(|()| surface::run_wgl(device.clone(), &window, &output.join("surface")));
    let diagnostic_result = observer.write(&device, output);
    if result.is_err() || diagnostic_result.is_err() {
        // An error can leave an indeterminate native owner retained by the
        // Renderer. The window is its WGL context's external lifetime root;
        // preserve it even if draining native diagnostics itself failed.
        std::mem::forget(window);
        return result.and(diagnostic_result);
    }

    let old_pipeline = raster::make_pipeline(&device, fluxel_rhi::api::TextureFormat::Rgba8Unorm)?;
    // A WGL replacement must own a separate HWND and native context. Reusing
    // the original context here would not exercise RHI device provenance.
    let replacement_window = Window::new(WindowConfig::new("Fluxel replacement WGL", 320, 180)?)?;
    // SAFETY: the replacement window and its independently-created context
    // remain alive until its replacement-device workload has completed.
    let replacement_native =
        unsafe { GlInstance::from_wgl_window(&replacement_window, [320, 180]) }?;
    let replacement_id = InstanceId::allocate()?;
    // SAFETY: this Core instance owns the newly adopted replacement context.
    let replacement_instance =
        unsafe { Instance::<Gl>::from_native(replacement_id, replacement_native) };
    let replacement_adapter = replacement_instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL replacement adapter was enumerated")?;
    let replacement = block_on(replacement_instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: replacement_id,
            ordinal: replacement_adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    let replacement_diagnostics = output.join("replacement-diagnostics");
    fs::create_dir_all(&replacement_diagnostics)?;
    let mut replacement_observer = diagnostics::WglObserver::install(&replacement)?;
    let replacement_result = raster::replacement_rejects_old_pipeline(
        old_pipeline,
        replacement.clone(),
        old_submission
            .as_ref()
            .ok_or("old copy submission is missing")?,
        output,
    );
    let replacement_diagnostic_result =
        replacement_observer.write(&replacement, &replacement_diagnostics);
    // Include old-context pipeline creation and product destruction used by
    // the provenance refusal, without discarding the earlier queue drain.
    let old_diagnostic_result = observer.write(&device, output);
    if replacement_result.is_err() || replacement_diagnostic_result.is_err() {
        // The replacement context is independently owned. Preserve this Host
        // root on a failed replacement submission rather than dropping it
        // before completion evidence exists.
        std::mem::forget(replacement_window);
    }
    if old_diagnostic_result.is_err() {
        std::mem::forget(window);
    }
    replacement_result?;
    replacement_diagnostic_result?;
    old_diagnostic_result
}

fn main() -> Result<(), Box<dyn Error>> {
    // Pixel-sized RHI surfaces and desktop client captures must share one
    // coordinate space. This configures only this fixture process.
    #[cfg(windows)]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    let output = output_directory()?;
    #[cfg(all(windows, feature = "dx12", feature = "native-gl-wgl"))]
    return Err("select exactly one rendering integration backend feature".into());
    #[cfg(all(windows, feature = "dx12", not(feature = "native-gl-wgl")))]
    return run_dx12(&output);
    #[cfg(all(windows, feature = "native-gl-wgl", not(feature = "dx12")))]
    return run_wgl(&output);
    #[cfg(not(all(
        windows,
        any(feature = "dx12", feature = "native-gl-wgl"),
        not(all(feature = "dx12", feature = "native-gl-wgl"))
    )))]
    {
        let _ = output;
        Err("select --features dx12 or --features native-gl-wgl on Windows".into())
    }
}
