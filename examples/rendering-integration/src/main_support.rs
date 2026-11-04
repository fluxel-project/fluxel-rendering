//! Shared CLI, asset, and executor helpers for integration workloads.
use super::*;

pub(super) fn verify_s2_cache_replacement<B: fluxel_rhi::backend::Backend>(
    consumer: &mut s2_shader::S2ShaderConsumer<B>,
    old: fluxel_rhi::api::DeviceId,
    replacement: fluxel_rhi::api::DeviceId,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    if old == replacement || !consumer.has_device(old) || !consumer.has_device(replacement) {
        return Err("S2 cache did not retain separate products for both devices".into());
    }
    consumer.retire_device(old);
    if consumer.has_device(old) || !consumer.has_device(replacement) {
        return Err("S2 cache replacement retirement did not isolate device products".into());
    }
    fs::write(
        output.join("s2-cache-device-transition.json"),
        format!(
            "{{\"oldDeviceId\":{},\"replacementDeviceId\":{},\"bothProductsPresentBeforeRetirement\":true,\"oldProductRetiredAfterCompletion\":true,\"replacementProductRetained\":true}}\n",
            json_string(&format!("{old:?}")),
            json_string(&format!("{replacement:?}"))
        ),
    )?;
    Ok(())
}

pub(super) fn write_capabilities<B>(
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

pub(super) struct Bytes;
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

pub(super) fn block_on<F: Future>(future: F) -> F::Output {
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

pub(super) fn output_directory() -> Result<(PathBuf, bool, bool, bool, bool, bool), Box<dyn Error>>
{
    let mut args = std::env::args_os().skip(1);
    let mut output = None;
    let mut s3_only = false;
    let mut s5_only = false;
    let mut s5_surface_only = false;
    let mut s5_device_only = false;
    let mut s2_cache_only = false;
    while let Some(argument) = args.next() {
        if argument == "--output" {
            output = Some(
                args.next()
                    .ok_or("--output requires a directory argument")?
                    .into(),
            );
        } else if argument == "--s5-only" {
            s5_only = true;
        } else if argument == "--s5-surface-only" {
            s5_surface_only = true;
        } else if argument == "--s5-device-only" {
            s5_device_only = true;
        } else if argument == "--s3-only" {
            s3_only = true;
        } else if argument == "--s2-cache-only" {
            s2_cache_only = true;
        } else {
            return Err(format!("unsupported argument: {}", argument.to_string_lossy()).into());
        }
    }
    if u8::from(s3_only)
        + u8::from(s2_cache_only)
        + u8::from(s5_only)
        + u8::from(s5_surface_only)
        + u8::from(s5_device_only)
        > 1
    {
        return Err("workload selectors cannot be combined".into());
    }
    Ok((
        output.unwrap_or_else(|| PathBuf::from("target/integration-copy")),
        s3_only,
        s2_cache_only,
        s5_only,
        s5_surface_only,
        s5_device_only,
    ))
}

pub(super) fn producer(
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

pub(super) fn replacement(
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

pub(super) fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    fs::write(path, bytes)?;
    Ok(())
}

pub(super) fn framed_copy_source(
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
