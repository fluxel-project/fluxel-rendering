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

/// Exactly one named native workload is selected for an executable attempt.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum Workload {
    #[default]
    Foundation,
    S3Material,
    S2Cache,
    S5Forward,
    S5Surface,
    S5DeviceIsolation,
    S6Composed,
    S6Surface,
    S7Deferred,
    S7DeferredSurface,
    S7DeviceIsolation,
    DemoForward,
    DemoDeferred,
    S8Material(crate::s8_material::Config),
    S8Lifecycle,
}

pub(super) fn output_directory() -> Result<(PathBuf, Workload), Box<dyn Error>> {
    parse_arguments(std::env::args_os().skip(1))
}

fn parse_arguments(
    mut args: impl Iterator<Item = std::ffi::OsString>,
) -> Result<(PathBuf, Workload), Box<dyn Error>> {
    let mut output = None;
    let mut selected = None;
    let mut s8_scenario = None;
    let mut s8_smoke = false;
    while let Some(argument) = args.next() {
        if argument == "--output" {
            output = Some(
                args.next()
                    .ok_or("--output requires a directory argument")?
                    .into(),
            );
            continue;
        }
        if argument == "--s8-scenario" {
            let value = args.next().ok_or("--s8-scenario requires a scenario")?;
            s8_scenario = Some(crate::s8_material::Scenario::parse(
                value.to_str().ok_or("--s8-scenario must be UTF-8")?,
            )?);
            continue;
        }
        if argument == "--s8-smoke" {
            s8_smoke = true;
            continue;
        }
        let workload = match argument.to_str() {
            Some("--s3-only") => Workload::S3Material,
            Some("--s2-cache-only") => Workload::S2Cache,
            Some("--s5-only") => Workload::S5Forward,
            Some("--s5-surface-only") => Workload::S5Surface,
            Some("--s5-device-only") => Workload::S5DeviceIsolation,
            Some("--s6-only") => Workload::S6Composed,
            Some("--s6-surface-only") => Workload::S6Surface,
            Some("--s7-deferred-only") => Workload::S7Deferred,
            Some("--s7-deferred-surface-only") => Workload::S7DeferredSurface,
            Some("--s7-device-isolation-only") => Workload::S7DeviceIsolation,
            Some("--demo-forward") => Workload::DemoForward,
            Some("--demo-deferred") => Workload::DemoDeferred,
            Some("--s8-material-only") => Workload::S8Material(crate::s8_material::Config::new(
                crate::s8_material::Scenario::Cold,
                false,
            )),
            Some("--s8-lifecycle-only") => Workload::S8Lifecycle,
            _ => {
                return Err(format!("unsupported argument: {}", argument.to_string_lossy()).into());
            }
        };
        if selected.is_some_and(|previous| previous != workload) {
            return Err("workload selectors cannot be combined".into());
        }
        selected = Some(workload);
    }
    let selected = match selected.unwrap_or_default() {
        Workload::S8Material(_) => Workload::S8Material(crate::s8_material::Config::new(
            s8_scenario.unwrap_or(crate::s8_material::Scenario::Cold),
            s8_smoke,
        )),
        Workload::S8Lifecycle if s8_scenario.is_some() || s8_smoke => {
            return Err("S8 options require --s8-material-only".into());
        }
        _ if s8_scenario.is_some() || s8_smoke => {
            return Err("S8 options require --s8-material-only".into());
        }
        selected => selected,
    };
    Ok((
        output.unwrap_or_else(|| PathBuf::from("target/integration-copy")),
        selected,
    ))
}

#[cfg(test)]
#[path = "tests/cli.rs"]
mod cli_tests;

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
