use std::{
    error::Error,
    fs,
    future::Future,
    path::Path,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

pub(crate) fn write_probes(
    output: &Path,
    route: &str,
    rgba: &[u8],
    extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    if rgba.len() != (extent[0] * extent[1] * 4) as usize {
        return Err("readback byte count differs from configured extent".into());
    }
    let samples = [
        (extent[0] / 4, extent[1] / 4),
        (extent[0] * 3 / 4, extent[1] / 4),
        (extent[0] / 4, extent[1] * 3 / 4),
        (extent[0] * 3 / 4, extent[1] * 3 / 4),
    ];
    let values: Vec<_> = samples
        .into_iter()
        .map(|(x, y)| {
            let at = ((y * extent[0] + x) * 4) as usize;
            format!("{{\"x\":{x},\"y\":{y},\"rgba\":{:?}}}", &rgba[at..at + 4])
        })
        .collect();
    fs::write(
        output.join(format!("{route}-probes.json")),
        format!(
            "{{\"schemaVersion\":1,\"origin\":\"top-left Core row zero\",\"probes\":[{}]}}\n",
            values.join(",")
        ),
    )?;
    Ok(())
}

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
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

pub(crate) fn write_output_hashes(output: &Path) -> Result<(), Box<dyn Error>> {
    let names = [
        "Cargo.lock",
        "Cargo.toml",
        "source-main.rs",
        "source-support.rs",
        "source-srgb-filter.rs",
        "source-sample-oracle.rs",
        "source-evidence.rs",
        "source-native.rs",
        "source.json",
        "command.txt",
        "geometry.bin",
        "material.txt",
        "texture.rgba",
        "forward-expected-probes.json",
        "forward-readback.rgba",
        "forward-probes.json",
        "deferred-expected-probes.json",
        "deferred-readback.rgba",
        "deferred-probes.json",
        "result.json",
    ];
    let lines = names
        .into_iter()
        .map(|name| {
            fs::read(output.join(name)).map(|bytes| format!("{}  {name}", crate::sha256(&bytes)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    fs::write(output.join("sha256.txt"), format!("{}\n", lines.join("\n")))?;
    Ok(())
}
