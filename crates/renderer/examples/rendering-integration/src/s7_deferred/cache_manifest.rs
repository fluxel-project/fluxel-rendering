//! Complete artifact ownership for the six Deferred cache frames.

use std::{error::Error, fs, path::Path};

pub(super) fn write(output: &Path) -> Result<(), Box<dyn Error>> {
    let mut cases = Vec::new();
    for name in [
        "cold",
        "warm-state",
        "warm-settled",
        "tint",
        "texture-replacement",
        "source-replacement",
    ] {
        let artifacts: Vec<_> = [
            "input.json",
            "material.txt",
            "expected.rgba",
            "expected.ppm",
            "stable-mask.bin",
            "facts.json",
            "actual.rgba",
            "actual.ppm",
            "composed.wgsl",
            "source-map.txt",
        ]
        .map(|suffix| format!("s7-deferred-cache-{name}-{suffix}"))
        .into();
        for artifact in &artifacts {
            if !output.join(artifact).is_file() {
                return Err(
                    format!("Deferred cache artifact missing before manifest: {artifact}").into(),
                );
            }
        }
        cases.push(format!(
            "{{\"name\":\"{name}\",\"artifacts\":{artifacts:?}}}"
        ));
    }
    fs::write(
        output.join("s7-deferred-cache-workload.json"),
        format!(
            "{{\"schemaVersion\":1,\"workload\":\"s7-deferred-cache\",\"sharedArtifacts\":[\"s7-deferred-cache-facts.json\"],\"cases\":[{}]}}\n",
            cases.join(","),
        ),
    )?;
    Ok(())
}
