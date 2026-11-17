//! Pre-submission source identity and immutable workload inputs.

use super::{
    encoding_name, geometry, material, Options, TextureInput, LOCKED_RHI_REVISION,
    REFERENCE_RHI_REVISION,
};
use std::{error::Error, fs};

pub(super) fn write_source_identity(
    options: &Options,
    texture: &TextureInput,
) -> Result<(), Box<dyn Error>> {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let renderer_revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_owned())
        .unwrap_or_else(|| "unavailable".into());
    fs::write(
        options.output.join("source.json"),
        format!(
            "{{\"schemaVersion\":1,\"rendererRevision\":\"{renderer_revision}\",\"lockedRhiRevision\":\"{LOCKED_RHI_REVISION}\",\"referenceRhiCheckoutRevision\":\"{REFERENCE_RHI_REVISION}\",\"sourceAsset\":{},\"assetSha256\":\"{}\",\"extent\":[{},{}],\"encoding\":\"{}\",\"inputKind\":{:?},\"bindingLayout\":\"{}\",\"sourceGeometry\":\"RHI 02 indexed quad expanded to non-indexed Renderer geometry\"}}\n",
            options.asset.as_ref().map(|path| format!("{:?}", path.to_string_lossy())).unwrap_or_else(|| "null".into()), texture.sha256, texture.extent[0], texture.extent[1], encoding_name(texture.encoding), texture.source, if options.permuted_bindings { "tint=2,texture=0,sampler=1" } else { "tint=0,texture=1,sampler=2" }
        ),
    )?;
    fs::write(options.output.join("command.txt"), format!("{command}\n"))?;
    fs::write(
        options.output.join("Cargo.lock"),
        include_bytes!("../Cargo.lock"),
    )?;
    fs::write(
        options.output.join("source-main.rs"),
        include_bytes!("main.rs"),
    )?;
    fs::write(
        options.output.join("source-evidence.rs"),
        include_bytes!("evidence.rs"),
    )?;
    fs::write(
        options.output.join("source-native.rs"),
        include_bytes!("native.rs"),
    )?;
    fs::write(
        options.output.join("source-support.rs"),
        include_bytes!("support.rs"),
    )?;
    fs::write(
        options.output.join("source-srgb-filter.rs"),
        include_bytes!("srgb_filter.rs"),
    )?;
    fs::write(
        options.output.join("source-sample-oracle.rs"),
        include_bytes!("sample_oracle.rs"),
    )?;
    #[cfg(windows)]
    fs::write(
        options.output.join("source-window-oracle.rs"),
        include_bytes!("window_oracle.rs"),
    )?;
    #[cfg(windows)]
    fs::write(
        options.output.join("source-capture.rs"),
        include_bytes!("../../rendering-integration/src/capture.rs"),
    )?;
    #[cfg(windows)]
    fs::write(
        options.output.join("source-diagnostics.rs"),
        include_bytes!("../../rendering-integration/src/diagnostics.rs"),
    )?;
    fs::write(
        options.output.join("Cargo.toml"),
        include_bytes!("../Cargo.toml"),
    )?;
    fs::write(
        options.output.join("geometry.bin"),
        &geometry(options.srgb_filter).streams[0],
    )?;
    fs::write(
        options.output.join("material.txt"),
        format!("{:#?}\n", material(options.permuted_bindings)),
    )?;
    fs::write(options.output.join("texture.rgba"), &texture.rgba8)?;
    Ok(())
}
