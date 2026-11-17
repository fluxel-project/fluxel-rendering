//! Immutable pre-submission evidence for the paired capture attempts.

use std::{error::Error, fs, path::Path};

use fluxel_renderer::GeometryData;

use crate::{
    inputs::{CaptureSceneInputs, EXTENT},
    oracle::{CONTROL_ALBEDO, CaptureOracle, PRIMARY_ALBEDO},
};

/// Saves actual CPU inputs and independent expected probes before GPU work.
pub fn write_pre_submission(
    output: &Path,
    dragon: &GeometryData,
    plane: &GeometryData,
    oracle: &CaptureOracle,
    primary: &CaptureSceneInputs,
    control: &CaptureSceneInputs,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;
    fs::write(output.join("dragon-geometry.bin"), geometry_bytes(dragon)?)?;
    fs::write(output.join("plane-geometry.bin"), geometry_bytes(plane)?)?;
    fs::write(
        output.join("capture-oracle.json"),
        format!(
            "{{\"schemaVersion\":1,\"extent\":[{},{}],\"clearRgba\":{:?},\"primaryAlbedo\":{:?},\"controlAlbedo\":{:?},\"colorProbes\":[{}],\"backgroundProbe\":{}}}\n",
            EXTENT[0],
            EXTENT[1],
            CLEAR_RGBA_JSON,
            PRIMARY_ALBEDO,
            CONTROL_ALBEDO,
            oracle
                .color_probes
                .iter()
                .map(probe_json)
                .collect::<Vec<_>>()
                .join(","),
            probe_json(&oracle.background_probe),
        ),
    )?;
    fs::write(
        output.join("capture-contract.txt"),
        "The two attempts differ only in producer RGBA8 albedo. The receiver has no CPU texture asset and samples captured_dragon in one Graph. Exact interior probes must equal the respective producer albedo; the selected background probe must remain transparent clear in both attempts.\n",
    )?;
    crate::window_oracle::write_contract(output, EXTENT)?;
    write_actual_input(output, "primary", primary)?;
    write_actual_input(output, "control", control)?;
    Ok(())
}

const CLEAR_RGBA_JSON: [u8; 4] = crate::oracle::CLEAR_RGBA;

fn geometry_bytes(geometry: &GeometryData) -> Result<&[u8], Box<dyn Error>> {
    if geometry.streams.len() == 1 {
        Ok(&geometry.streams[0])
    } else {
        Err("07 evidence expects one interleaved geometry stream".into())
    }
}

fn probe_json(probe: &crate::oracle::Probe) -> String {
    format!(
        "{{\"finalPixel\":[{},{}],\"captureTexel\":[{},{}]}}",
        probe.final_pixel[0], probe.final_pixel[1], probe.capture_texel[0], probe.capture_texel[1]
    )
}

fn write_actual_input(
    output: &Path,
    name: &str,
    input: &CaptureSceneInputs,
) -> Result<(), Box<dyn Error>> {
    let record = input
        .capture_snapshot
        .record_ids()
        .next()
        .ok_or("capture input has no producer record")?;
    fs::write(
        output.join(format!("{name}-capture-record.txt")),
        format!("{:#?}\n", input.capture_snapshot.record(record)?),
    )?;
    fs::write(
        output.join(format!("{name}-capture-view.txt")),
        format!("{:#?}\n", input.capture_snapshot.view(input.capture_view)?),
    )?;
    fs::write(
        output.join(format!("{name}-receiver-input.txt")),
        format!(
            "receiverView={:#?}\nreceiverTransform={:#?}\nreceiverTint={:?}\n",
            input.receiver_view, input.receiver_transform, input.receiver_tint,
        ),
    )?;
    let producer_material = input
        .assets
        .materials
        .first()
        .ok_or("capture input has no producer material")?;
    let receiver_material = input
        .assets
        .materials
        .get(1)
        .ok_or("capture input has no receiver material")?;
    let producer_albedo = input
        .assets
        .textures
        .first()
        .ok_or("capture input has no producer albedo")?;
    fs::write(
        output.join(format!("{name}-producer-material.txt")),
        format!("{:#?}\n", producer_material.value()),
    )?;
    fs::write(
        output.join(format!("{name}-receiver-material.txt")),
        format!("{:#?}\n", receiver_material.value()),
    )?;
    fs::write(
        output.join(format!("{name}-producer-albedo.rgba")),
        &producer_albedo.value().rgba8,
    )?;
    Ok(())
}
