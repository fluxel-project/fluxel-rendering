//! Per-frame artifacts derived from the exact submitted scene and marker.

use std::{error::Error, fs, path::Path};

use external_renderer::{QuadInput, QuadMarker};
use fluxel_assets::AssetKind;
use fluxel_renderer::{ForwardResources, FrameSnapshot, RenderBounds, SceneAssetRef};

use super::oracle;

#[allow(clippy::too_many_arguments)]
pub(super) fn write_frame_inputs(
    output: &Path,
    name: &str,
    snapshot: &FrameSnapshot<ForwardResources>,
    typed_version: u64,
    input: &QuadInput,
    feature_enabled: bool,
    volume: RenderBounds,
    expected_mesh_visible: bool,
    expected_feature_visible: bool,
    expected: &[u8],
) -> Result<(), Box<dyn Error>> {
    let marker = input.markers.first().ok_or("S6 input has no marker")?;
    let mesh = snapshot.record(
        snapshot
            .record_ids()
            .next()
            .ok_or("S6 snapshot has no mesh record")?,
    )?;
    let mesh_visible = intersects(mesh.bounds, volume);
    let feature_visible = feature_enabled && marker_intersects(marker, volume);
    if expected_mesh_visible != mesh_visible || expected_feature_visible != feature_visible {
        return Err("S6 expected visibility disagrees with actual snapshot/marker inputs".into());
    }
    let marker_json = format!(
        "{{\"id\":{},\"center\":{:?},\"halfExtent\":{:?},\"tint\":{:?}}}",
        marker.id, marker.center, marker.half_extent, marker.tint
    );
    let mesh_json = format!(
        "{{\"bounds\":{{\"minimum\":{:?},\"maximum\":{:?}}},\"transformColumns\":{:?}}}",
        mesh.bounds.minimum(),
        mesh.bounds.maximum(),
        mesh.transform.columns()
    );
    let assets_json = format!(
        "{{\"mesh\":{},\"quad\":{}}}",
        resources_json(&mesh.resources),
        quad_assets_json(marker)
    );
    let input_json = format!(
        concat!(
            "{{\"case\":\"{}\",\"sceneVersion\":{},\"typedVersion\":{},",
            "\"meshVisible\":{},\"featureEnabled\":{},\"featureVisible\":{},",
            "\"featureOrder\":{},\"marker\":{},\"mesh\":{},\"assets\":{}}}\n"
        ),
        name,
        snapshot.version(),
        typed_version,
        mesh_visible,
        feature_enabled,
        feature_visible,
        marker.order,
        marker_json,
        mesh_json,
        assets_json
    );
    fs::write(output.join(format!("s6-{name}-input.json")), input_json)?;
    fs::write(output.join(format!("s6-{name}-expected.rgba")), expected)?;
    fs::write(
        output.join(format!("s6-{name}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(output.join("s6-stable-mask.bin"), oracle::mask())?;
    Ok(())
}

fn resources_json(resources: &ForwardResources) -> String {
    format!(
        "{{\"geometry\":{},\"material\":{},\"texture\":{}}}",
        asset_json(resources.geometry),
        asset_json(resources.material),
        asset_json(resources.texture)
    )
}
fn quad_assets_json(marker: &QuadMarker) -> String {
    format!(
        "{{\"geometry\":{},\"material\":{},\"texture\":{}}}",
        asset_json(marker.assets.geometry),
        asset_json(marker.assets.material),
        asset_json(marker.assets.texture)
    )
}
fn asset_json<K: AssetKind>(reference: SceneAssetRef<K>) -> String {
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        reference.id().slot(),
        reference.id().slot_generation(),
        reference.generation().get()
    )
}
fn intersects(left: RenderBounds, right: RenderBounds) -> bool {
    left.minimum()
        .iter()
        .zip(left.maximum())
        .zip(right.minimum())
        .zip(right.maximum())
        .all(|(((a, b), c), d)| a <= &d && c <= b)
}
fn marker_intersects(marker: &QuadMarker, volume: RenderBounds) -> bool {
    let min = [
        marker.center[0] - marker.half_extent[0],
        marker.center[1] - marker.half_extent[1],
        marker.center[2],
    ];
    let max = [
        marker.center[0] + marker.half_extent[0],
        marker.center[1] + marker.half_extent[1],
        marker.center[2],
    ];
    (0..3).all(|i| min[i] <= volume.maximum()[i] && volume.minimum()[i] <= max[i])
}
