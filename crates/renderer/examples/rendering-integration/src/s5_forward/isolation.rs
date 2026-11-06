//! Shared logical-asset facts for the two-device isolation workload.

use std::error::Error;

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_material::{MaterialAsset, MaterialDefinition};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardResources, GeometryAsset, GeometryData,
    RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene, RenderTransform,
    RenderView, RenderViewId, SceneAssetRef,
};

use super::EXTENT;

pub(super) fn snapshot_identity_json<K: AssetKind, V>(snapshots: &[AssetSnapshot<K, V>]) -> String {
    let values = snapshots
        .iter()
        .map(|snapshot| {
            let id = snapshot.id();
            format!(
                "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
                id.slot(),
                id.slot_generation(),
                snapshot.generation().get(),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

pub(super) fn scene(
    geometry: &[AssetSnapshot<GeometryAsset, GeometryData>; 4],
    material: &[AssetSnapshot<MaterialAsset, MaterialDefinition>; 4],
    texture: &[AssetSnapshot<ColorTextureAsset, ColorTextureData>; 4],
) -> Result<(RenderScene<ForwardResources>, RenderViewId), Box<dyn Error>> {
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    for index in 0..geometry.len() {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&geometry[index]),
                material: SceneAssetRef::from_snapshot(&material[index]),
                texture: SceneAssetRef::from_snapshot(&texture[index]),
                tint: [1.0; 4],
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok((scene, receipt.inserted_views()[0]))
}
