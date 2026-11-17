use renderer_07_offscreen_material::{inputs, oracle, scene};

#[test]
fn capture_scene_keeps_producer_asset_and_receiver_slot_distinct() {
    let scene = inputs::build_capture_scene(scene::mirror_geometry(), scene::mirror_geometry())
        .expect("scene inputs");

    assert_eq!(scene.capture_snapshot.version(), 1);
    assert_eq!(scene.capture_snapshot.record_ids().count(), 1);
    assert_eq!(scene.capture_snapshot.view_ids().count(), 1);
    assert_eq!(
        scene
            .capture_snapshot
            .view(scene.capture_view)
            .unwrap()
            .extent,
        inputs::EXTENT
    );
    assert_eq!(scene.assets.geometry.len(), 2);
    assert_eq!(scene.assets.materials.len(), 2);
    assert_eq!(scene.assets.textures.len(), 1);
    assert_eq!(scene.slot.name(), scene::CAPTURE_TEXTURE_SLOT);
}

#[test]
fn producer_color_control_changes_only_the_producer_texture_input() {
    let primary = inputs::build_capture_scene(scene::mirror_geometry(), scene::mirror_geometry())
        .expect("primary scene inputs");
    let control = inputs::build_capture_scene_with_albedo(
        scene::mirror_geometry(),
        scene::mirror_geometry(),
        oracle::CONTROL_ALBEDO,
    )
    .expect("control scene inputs");

    assert_eq!(
        primary.assets.textures[0].value().rgba8,
        oracle::PRIMARY_ALBEDO
    );
    assert_eq!(
        control.assets.textures[0].value().rgba8,
        oracle::CONTROL_ALBEDO
    );
    assert_eq!(primary.slot.name(), control.slot.name());
    assert_eq!(primary.assets.geometry.len(), control.assets.geometry.len());
    assert_eq!(
        primary.assets.materials.len(),
        control.assets.materials.len()
    );
}
