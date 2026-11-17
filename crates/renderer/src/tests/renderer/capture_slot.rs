use super::CaptureTextureSlot;

#[test]
fn capture_slot_requires_an_explicit_nonempty_material_name() {
    assert!(CaptureTextureSlot::new("").is_err());
    let slot = CaptureTextureSlot::new("captured_dragon").expect("explicit capture slot");
    assert_eq!(slot.name(), "captured_dragon");
    let producer = CaptureTextureSlot::new("producer_capture").expect("producer slot");
    assert_ne!(producer.name(), slot.name());
}
