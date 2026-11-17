use super::{Error, FrameTopology, validate_forward_topology, validate_surface_topology};

#[test]
fn clear_only_offscreen_rejects_draw_or_batch_facts() {
    assert!(matches!(
        validate_forward_topology(FrameTopology::ClearOnly, 1, 1),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        validate_forward_topology(FrameTopology::ClearOnly, 0, 1),
        Err(Error::InvalidState)
    ));
    assert!(validate_forward_topology(FrameTopology::ClearOnly, 0, 0).is_ok());
}

#[test]
fn clear_only_surface_rejects_draw_or_batch_facts() {
    assert!(matches!(
        validate_surface_topology(FrameTopology::ClearOnly, 1, 1),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        validate_surface_topology(FrameTopology::ClearOnly, 0, 1),
        Err(Error::InvalidState)
    ));
    assert!(validate_surface_topology(FrameTopology::ClearOnly, 0, 0).is_ok());
    assert!(validate_surface_topology(FrameTopology::OpaqueForward, 1, 1).is_ok());
}
