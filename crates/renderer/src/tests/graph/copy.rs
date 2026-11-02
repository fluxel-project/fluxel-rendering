use super::*;
use crate::graph::compiler::RecompileReason;

#[test]
fn resized_copy_frames_require_explicit_dynamic_extent_compatibility() {
    let small = CopyFrame::compile_range(4, 0..4).unwrap();
    let larger = CopyFrame::compile_range(12, 0..12).unwrap();
    assert_eq!(
        small
            .execution
            .logical()
            .recompile_reason(larger.execution.logical()),
        Some(RecompileReason::BufferLayout)
    );
    assert_eq!(
        small
            .execution
            .logical()
            .recompile_reason_for_dynamic_buffers(larger.execution.logical()),
        None
    );
    for requirement in larger.execution.logical().buffer_bindings().requirements() {
        assert_eq!(requirement.size, 12);
        assert!(requirement.ranges.iter().all(|range| *range == (0..12)));
    }
}
