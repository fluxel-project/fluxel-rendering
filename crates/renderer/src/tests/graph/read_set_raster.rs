use super::*;

#[test]
fn dynamic_uniform_contract_describes_physical_bytes_not_binding_base() {
    let first = ReadSetRasterFrame::compile(0..256).unwrap();
    let next = ReadSetRasterFrame::compile(256..512).unwrap();
    let requirements = next.execution.logical().buffer_bindings().requirements();
    let uniform = requirements
        .iter()
        .find(|entry| entry.slot == next.uniform_slot)
        .unwrap();
    assert_eq!(uniform.size, 512);
    assert!(uniform.ranges.iter().all(|range| *range == (256..272)));
    assert_eq!(
        first
            .execution
            .logical()
            .recompile_reason_for_dynamic_buffers(next.execution.logical()),
        None
    );
}

impl ReadSetRasterFrame {
    /// Fault-inject a candidate topology which must never reach recording.
    pub(crate) fn changed_topology_for_test() -> Self {
        let mut frame = Self::compile(0..256).unwrap();
        let mut graph = GraphBuilder::default();
        graph
            .add_pass("incompatible-side-effect", |pass| {
                pass.side_effect();
                Ok(())
            })
            .unwrap();
        frame.execution = CompiledExecution::compile(&graph).unwrap();
        frame
    }
}
