//! Section 23.1: what a pipeline interface may declare.
//!
//! The aggregate counts, the group-device rule, and the limits a device may
//! impose on a layout sequence. `use super::*` brings in the fixtures and the
//! vocabulary the whole chapter's tests share; the banner below is the original
//! section banner.

use super::*;

// ---------------------------------------------------------------------------
// Section 23.1: what a pipeline interface may declare.
// ---------------------------------------------------------------------------

#[test]
fn an_interface_may_not_declare_more_groups_than_the_device_allows() {
    let interface = interface_of(vec![
        layout(vec![layout_slot(
            0,
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 64 },
        )]),
        layout(vec![]),
    ]);
    let facts = permissive().limit(LimitKey::MaxBindGroups, 1);
    assert_kind(
        validate_pipeline_interface_descriptor(
            interface.descriptor(),
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_interface_may_not_mix_layouts_from_two_devices() {
    let foreign = BindGroupLayout::new(
        object(31),
        other_device(),
        BindGroupLayoutDescriptor::new(vec![]).canonicalized(),
        BindGroupLayoutCompatibilityId::new(1),
        LayoutFingerprint([3; 32]),
    );
    let interface = interface_of(vec![layout(vec![]), foreign]);
    let facts = permissive();
    assert_kind(
        validate_pipeline_interface_descriptor(
            interface.descriptor(),
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::WrongDevice,
    );
}

#[test]
fn a_fixed_binding_counts_as_all_of_its_elements_against_the_stage_limit() {
    // "Count Fixed(n) as n elements" is what makes the aggregate a resource count
    // rather than a slot count.
    let interface = interface_of(vec![layout(vec![
        layout_slot(
            0,
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 64 },
        )
        .with_count(BindingCount::Fixed(4)),
    ])]);
    let facts =
        permissive().binding_limit(ShaderStage::Vertex, BindingLimitClass::UniformBuffers, 3);
    assert_kind(
        validate_pipeline_interface_descriptor(
            interface.descriptor(),
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::InvalidUsage,
    );

    // A count within the ceiling is accepted, and a class the stage does not use is
    // not counted against it.
    let facts = permissive().binding_limit(ShaderStage::Vertex, BindingLimitClass::Samplers, 0);
    assert!(
        validate_pipeline_interface_descriptor(
            interface.descriptor(),
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        )
        .is_ok()
    );
}

#[test]
fn dynamic_buffer_elements_are_counted_per_pipeline_layout() {
    // A `Fixed(n)` binding with a dynamic offset contributes `n`, not one, to the
    // per-layout dynamic count.
    let interface = interface_of(vec![layout(vec![
        layout_slot(
            0,
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 64 },
        )
        .with_count(BindingCount::Fixed(2))
        .with_dynamic_offset(true),
    ])]);
    let facts = permissive().limit(LimitKey::MaxDynamicUniformBuffersPerPipelineLayout, 1);
    assert_kind(
        validate_pipeline_interface_descriptor(
            interface.descriptor(),
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn an_interface_reports_its_groups_by_index() {
    let interface = interface_of(vec![layout(vec![]), layout(vec![])]);
    assert_eq!(interface.descriptor().groups.len(), 2);
    assert!(interface.group(BindGroupIndex::new(0)).is_some());
    assert!(interface.group(BindGroupIndex::new(1)).is_some());
    assert!(
        interface.group(BindGroupIndex::new(2)).is_none(),
        "an absent group is None, not a default empty layout"
    );
}

#[test]
fn immediate_ranges_require_order_alignment_and_the_declared_maximum() {
    let base = interface_of(vec![layout(vec![])]);
    let facts = permissive()
        .limit(LimitKey::MaxImmediateSize, 16)
        .limit(LimitKey::ImmediateDataAlignment, 4);

    let good = PipelineInterfaceDescriptor::new(base.descriptor().groups.clone())
        .with_immediate_range(ImmediateRange::new(0, 4, ShaderStages::VERTEX))
        .with_immediate_range(ImmediateRange::new(4, 12, ShaderStages::FRAGMENT));
    assert!(
        validate_pipeline_interface_descriptor(
            &good,
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class)
        )
        .is_ok()
    );

    let unaligned = PipelineInterfaceDescriptor::new(base.descriptor().groups.clone())
        .with_immediate_range(ImmediateRange::new(2, 4, ShaderStages::VERTEX));
    assert_kind(
        validate_pipeline_interface_descriptor(
            &unaligned,
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::InvalidUsage,
    );

    let overlap = PipelineInterfaceDescriptor::new(base.descriptor().groups.clone())
        .with_immediate_range(ImmediateRange::new(0, 8, ShaderStages::VERTEX))
        .with_immediate_range(ImmediateRange::new(4, 4, ShaderStages::VERTEX));
    assert_kind(
        validate_pipeline_interface_descriptor(
            &overlap,
            |key| (facts.device().limit)(key),
            |stage, class| (facts.device().binding_limit)(stage, class),
        ),
        RhiErrorKind::InvalidUsage,
    );
}

#[test]
fn shader_immediate_requirements_need_a_stage_visible_interface_superset() {
    let shader = ShaderInterface::new()
        .with_compute_workgroup_size(crate::api::shader::ComputeWorkgroupSize::new(1, 1, 1))
        .with_immediate_requirement(ShaderImmediateRequirement::new(4, 8));
    let shader = module_on(
        device(),
        42,
        ShaderStage::Compute,
        shader,
        ShaderRequirements::new(),
    );
    let facts = permissive()
        .limit(LimitKey::MaxImmediateSize, 16)
        .limit(LimitKey::ImmediateDataAlignment, 4);
    let interface = |range: Option<ImmediateRange>| {
        let mut descriptor = PipelineInterfaceDescriptor::new(Vec::new());
        if let Some(range) = range {
            descriptor = descriptor.with_immediate_range(range);
        }
        PipelineInterface::new(
            object(41),
            device(),
            descriptor,
            PipelineInterfaceCompatibilityId::new(41),
            LayoutFingerprint([41; 32]),
        )
    };

    assert!(
        check_compute(
            &ComputePipelineDescriptor::new(
                shader.clone(),
                interface(Some(ImmediateRange::new(0, 12, ShaderStages::COMPUTE))),
            ),
            &facts,
        )
        .is_ok()
    );

    for bad in [
        interface(None),
        interface(Some(ImmediateRange::new(0, 4, ShaderStages::COMPUTE))),
        interface(Some(ImmediateRange::new(0, 12, ShaderStages::VERTEX))),
    ] {
        assert_kind(
            check_compute(&ComputePipelineDescriptor::new(shader.clone(), bad), &facts),
            RhiErrorKind::IncompatibleInterface,
        );
    }
}

// ---------------------------------------------------------------------------
