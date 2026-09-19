//! Section 23.3: the merge lattice and the merged requirement against an
//! interface.
//!
//! The merge of the participating stages' requirements - counts, kinds, buffer
//! access, storage-texture access - and then the check of the merged requirement
//! against a `PipelineInterface`. `use super::*` brings in the fixtures and the
//! vocabulary the whole chapter's tests share; the two banners below are the
//! original section banners.

use super::*;
// ---------------------------------------------------------------------------
// Section 23.3: merging the participating stages.
// ---------------------------------------------------------------------------

#[test]
fn merging_unions_the_stages_and_keeps_the_largest_requirement() {
    let vertex = requirements(vec![uniform(0, 0, 64)]);
    let fragment = requirements(vec![uniform(0, 0, 256)]);
    let merged = merge_shader_resources([
        (ShaderStage::Vertex, &vertex),
        (ShaderStage::Fragment, &fragment),
    ])
    .expect("the two stages agree on the kind and count");

    assert_eq!(merged.len(), 1);
    assert_eq!(
        merged[0].stages,
        ShaderStages::VERTEX.union(ShaderStages::FRAGMENT)
    );
    assert_eq!(
        merged[0].kind,
        BindingKind::UniformBuffer { min_size: 256 },
        "the layout must cover every stage that declares the binding"
    );
}

#[test]
fn merging_refuses_two_stages_that_disagree_about_a_count_or_a_kind() {
    let vertex = requirements(vec![uniform(0, 0, 64)]);
    let other_count = requirements(vec![ShaderResourceRequirement {
        count: BindingCount::Fixed(2),
        ..uniform(0, 0, 64)
    }]);
    assert_kind(
        merge_shader_resources([
            (ShaderStage::Vertex, &vertex),
            (ShaderStage::Fragment, &other_count),
        ])
        .map(|_| ()),
        RhiErrorKind::IncompatibleInterface,
    );

    let other_kind = requirements(vec![storage_buffer(
        0,
        0,
        BufferBindingAccess::ReadOnly,
        64,
    )]);
    assert_kind(
        merge_shader_resources([
            (ShaderStage::Vertex, &vertex),
            (ShaderStage::Fragment, &other_kind),
        ])
        .map(|_| ()),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn merging_takes_read_write_for_a_storage_buffer_when_any_stage_needs_it() {
    let vertex = requirements(vec![storage_buffer(
        0,
        0,
        BufferBindingAccess::ReadOnly,
        64,
    )]);
    let fragment = requirements(vec![storage_buffer(
        0,
        0,
        BufferBindingAccess::ReadWrite,
        64,
    )]);
    let merged = merge_shader_resources([
        (ShaderStage::Vertex, &vertex),
        (ShaderStage::Fragment, &fragment),
    ])
    .expect("the access lattice has a merge for these");

    assert_eq!(
        merged[0].kind,
        BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadWrite,
            min_size: 64,
        }
    );

    // Two read-only stages stay read-only, which is what keeps the merge from
    // widening every shared storage buffer.
    let other = requirements(vec![storage_buffer(
        0,
        0,
        BufferBindingAccess::ReadOnly,
        64,
    )]);
    let merged = merge_shader_resources([
        (ShaderStage::Vertex, &vertex),
        (ShaderStage::Fragment, &other),
    ])
    .expect("two read-only stages agree");
    assert_eq!(
        merged[0].kind,
        BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadOnly,
            min_size: 64,
        }
    );
}

#[test]
fn merging_turns_different_storage_texture_accesses_into_read_write() {
    let vertex = requirements(vec![storage_texture(0, 0, StorageAccess::ReadOnly)]);
    let fragment = requirements(vec![storage_texture(0, 0, StorageAccess::WriteOnly)]);
    let merged = merge_shader_resources([
        (ShaderStage::Vertex, &vertex),
        (ShaderStage::Fragment, &fragment),
    ])
    .expect("mixed accesses merge to ReadWrite");

    assert_eq!(
        merged[0].kind,
        BindingKind::StorageTexture {
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            access: StorageAccess::ReadWrite,
        }
    );
    assert!(
        merged[0].storage_access_from_merge,
        "the merged access must be asked about against the complete stage set"
    );
}

#[test]
fn a_merged_access_the_device_cannot_express_is_unsupported() {
    // The clause "if BindingSupport does not support ReadWrite -> Unsupported" is
    // asked with the union of the stages that forced the merge, which is why the
    // merge records that the access came from the lattice.
    let vertex = requirements(vec![storage_texture(0, 0, StorageAccess::ReadOnly)]);
    let fragment = requirements(vec![storage_texture(0, 0, StorageAccess::WriteOnly)]);
    let merged = merge_shader_resources([
        (ShaderStage::Vertex, &vertex),
        (ShaderStage::Fragment, &fragment),
    ])
    .expect("mixed accesses merge to ReadWrite");

    let interface = interface_of(vec![layout(vec![layout_slot(
        0,
        ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
        BindingKind::StorageTexture {
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            access: StorageAccess::ReadWrite,
        },
    )])]);

    let facts = permissive();
    assert!(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        })
        .is_ok()
    );

    let facts = permissive().refuses_binding(BindingSupportQuery {
        visibility: ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
        kind: BindingKind::StorageTexture {
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            access: StorageAccess::ReadWrite,
        },
        count: BindingCount::One,
        dynamic_offset: false,
    });
    assert_kind(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::Unsupported,
    );
}

// ---------------------------------------------------------------------------
// Section 23.3: a requirement against the interface.
// ---------------------------------------------------------------------------

/// One merged requirement, as the interface checks consume it.
fn one_merged(
    vertex: &ShaderInterface,
    fragment: Option<&ShaderInterface>,
) -> Vec<crate::api::pipeline::resources::MergedShaderResource> {
    match fragment {
        Some(fragment) => merge_shader_resources([
            (ShaderStage::Vertex, vertex),
            (ShaderStage::Fragment, fragment),
        ]),
        None => merge_shader_resources([(ShaderStage::Vertex, vertex)]),
    }
    .expect("the fixtures merge cleanly")
}

#[test]
fn a_layout_must_be_visible_to_every_stage_that_uses_the_binding() {
    let vertex = requirements(vec![uniform(0, 0, 64)]);
    let merged = one_merged(&vertex, None);
    let interface = interface_of(vec![layout(vec![layout_slot(
        0,
        ShaderStages::FRAGMENT,
        BindingKind::UniformBuffer { min_size: 64 },
    )])]);

    let facts = permissive();
    assert_kind(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn a_layout_must_declare_the_group_and_slot_the_shader_requires() {
    let vertex = requirements(vec![uniform(0, 0, 64)]);
    let merged = one_merged(&vertex, None);
    let facts = permissive();

    let no_group = no_bindings();
    assert_kind(
        validate_shader_resource_requirements(&merged, &no_group, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::IncompatibleInterface,
    );

    let no_slot = interface_of(vec![layout(vec![layout_slot(
        1,
        ShaderStages::VERTEX,
        BindingKind::UniformBuffer { min_size: 64 },
    )])]);
    assert_kind(
        validate_shader_resource_requirements(&merged, &no_slot, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn a_layout_must_guarantee_at_least_the_size_the_shader_requires() {
    let vertex = requirements(vec![uniform(0, 0, 256)]);
    let merged = one_merged(&vertex, None);
    let interface = interface_of(vec![layout(vec![layout_slot(
        0,
        ShaderStages::VERTEX,
        BindingKind::UniformBuffer { min_size: 64 },
    )])]);

    let facts = permissive();
    assert_kind(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn a_layout_access_must_cover_the_access_the_shader_uses() {
    let vertex = requirements(vec![storage_buffer(
        0,
        0,
        BufferBindingAccess::ReadWrite,
        64,
    )]);
    let merged = one_merged(&vertex, None);
    let interface = interface_of(vec![layout(vec![layout_slot(
        0,
        ShaderStages::VERTEX,
        BindingKind::StorageBuffer {
            access: BufferBindingAccess::ReadOnly,
            min_size: 64,
        },
    )])]);

    let facts = permissive();
    assert_kind(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        }),
        RhiErrorKind::IncompatibleInterface,
    );
}

#[test]
fn an_interface_may_declare_bindings_no_shader_uses() {
    // Section 23.3 keeps the interface free to be larger than one shader's
    // requirements, so that a Renderer can share one interface among several
    // pipelines.
    let vertex = requirements(vec![uniform(0, 0, 64)]);
    let merged = one_merged(&vertex, None);
    let interface = interface_of(vec![layout(vec![
        layout_slot(
            0,
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: 64 },
        ),
        layout_slot(
            5,
            ShaderStages::VERTEX,
            BindingKind::Sampler {
                kind: crate::api::binding::SamplerKind::Filtering,
            },
        ),
    ])]);

    let facts = permissive();
    assert!(
        validate_shader_resource_requirements(&merged, &interface, |query| {
            (facts.device().binding_support)(query)
        })
        .is_ok()
    );
}
