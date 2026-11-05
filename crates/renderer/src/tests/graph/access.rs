use super::*;

use fluxel_rhi::api::{
    AccessMask, BufferDesc, BufferUsage, BufferUses, ImageLayout, MemoryClass, StageMask,
    TextureAspects, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
};

use crate::graph::{
    binding::{BufferSlot, TextureSlot},
    definition::{AccessKind, GraphBuilder, TextureRange},
};

fn buffer_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        descriptor_usage: BufferUsage::COPY_SRC,
    }
}

fn texture_contract() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        layout: ImageLayout::ShaderReadOnly,
        descriptor_usage: TextureUsage::TEXTURE_BINDING,
    }
}

fn imported_buffer(graph: &mut GraphBuilder) -> crate::graph::definition::BufferHandle {
    graph
        .import_buffer_desc(
            BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            },
            buffer_contract().uses,
        )
        .unwrap()
}

fn imported_texture(graph: &mut GraphBuilder) -> crate::graph::definition::TextureHandle {
    graph
        .import_texture_with_state(
            TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [4, 4, 1],
                array_layers: 2,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING,
                view_formats: Vec::new(),
            },
            texture_contract().uses,
            ImageLayout::ShaderReadOnly,
        )
        .unwrap()
}

#[test]
fn rejects_resource_not_declared_by_this_pass() {
    let mut graph = GraphBuilder::default();
    let declared = imported_buffer(&mut graph);
    let undeclared = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("declared", |pass| {
            pass.read_with_use(
                declared,
                0..8,
                buffer_contract().uses,
                BufferUsage::COPY_SRC,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert_eq!(
        plan.pass(pass).unwrap().require_buffer(
            BufferSlot::from(undeclared),
            0..8,
            AccessKind::Read,
            buffer_contract()
        ),
        Err(AccessError::UndeclaredBuffer)
    );
}

#[test]
fn rejects_buffer_range_escaping_its_declaration() {
    let mut graph = GraphBuilder::default();
    let buffer = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("bounded", |pass| {
            pass.read_with_use(buffer, 0..8, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert_eq!(
        plan.pass(pass).unwrap().require_buffer(
            BufferSlot::from(buffer),
            0..9,
            AccessKind::Read,
            buffer_contract()
        ),
        Err(AccessError::BufferRangeOutsideDeclaration)
    );
}

#[test]
fn rejects_slot_from_a_different_graph() {
    let mut graph = GraphBuilder::default();
    let buffer = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("local", |pass| {
            pass.read_with_use(buffer, 0..8, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let foreign = imported_buffer(&mut GraphBuilder::default());
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert_eq!(
        plan.pass(pass).unwrap().require_buffer(
            BufferSlot::from(foreign),
            0..8,
            AccessKind::Read,
            buffer_contract()
        ),
        Err(AccessError::UndeclaredBuffer)
    );
}

#[test]
fn requires_declared_texture_subresources_and_layout() {
    let mut graph = GraphBuilder::default();
    let texture = imported_texture(&mut graph);
    let range = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);
    let (pass, ()) = graph
        .add_pass("sample", |pass| {
            pass.read_texture_with_use(
                texture,
                range.clone(),
                texture_contract().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert!(
        plan.pass(pass)
            .unwrap()
            .require_texture(
                TextureSlot::from(texture),
                range,
                AccessKind::Read,
                texture_contract()
            )
            .is_ok()
    );
    assert_eq!(
        plan.pass(pass).unwrap().require_texture(
            TextureSlot::from(texture),
            TextureRange::new(TextureAspects::COLOR, 0..2, 0..1),
            AccessKind::Read,
            texture_contract(),
        ),
        Err(AccessError::TextureRangeOutsideDeclaration)
    );
}

#[test]
fn permits_adjacent_compatible_buffer_declarations() {
    let mut graph = GraphBuilder::default();
    let buffer = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("adjacent", |pass| {
            pass.read_with_use(buffer, 0..8, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.read_with_use(buffer, 8..16, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert!(
        plan.pass(pass)
            .unwrap()
            .require_buffer(
                BufferSlot::from(buffer),
                0..16,
                AccessKind::Read,
                buffer_contract()
            )
            .is_ok()
    );
}

// Copy the helper after `imported_texture` and the tests after the current
// final test in `crates/renderer/src/tests/graph/access.rs`.

fn imported_depth_stencil_texture(
    graph: &mut GraphBuilder,
) -> crate::graph::definition::TextureHandle {
    graph
        .import_texture_with_state(
            TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [4, 4, 1],
                array_layers: 2,
                format: TextureFormat::Depth24PlusStencil8,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING,
                view_formats: Vec::new(),
            },
            texture_contract().uses,
            ImageLayout::ShaderReadOnly,
        )
        .unwrap()
}

#[test]
fn normalizes_overlapping_ranges_but_preserves_holes() {
    let mut graph = GraphBuilder::default();
    let buffer = graph
        .import_buffer_desc(
            BufferDesc {
                label: None,
                size: 32,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            },
            buffer_contract().uses,
        )
        .unwrap();
    let (pass, ()) = graph
        .add_pass("normalized", |pass| {
            pass.read_with_use(buffer, 0..8, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.read_with_use(buffer, 4..12, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.read_with_use(
                buffer,
                16..24,
                buffer_contract().uses,
                BufferUsage::COPY_SRC,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    let access = plan.pass(pass).unwrap();
    assert!(
        access
            .require_buffer(
                BufferSlot::from(buffer),
                0..12,
                AccessKind::Read,
                buffer_contract()
            )
            .is_ok()
    );
    assert_eq!(
        access.require_buffer(
            BufferSlot::from(buffer),
            0..24,
            AccessKind::Read,
            buffer_contract()
        ),
        Err(AccessError::BufferRangeOutsideDeclaration)
    );
}

#[test]
fn rejects_undeclared_texture_aspect_bits() {
    let mut graph = GraphBuilder::default();
    let texture = imported_texture(&mut graph);
    let (pass, ()) = graph
        .add_pass("known-color", |pass| {
            pass.read_texture_with_use(
                texture,
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
                texture_contract().uses,
                texture_contract().layout,
                texture_contract().descriptor_usage,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    let authority = plan.pass(pass).unwrap();
    authority
        .require_texture(
            TextureSlot::from(texture),
            TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
            AccessKind::Read,
            texture_contract(),
        )
        .unwrap();
    for mask in [8, TextureAspects::COLOR.0 | 8] {
        assert_eq!(
            authority.require_texture(
                TextureSlot::from(texture),
                TextureRange::new(TextureAspects(mask), 0..1, 0..1),
                AccessKind::Read,
                texture_contract()
            ),
            Err(AccessError::TextureRangeOutsideDeclaration)
        );
    }
}

#[test]
fn unions_compatible_contract_groups_and_keeps_kind_and_contract_errors() {
    let stronger = BufferAccessContract {
        uses: BufferUses {
            stages: StageMask::COPY | StageMask::VERTEX,
            accesses: AccessMask::COPY_READ,
        },
        descriptor_usage: BufferUsage::COPY_SRC,
    };
    let mut graph = GraphBuilder::default();
    let buffer = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("mixed-contracts", |pass| {
            pass.read_with_use(buffer, 0..8, stronger.uses, stronger.descriptor_usage)?;
            pass.read_with_use(buffer, 8..16, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    let access = plan.pass(pass).unwrap();
    assert!(
        access
            .require_buffer(
                BufferSlot::from(buffer),
                0..16,
                AccessKind::Read,
                buffer_contract()
            )
            .is_ok()
    );
    assert_eq!(
        access.require_buffer(BufferSlot::from(buffer), 0..16, AccessKind::Read, stronger),
        Err(AccessError::BufferRangeOutsideDeclaration)
    );
    assert_eq!(
        access.require_buffer(
            BufferSlot::from(buffer),
            0..8,
            AccessKind::Write,
            buffer_contract()
        ),
        Err(AccessError::IncompatibleBufferContract)
    );
}

#[test]
fn rejects_a_stronger_request_when_only_the_weaker_contract_is_declared() {
    let stronger = BufferAccessContract {
        uses: BufferUses {
            stages: StageMask::COPY | StageMask::VERTEX,
            accesses: AccessMask::COPY_READ,
        },
        descriptor_usage: BufferUsage::COPY_SRC,
    };
    let mut graph = GraphBuilder::default();
    let buffer = imported_buffer(&mut graph);
    let (pass, ()) = graph
        .add_pass("weaker-only", |pass| {
            pass.read_with_use(buffer, 0..8, buffer_contract().uses, BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    assert_eq!(
        plan.pass(pass).unwrap().require_buffer(
            BufferSlot::from(buffer),
            0..8,
            AccessKind::Read,
            stronger,
        ),
        Err(AccessError::IncompatibleBufferContract)
    );
}

#[test]
fn preserves_texture_mip_layer_and_aspect_union_coverage() {
    let mut graph = GraphBuilder::default();
    let color = imported_texture(&mut graph);
    let depth_stencil = imported_depth_stencil_texture(&mut graph);
    let (pass, ()) = graph
        .add_pass("texture-union", |pass| {
            for range in [
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..2),
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..2),
                TextureRange::new(TextureAspects::COLOR, 1..2, 0..2),
            ] {
                pass.read_texture_with_use(
                    color,
                    range,
                    texture_contract().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
            }
            for aspects in [TextureAspects::DEPTH, TextureAspects::STENCIL] {
                pass.read_texture_with_use(
                    depth_stencil,
                    TextureRange::new(aspects, 0..2, 0..2),
                    texture_contract().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
            }
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let plan = AccessPlan::compile(&graph, &[pass]).unwrap();
    let access = plan.pass(pass).unwrap();
    assert!(
        access
            .require_texture(
                TextureSlot::from(color),
                TextureRange::new(TextureAspects::COLOR, 0..2, 0..2),
                AccessKind::Read,
                texture_contract(),
            )
            .is_ok()
    );
    assert!(
        access
            .require_texture(
                TextureSlot::from(depth_stencil),
                TextureRange::new(TextureAspects::DEPTH | TextureAspects::STENCIL, 0..2, 0..2),
                AccessKind::Read,
                texture_contract(),
            )
            .is_ok()
    );
    assert_eq!(
        access.require_texture(
            TextureSlot::from(color),
            TextureRange::new(TextureAspects::COLOR, 0..2, 0..3),
            AccessKind::Read,
            texture_contract(),
        ),
        Err(AccessError::TextureRangeOutsideDeclaration)
    );
}
