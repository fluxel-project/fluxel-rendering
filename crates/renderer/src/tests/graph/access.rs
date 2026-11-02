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
