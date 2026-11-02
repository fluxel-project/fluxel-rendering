use fluxel_rhi::api::{
    AccessMask, BufferDesc, BufferUsage, BufferUses, ImageLayout, MemoryClass, StageMask,
    TextureAspects, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
};

use super::*;
use crate::graph::definition::{GraphBuilder, TextureRange};

fn empty_buffer_use() -> BufferUses {
    BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    }
}

fn copy_write() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}

fn compute_read() -> BufferUses {
    BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    }
}

fn compute_write() -> BufferUses {
    BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    }
}

fn texture_read() -> TextureUses {
    TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    }
}

fn texture_copy_write() -> TextureUses {
    TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    }
}

fn texture_compute_read_write() -> TextureUses {
    TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    }
}

fn device_buffer(size: u64) -> BufferDesc {
    BufferDesc {
        label: None,
        size,
        usage: BufferUsage::COPY_DST | BufferUsage::STORAGE,
        memory: MemoryClass::Device,
    }
}

fn texture_desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

#[test]
fn same_state_overlapping_buffer_writes_keep_the_hazard() {
    let mut graph = GraphBuilder::default();
    let buffer = graph.create_buffer_desc(device_buffer(64)).unwrap();
    let (first, first_output) = graph
        .add_pass("first", |pass| {
            pass.write_with_use(buffer, 0..32, copy_write(), BufferUsage::COPY_DST)
        })
        .unwrap();
    let (second, second_output) = graph
        .add_pass("second", |pass| {
            pass.write_with_use(first_output, 32..64, copy_write(), BufferUsage::COPY_DST)
        })
        .unwrap();
    let (third, _) = graph
        .add_pass("third", |pass| {
            pass.write_with_use(second_output, 0..32, copy_write(), BufferUsage::COPY_DST)
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[first, second, third]).unwrap();
    assert_eq!(plan.pass(first).unwrap().buffers.len(), 1);
    assert!(plan.pass(second).unwrap().buffers.is_empty());
    let transitions = &plan.pass(third).unwrap().buffers;
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[0].range, 0..64);
    assert_eq!(transitions[0].from, copy_write());
    assert_eq!(transitions[0].to, copy_write());
}

#[test]
fn same_pass_compute_read_write_merges_the_declared_uses() {
    let mut graph = GraphBuilder::default();
    let buffer = graph
        .import_buffer_desc(device_buffer(64), empty_buffer_use())
        .unwrap();
    let (pass, _) = graph
        .add_pass("compute", |pass| {
            pass.read_with_use(buffer, 0..64, compute_read(), BufferUsage::STORAGE)?;
            pass.write_with_use(buffer, 0..64, compute_write(), BufferUsage::STORAGE)
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[pass]).unwrap();
    let transitions = &plan.pass(pass).unwrap().buffers;
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[0].to.stages, StageMask::COMPUTE);
    assert!(transitions[0].to.accesses.contains(AccessMask::SHADER_READ));
    assert!(
        transitions[0]
            .to
            .accesses
            .contains(AccessMask::SHADER_WRITE)
    );
}

#[test]
fn different_ranges_in_one_pass_produce_one_whole_buffer_state() {
    let mut graph = GraphBuilder::default();
    let buffer = graph
        .import_buffer_desc(device_buffer(64), empty_buffer_use())
        .unwrap();
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let (pass, _) = graph
        .add_pass("mixed-ranges", |pass| {
            pass.read_with_use(buffer, 0..32, copy_read, BufferUsage::COPY_SRC)?;
            pass.read_with_use(buffer, 32..64, compute_read(), BufferUsage::STORAGE)
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[pass]).unwrap();
    let transitions = &plan.pass(pass).unwrap().buffers;
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[0].range, 0..64);
    assert!(transitions[0].to.stages.contains(StageMask::COPY));
    assert!(transitions[0].to.stages.contains(StageMask::COMPUTE));
}

#[test]
fn same_state_raw_and_war_accesses_are_not_lost() {
    let mut graph = GraphBuilder::default();
    let buffer = graph.create_buffer_desc(device_buffer(64)).unwrap();
    let combined = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    };
    let (write, written) = graph
        .add_pass("write", |pass| {
            pass.write_with_use(buffer, 0..32, combined, BufferUsage::STORAGE)
        })
        .unwrap();
    let (read, _) = graph
        .add_pass("read", |pass| {
            pass.read_with_use(written, 0..32, combined, BufferUsage::STORAGE)
        })
        .unwrap();
    let (write_after_read, _) = graph
        .add_pass("write-after-read", |pass| {
            pass.write_with_use(written, 0..32, combined, BufferUsage::STORAGE)
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[write, read, write_after_read]).unwrap();
    assert_eq!(
        plan.pass(read).unwrap().buffers.len(),
        1,
        "RAW needs ordering"
    );
    assert_eq!(
        plan.pass(write_after_read).unwrap().buffers.len(),
        1,
        "WAR needs ordering"
    );
}

#[test]
fn texture_state_is_independent_per_mip_and_preserves_untouched_incoming_state() {
    let mut graph = GraphBuilder::default();
    let texture = graph
        .import_texture_with_state(texture_desc(), texture_read(), ImageLayout::ShaderReadOnly)
        .unwrap();
    let mip_zero = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);
    let mip_one = TextureRange::new(TextureAspects::COLOR, 1..2, 0..1);
    let (write, written) = graph
        .add_pass("write-mip-zero", |pass| {
            pass.write_texture_with_use(
                texture,
                mip_zero,
                texture_copy_write(),
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            )
        })
        .unwrap();
    let (read, _) = graph
        .add_pass("read-mip-one", |pass| {
            pass.read_texture_with_use(
                written,
                mip_one,
                texture_read(),
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[write, read]).unwrap();
    let write_transitions = &plan.pass(write).unwrap().textures;
    assert_eq!(write_transitions.len(), 1);
    assert_eq!(
        write_transitions[0].layout,
        (ImageLayout::ShaderReadOnly, ImageLayout::CopyDst)
    );
    assert!(plan.pass(read).unwrap().textures.is_empty());
}

#[test]
fn same_texture_state_orders_raw_once_without_repeating_the_write_history() {
    let mut graph = GraphBuilder::default();
    let texture = graph
        .create_texture(TextureDesc {
            usage: TextureUsage::STORAGE_BINDING,
            ..texture_desc()
        })
        .unwrap();
    let range = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);
    let uses = texture_compute_read_write();
    let (write, written) = graph
        .add_pass("write", |pass| {
            pass.write_texture_with_use(
                texture,
                range.clone(),
                uses,
                ImageLayout::General,
                TextureUsage::STORAGE_BINDING,
            )
        })
        .unwrap();
    let (first_read, ()) = graph
        .add_pass("first-read", |pass| {
            pass.read_texture_with_use(
                written,
                range.clone(),
                uses,
                ImageLayout::General,
                TextureUsage::STORAGE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let (second_read, ()) = graph
        .add_pass("second-read", |pass| {
            pass.read_texture_with_use(
                written,
                range,
                uses,
                ImageLayout::General,
                TextureUsage::STORAGE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();

    let plan = StatePlan::compile(&graph, &[write, first_read, second_read]).unwrap();
    assert_eq!(
        plan.pass(first_read).unwrap().textures.len(),
        1,
        "RAW needs ordering"
    );
    assert!(
        plan.pass(second_read).unwrap().textures.is_empty(),
        "the RAW barrier must consume the prior write history"
    );
}

#[test]
fn overlapping_texture_uses_with_different_layouts_are_rejected_before_the_pass() {
    let mut graph = GraphBuilder::default();
    let texture = graph
        .import_texture_with_state(texture_desc(), texture_read(), ImageLayout::ShaderReadOnly)
        .unwrap();
    let range = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);
    let (pass, _) = graph
        .add_pass("conflicting-layouts", |pass| {
            pass.read_texture_with_use(
                texture,
                range.clone(),
                texture_read(),
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.read_texture_with_use(
                texture,
                range,
                texture_copy_write(),
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            )
        })
        .unwrap();

    assert!(matches!(
        StatePlan::compile(&graph, &[pass]),
        Err(StateError::ConflictingTextureUse {
            pass: failed_pass,
            slot,
        }) if failed_pass == pass && slot == texture.into()
    ));
}

#[test]
fn fixed_memory_classes_reject_illegal_graph_contracts() {
    let mut graph = GraphBuilder::default();
    let upload = graph
        .import_buffer_desc(
            BufferDesc {
                memory: MemoryClass::Upload,
                ..device_buffer(64)
            },
            empty_buffer_use(),
        )
        .unwrap();
    let (pass, _) = graph
        .add_pass("illegal-upload-write", |pass| {
            pass.write_with_use(upload, 0..64, copy_write(), BufferUsage::COPY_DST)
        })
        .unwrap();

    assert!(matches!(
        StatePlan::compile(&graph, &[pass]),
        Err(StateError::IllegalFixedMemoryUse { .. })
    ));
}

#[test]
fn readback_accepts_only_copy_write_destinations() {
    let mut graph = GraphBuilder::default();
    let readback = graph
        .import_buffer_desc(
            BufferDesc {
                memory: MemoryClass::Readback,
                ..device_buffer(64)
            },
            empty_buffer_use(),
        )
        .unwrap();
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let (pass, _) = graph
        .add_pass("illegal-readback-read", |pass| {
            pass.read_with_use(readback, 0..64, copy_read, BufferUsage::COPY_SRC)
        })
        .unwrap();

    assert!(matches!(
        StatePlan::compile(&graph, &[pass]),
        Err(StateError::IllegalFixedMemoryUse { .. })
    ));
}
