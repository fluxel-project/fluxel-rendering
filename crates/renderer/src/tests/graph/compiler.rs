use super::*;
use fluxel_rhi::api::{
    AccessMask, BufferDesc, BufferUsage, BufferUses, MemoryClass, StageMask, TextureAspects,
    TextureDesc, TextureDimension, TextureFormat, TextureUsage,
};

fn read_graph(range: std::ops::Range<u64>, stage: StageMask) -> CompiledGraph {
    let mut graph = GraphBuilder::default();
    let source = graph
        .import_buffer_desc(
            BufferDesc {
                label: None,
                size: 32,
                usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
                memory: MemoryClass::Upload,
            },
            BufferUses {
                stages: StageMask::HOST,
                accesses: AccessMask::HOST_WRITE,
            },
        )
        .unwrap();
    graph
        .add_pass("read", |pass| {
            pass.read_with_use(
                source,
                range,
                BufferUses {
                    stages: stage,
                    accesses: AccessMask::COPY_READ,
                },
                BufferUsage::COPY_SRC,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    GraphCompiler.compile(&graph).unwrap()
}

#[test]
fn compatible_buffer_range_changes_reuse_the_same_logical_plan() {
    let cached = read_graph(0..8, StageMask::COPY);
    let candidate = read_graph(16..32, StageMask::COPY);
    assert_eq!(cached.recompile_reason(&candidate), None);
}

#[test]
fn changed_buffer_use_contract_requires_recompilation() {
    let cached = read_graph(0..8, StageMask::COPY);
    let candidate = read_graph(0..8, StageMask::VERTEX_INPUT);
    assert_eq!(
        cached.recompile_reason(&candidate),
        Some(RecompileReason::BufferAccess)
    );
    assert_eq!(
        cached.recompile_reason_for_dynamic_buffers(&candidate),
        Some(RecompileReason::BufferAccess)
    );
}

#[test]
fn changed_dependency_edges_require_recompilation() {
    fn graph(with_edge: bool) -> CompiledGraph {
        let mut graph = GraphBuilder::default();
        let first = graph
            .add_pass("first", |pass| {
                pass.side_effect();
                Ok(())
            })
            .unwrap()
            .0;
        graph
            .add_pass("second", |pass| {
                if with_edge {
                    pass.depends_on(first)?;
                }
                pass.side_effect();
                Ok(())
            })
            .unwrap();
        GraphCompiler.compile(&graph).unwrap()
    }
    assert_eq!(
        graph(false).recompile_reason(&graph(true)),
        Some(RecompileReason::Dependencies)
    );
}

#[test]
fn changed_texture_subresources_require_recompilation() {
    fn graph(mip: u32) -> CompiledGraph {
        let mut graph = GraphBuilder::default();
        let texture = graph.create_texture(texture_desc(2, 1)).unwrap();
        graph
            .add_pass("write", |pass| {
                let _ = pass.write_texture_with_use(
                    texture,
                    subresource(mip..mip + 1, 0..1),
                    fluxel_rhi::api::TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    },
                    fluxel_rhi::api::ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )?;
                pass.side_effect();
                Ok(())
            })
            .unwrap();
        GraphCompiler.compile(&graph).unwrap()
    }
    assert_eq!(
        graph(0).recompile_reason(&graph(1)),
        Some(RecompileReason::TextureAccess)
    );
}

fn texture_desc(mips: u32, layers: u32) -> TextureDesc {
    TextureDesc {
        label: Some("graph test texture".to_owned()),
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: layers,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: mips,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn subresource(
    mips: std::ops::Range<u32>,
    layers: std::ops::Range<u32>,
) -> super::super::definition::TextureRange {
    super::super::definition::TextureRange::new(TextureAspects::COLOR, mips, layers)
}

#[test]
fn rejects_undefined_reads_and_invalid_ranges() {
    let mut graph = GraphBuilder::default();
    let undefined = graph.create_buffer(8).unwrap();
    assert_eq!(
        graph.add_pass("bad", |pass| pass.read(undefined, 0..8)),
        Err(GraphError::UndefinedRead)
    );
    let imported = graph.create_buffer(8).unwrap();
    assert_eq!(
        graph.add_pass("range", |pass| pass.read(imported, 8..9)),
        Err(GraphError::InvalidRange)
    );
}

#[test]
fn partial_writes_do_not_define_the_rest_of_a_version() {
    let mut graph = GraphBuilder::default();
    let transient = graph.create_buffer(8).unwrap();
    let (_, partial) = graph
        .add_pass("partial", |pass| pass.write(transient, 0..4))
        .unwrap();
    assert_eq!(
        graph.add_pass("bad-read", |pass| pass.read(partial, 0..8)),
        Err(GraphError::UndefinedRead)
    );
    assert!(
        graph
            .add_pass("good-read", |pass| pass.read(partial, 0..4))
            .is_ok()
    );
}

#[test]
fn adjacent_partial_writes_merge_definedness_but_holes_remain_rejected() {
    let mut graph = GraphBuilder::default();
    let transient = graph.create_buffer(12).unwrap();
    let (_, first) = graph
        .add_pass("first", |pass| pass.write(transient, 0..4))
        .unwrap();
    let (_, adjacent) = graph
        .add_pass("adjacent", |pass| pass.write(first, 4..8))
        .unwrap();
    assert!(
        graph
            .add_pass("merged-read", |pass| pass.read(adjacent, 0..8))
            .is_ok()
    );
    assert_eq!(
        graph.add_pass("hole-read", |pass| pass.read(adjacent, 0..12)),
        Err(GraphError::UndefinedRead)
    );
}

#[test]
fn rejects_export_of_an_undefined_version() {
    let mut graph = GraphBuilder::default();
    let undefined = graph.create_buffer(8).unwrap();
    assert_eq!(graph.export(undefined), Err(GraphError::UndefinedRead));
}

#[test]
fn failed_authoring_restores_the_prior_definition() {
    let mut graph = GraphBuilder::default();
    let imported = graph.create_buffer(8).unwrap();
    let (_, written) = graph
        .add_pass("write", |pass| pass.write(imported, 0..4))
        .unwrap();
    let before_passes = graph.passes.len();
    let before_version = graph.buffers[0].current;
    assert_eq!(
        graph.add_pass("broken", |pass| {
            let partial = pass.write(written, 4..6)?;
            pass.read(partial, 6..8)
        }),
        Err(GraphError::UndefinedRead)
    );
    assert_eq!(graph.passes.len(), before_passes);
    assert_eq!(graph.buffers[0].current, before_version);
}

#[test]
fn failed_transaction_handles_and_passes_never_alias_later_work() {
    let mut graph = GraphBuilder::default();
    let buffer = graph.import_buffer(8).unwrap();
    let texture = graph.import_texture(texture_desc(1, 1)).unwrap();
    let mut escaped_buffer = None;
    let mut escaped_texture = None;
    let mut escaped_pass = None;
    assert_eq!(
        graph.add_pass("rollback", |pass| {
            escaped_pass = Some(pass.current());
            escaped_buffer = Some(pass.write(buffer, 0..8)?);
            escaped_texture = Some(pass.write_texture(texture, subresource(0..1, 0..1))?);
            Err::<(), _>(GraphError::UndefinedRead)
        }),
        Err(GraphError::UndefinedRead)
    );
    let (_, committed_buffer) = graph
        .add_pass("commit", |pass| pass.write(buffer, 0..8))
        .unwrap();
    assert_eq!(
        graph.add_pass("escaped-buffer", |pass| pass
            .read(escaped_buffer.unwrap(), 0..8)),
        Err(GraphError::StaleVersion)
    );
    assert_eq!(
        graph.add_pass("escaped-texture", |pass| pass
            .read_texture(escaped_texture.unwrap(), subresource(0..1, 0..1))),
        Err(GraphError::StaleVersion)
    );
    assert_eq!(
        graph.add_pass("escaped-pass", |pass| pass
            .depends_on(escaped_pass.unwrap())),
        Err(GraphError::UnknownPass)
    );
    graph.export(committed_buffer).unwrap();
}

#[test]
fn one_pass_can_read_then_write_the_same_logical_version() {
    let mut graph = GraphBuilder::default();
    let buffer = graph.import_buffer(8).unwrap();
    let texture = graph.import_texture(texture_desc(1, 1)).unwrap();
    let (pass, (next_buffer, next_texture)) = graph
        .add_pass("in-place", |pass| {
            pass.read(buffer, 0..8)?;
            let next_buffer = pass.write(buffer, 0..8)?;
            pass.read_texture(texture, subresource(0..1, 0..1))?;
            let next_texture = pass.write_texture(texture, subresource(0..1, 0..1))?;
            pass.side_effect();
            Ok((next_buffer, next_texture))
        })
        .unwrap();
    graph.export(next_buffer).unwrap();
    graph.export_texture(next_texture).unwrap();
    let compiled = GraphCompiler.compile(&graph).unwrap();
    assert_eq!(compiled.passes(), &[pass]);
}

#[test]
fn rejects_old_versions_after_a_successor_write() {
    let mut graph = GraphBuilder::default();
    let imported = graph.import_buffer(8).unwrap();
    let (_, written) = graph
        .add_pass("write", |pass| pass.write(imported, 0..8))
        .unwrap();
    assert_eq!(
        graph.add_pass("late-old-read", |pass| pass.read(imported, 0..8)),
        Err(GraphError::StaleVersion)
    );
    graph.export(written).unwrap();
}

#[test]
fn rejects_foreign_graph_handles_and_passes() {
    let mut first = GraphBuilder::default();
    let imported = first.import_buffer(4).unwrap();
    let (foreign_pass, ()) = first
        .add_pass("first", |pass| {
            pass.read(imported, 0..4)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let mut second = GraphBuilder::default();
    assert_eq!(
        second.add_pass("foreign-buffer", |pass| pass.read(imported, 0..4)),
        Err(GraphError::UnknownResource)
    );
    let local = second.import_buffer(4).unwrap();
    assert_eq!(
        second.add_pass("foreign-pass", |pass| {
            pass.read(local, 0..4)?;
            pass.depends_on(foreign_pass)
        }),
        Err(GraphError::UnknownPass)
    );
}

#[test]
fn rejects_self_dependencies_and_detects_private_cycle_corruption() {
    let mut graph = GraphBuilder::default();
    let imported = graph.import_buffer(4).unwrap();
    assert_eq!(
        graph.add_pass("self", |pass| {
            pass.read(imported, 0..4)?;
            pass.depends_on(pass.current())
        }),
        Err(GraphError::SelfDependency)
    );

    let (first, ()) = graph
        .add_pass("first", |pass| {
            pass.read(imported, 0..4)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let (second, ()) = graph
        .add_pass("second", |pass| {
            pass.read(imported, 0..4)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph.passes[first.1].dependencies.push(second);
    graph.passes[second.1].dependencies.push(first);
    assert!(matches!(
        GraphCompiler.compile(&graph),
        Err(GraphError::Cycle)
    ));
}

#[test]
fn tracks_raw_war_waw_and_lifetimes_in_author_order() {
    let mut graph = GraphBuilder::default();
    let imported = graph.import_buffer(16).unwrap();
    let (first, a) = graph
        .add_pass("write-a", |pass| pass.write(imported, 0..16))
        .unwrap();
    let (reader, ()) = graph
        .add_pass("read-a", |pass| pass.read(a, 0..16))
        .unwrap();
    let (second, b) = graph
        .add_pass("write-b", |pass| pass.write(a, 0..16))
        .unwrap();
    graph.export(b).unwrap();
    let compiled = GraphCompiler.compile(&graph).unwrap();
    assert_eq!(compiled.passes(), &[first, reader, second]);
    assert_eq!(compiled.lifetimes().len(), 2);
    let a_lifetime = compiled
        .lifetimes()
        .iter()
        .find(|lifetime| lifetime.buffer == a)
        .unwrap();
    assert_eq!(a_lifetime.first_pass, 0);
    assert_eq!(a_lifetime.last_pass, 1);
}

#[test]
fn culls_unobserved_work_but_keeps_side_effects() {
    let mut graph = GraphBuilder::default();
    let source = graph.import_buffer(4).unwrap();
    let dead_source = graph.import_buffer(4).unwrap();
    let _dead = graph
        .add_pass("dead", |pass| pass.write(dead_source, 0..4))
        .unwrap();
    let (effect, ()) = graph
        .add_pass("effect", |pass| {
            pass.read(source, 0..4)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let compiled = GraphCompiler.compile(&graph).unwrap();
    assert_eq!(compiled.passes(), &[effect]);
}

#[test]
fn texture_subresources_track_definedness_hazards_and_lifetimes() {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(texture_desc(2, 2)).unwrap();
    let (writer, written) = graph
        .add_pass("write-mip-zero", |pass| {
            pass.write_texture(texture, subresource(0..1, 0..1))
        })
        .unwrap();
    assert_eq!(
        graph.add_pass("undefined-mip", |pass| pass
            .read_texture(written, subresource(1..2, 0..1))),
        Err(GraphError::UndefinedRead)
    );
    let (reader, ()) = graph
        .add_pass("read-mip-zero", |pass| {
            pass.read_texture(written, subresource(0..1, 0..1))?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let (overwrite, _) = graph
        .add_pass("overwrite", |pass| {
            pass.write_texture(written, subresource(0..1, 0..1))
        })
        .unwrap();
    graph.passes[overwrite.1].side_effect = true;
    let compiled = GraphCompiler.compile(&graph).unwrap();
    assert_eq!(compiled.passes(), &[writer, reader, overwrite]);
    let lifetime = compiled
        .texture_lifetimes()
        .iter()
        .find(|item| item.texture == written)
        .unwrap();
    assert_eq!((lifetime.first_pass, lifetime.last_pass), (0, 1));
}

#[test]
fn texture_definedness_unions_adjacent_written_layers() {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(texture_desc(1, 2)).unwrap();
    let (_, first_layer) = graph
        .add_pass("write-layer-zero", |pass| {
            pass.write_texture(texture, subresource(0..1, 0..1))
        })
        .unwrap();
    let (_, both_layers) = graph
        .add_pass("write-layer-one", |pass| {
            pass.write_texture(first_layer, subresource(0..1, 1..2))
        })
        .unwrap();

    assert!(
        graph
            .add_pass("read-both-layers", |pass| {
                pass.read_texture(both_layers, subresource(0..1, 0..2))
            })
            .is_ok(),
        "adjacent written layers jointly define the requested range"
    );
}

#[test]
fn texture_authoring_is_transactional_and_rejects_foreign_or_stale_handles() {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(texture_desc(2, 1)).unwrap();
    let (_, written) = graph
        .add_pass("write", |pass| {
            pass.write_texture(texture, subresource(0..1, 0..1))
        })
        .unwrap();
    let prior_passes = graph.passes.len();
    let prior_version = graph.textures[0].current;
    assert_eq!(
        graph.add_pass("rollback", |pass| {
            let next = pass.write_texture(written, subresource(0..1, 0..1))?;
            pass.read_texture(next, subresource(1..2, 0..1))
        }),
        Err(GraphError::UndefinedRead)
    );
    assert_eq!(
        (graph.passes.len(), graph.textures[0].current),
        (prior_passes, prior_version)
    );
    assert_eq!(
        graph.add_pass("stale", |pass| pass
            .read_texture(texture, subresource(0..1, 0..1))),
        Err(GraphError::StaleVersion)
    );
    let mut foreign = GraphBuilder::default();
    assert_eq!(
        foreign.add_pass("foreign", |pass| pass
            .read_texture(written, subresource(0..1, 0..1))),
        Err(GraphError::UnknownResource)
    );
    assert_eq!(
        graph.export_texture(written),
        Err(GraphError::UndefinedRead)
    );
}

#[test]
fn imported_texture_is_defined_without_erasing_subresource_bounds() {
    let mut graph = GraphBuilder::default();
    let imported = graph.import_texture(texture_desc(2, 2)).unwrap();
    assert!(
        graph
            .add_pass("read-import", |pass| {
                pass.read_texture(imported, subresource(1..2, 1..2))?;
                pass.side_effect();
                Ok(())
            })
            .is_ok()
    );
    assert_eq!(
        graph.add_pass("invalid-layer", |pass| {
            pass.read_texture(imported, subresource(0..1, 2..3))
        }),
        Err(GraphError::InvalidRange)
    );
}
