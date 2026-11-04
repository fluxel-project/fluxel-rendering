use super::*;
use fluxel_rhi::api::{AccessMask, StageMask, TextureDimension};

fn declaration() -> GraphBuilder {
    let mut graph = GraphBuilder::default();
    let buffer = graph
        .import_buffer_desc(
            BufferDesc {
                label: None,
                size: 64,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            },
            BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
        )
        .unwrap();
    let texture = graph
        .import_texture_with_state(
            TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [4, 4, 1],
                array_layers: 1,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING,
                view_formats: Vec::new(),
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
            ImageLayout::ShaderReadOnly,
        )
        .unwrap();
    let (first, ()) = graph
        .add_pass("buffer", |pass| {
            pass.read_with_use(
                buffer,
                0..16,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
                BufferUsage::COPY_SRC,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph
        .add_pass("texture", |pass| {
            pass.depends_on(first)?;
            pass.read_texture_with_use(
                texture,
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
                TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph.export(buffer).unwrap();
    graph.export_texture(texture).unwrap();
    graph
}

#[test]
fn forward_declaration_equivalence_ignores_graph_provenance_and_debug_labels() {
    let cached = declaration();
    let mut candidate = declaration();
    assert_ne!(cached.graph, candidate.graph);
    candidate.textures[0].desc.label = Some("another owner".into());
    candidate.passes[0].name = "another label".into();
    assert!(cached.same_declaration(&candidate));
    assert!(candidate.same_declaration(&cached));
    crate::graph::compiler::GraphCompiler
        .compile(&cached)
        .unwrap();
}

#[test]
fn forward_declaration_rejects_changed_resource_facts() {
    let cached = declaration();
    for change in 0..10 {
        let mut candidate = declaration();
        match change {
            0 => candidate.buffers[0].size += 16,
            1 => candidate.buffers[0].memory = MemoryClass::Upload,
            2 => candidate.buffers[0].usage = BufferUsage::COPY_DST,
            3 => candidate.buffers[0].initial = None,
            4 => candidate.textures[0].desc.extent = [8, 4, 1],
            5 => candidate.textures[0].desc.format = TextureFormat::Bgra8Unorm,
            6 => candidate.textures[0].desc.usage = TextureUsage::COPY_DST,
            7 => candidate.textures[0].initial.as_mut().unwrap().1 = ImageLayout::General,
            8 => candidate.buffers[0].versions[0].defined.clear(),
            9 => candidate.textures[0].versions[0].defined.clear(),
            _ => unreachable!(),
        }
        assert!(
            !cached.same_declaration(&candidate),
            "resource mutation {change}"
        );
    }
}

#[test]
fn forward_declaration_rejects_changed_access_dependencies_and_outputs() {
    let cached = declaration();
    for change in 0..9 {
        let mut candidate = declaration();
        match change {
            0 => candidate.passes[0].accesses[0].range = ResourceRange(16..32),
            1 => {
                candidate.passes[0].accesses[0]
                    .contract
                    .as_mut()
                    .unwrap()
                    .uses
                    .stages = StageMask::VERTEX
            }
            2 => candidate.passes[0].accesses.clear(),
            3 => candidate.passes[1].dependencies.clear(),
            4 => {
                candidate.passes[1].texture_accesses[0]
                    .contract
                    .as_mut()
                    .unwrap()
                    .layout = ImageLayout::General
            }
            5 => candidate.passes[1].texture_accesses.clear(),
            6 => candidate.passes[0].side_effect = false,
            7 => candidate.exports.clear(),
            8 => candidate.texture_exports.clear(),
            _ => unreachable!(),
        }
        assert!(
            !cached.same_declaration(&candidate),
            "access mutation {change}"
        );
    }
}

#[test]
fn forward_declaration_rejects_foreign_or_stale_handles_before_normalization() {
    let cached = declaration();
    let mut candidate = declaration();
    candidate.passes[0].accesses[0].buffer = cached.passes[0].accesses[0].buffer;
    assert!(!cached.same_declaration(&candidate));
    let mut candidate = declaration();
    candidate.passes[1].dependencies[0] = cached.passes[0].id;
    assert!(!cached.same_declaration(&candidate));
    let mut candidate = declaration();
    candidate.exports[0].nonce += 1;
    assert!(!cached.same_declaration(&candidate));
}
