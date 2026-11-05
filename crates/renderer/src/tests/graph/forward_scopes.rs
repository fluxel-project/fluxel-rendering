//! Forward scoped-read state-machine tests using a real Graph transaction.

use super::super::phase::{
    COLOR_WRITE, TEXTURE_READS, UNIFORM_READS, UPLOAD_DEPENDENCY, VERTEX_READS,
    opaque_requirements_met,
};
use super::*;
use crate::graph::definition::GraphBuilder;
use fluxel_rhi::api::{
    TextureDesc, TextureDimension, TextureFormat, TextureUsage, VertexBufferLayout, VertexStepMode,
};

fn empty_reads(graph: &mut GraphBuilder) -> ForwardReadState {
    let uniform = graph.import_buffer(16).expect("synthetic defined uniform");
    ForwardReadState::new(Vec::new(), uniform, Vec::new(), Vec::new())
}

fn nonempty_reads(graph: &mut GraphBuilder) -> ForwardReadState {
    let stream = graph.import_buffer(64).expect("defined stream");
    let uniform = graph.import_buffer(64).expect("defined uniform");
    let texture = || TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [1, 1, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    };
    let textures = vec![
        graph
            .import_texture(texture())
            .expect("first defined texture"),
        graph
            .import_texture(texture())
            .expect("second defined texture"),
    ];
    let draw = |access: std::ops::Range<u64>, uniform_range: std::ops::Range<u64>, texture| {
        ForwardGraphDraw {
            vertex_streams: vec![0],
            vertex_layouts: vec![VertexBufferLayout {
                stride: 4,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            }],
            vertex_ranges: vec![access.clone()],
            vertices: 0..1,
            instances: 0..1,
            uniform_ranges: vec![uniform_range],
            texture,
            accesses: vec![Some(access)],
        }
    };
    ForwardReadState::new(
        vec![stream],
        uniform,
        textures,
        vec![draw(0..16, 0..8, 0), draw(16..32, 8..16, 1)],
    )
}

fn complete_scope(scope: &mut ForwardReadScope<'_, '_>) {
    scope.read_vertex_streams().expect("vertex declaration");
    scope.read_frame_uniforms().expect("uniform declaration");
    scope.read_sampled_textures().expect("texture declaration");
}

#[test]
fn empty_legacy_opaque_rejects_omitted_read_declarations() {
    let mut graph = GraphBuilder::default();
    let reads = empty_reads(&mut graph);
    assert!(!opaque_requirements_met(
        UPLOAD_DEPENDENCY | COLOR_WRITE,
        &reads
    ));
    assert!(opaque_requirements_met(
        UPLOAD_DEPENDENCY | COLOR_WRITE | VERTEX_READS | UNIFORM_READS | TEXTURE_READS,
        &reads
    ));
}

#[test]
fn completed_reads_without_scope_finish_are_rejected() {
    let mut graph = GraphBuilder::default();
    let mut reads = empty_reads(&mut graph);
    graph
        .add_pass("dropped-forward-scope", |pass| {
            let indices = reads.select(&[]).expect("empty scope starts");
            {
                let mut scope = ForwardReadScope {
                    pass,
                    state: &mut reads,
                    indices,
                    declared: 0,
                };
                scope.read_vertex_streams().expect("empty vertex reads");
                scope.read_frame_uniforms().expect("empty uniform reads");
                scope.read_sampled_textures().expect("empty texture reads");
            }
            assert!(!reads.complete());
            assert!(!opaque_requirements_met(
                UPLOAD_DEPENDENCY | COLOR_WRITE,
                &reads
            ));
            Ok(())
        })
        .expect("real Graph transaction remains valid");
}

#[test]
fn empty_scope_blocks_legacy_global_mode() {
    let mut graph = GraphBuilder::default();
    let mut reads = empty_reads(&mut graph);
    reads.select(&[]).expect("empty scope starts");
    assert!(reads.begin_global_reads().is_err());
}

#[test]
fn finished_empty_scope_satisfies_scoped_opaque_reads() {
    let mut graph = GraphBuilder::default();
    let mut reads = empty_reads(&mut graph);
    graph
        .add_pass("finished-forward-scope", |pass| {
            let indices = reads.select(&[]).expect("empty scope starts");
            let mut scope = ForwardReadScope {
                pass,
                state: &mut reads,
                indices,
                declared: 0,
            };
            scope.read_vertex_streams().expect("empty vertex reads");
            scope.read_frame_uniforms().expect("empty uniform reads");
            scope.read_sampled_textures().expect("empty texture reads");
            scope.finish().expect("scope completion");
            assert!(reads.complete());
            assert!(opaque_requirements_met(
                UPLOAD_DEPENDENCY | COLOR_WRITE,
                &reads
            ));
            Ok(())
        })
        .expect("real Graph transaction remains valid");
}

#[test]
fn disjoint_nonempty_scopes_declare_each_physical_footprint_once() {
    let mut graph = GraphBuilder::default();
    let mut reads = nonempty_reads(&mut graph);
    let streams = reads.streams.clone();
    let uniform = reads.uniform;
    let textures = reads.textures.clone();
    let (pass, ()) = graph
        .add_pass("disjoint-forward-scopes", |pass| {
            for index in [0, 1] {
                let indices = reads.select(&[index]).expect("disjoint draw");
                let mut scope = ForwardReadScope {
                    pass,
                    state: &mut reads,
                    indices,
                    declared: 0,
                };
                complete_scope(&mut scope);
                scope.finish().expect("complete scope");
            }
            assert!(reads.complete());
            Ok(())
        })
        .expect("Graph declarations succeed");
    let declarations = &graph.passes[pass.1];
    let buffer_facts: Vec<_> = declarations
        .accesses
        .iter()
        .map(|access| (access.buffer.resource, access.range.bytes()))
        .collect();
    assert_eq!(
        buffer_facts,
        vec![
            (streams[0].resource, 0..16),
            (uniform.resource, 0..8),
            (streams[0].resource, 16..32),
            (uniform.resource, 8..16),
        ]
    );
    let texture_facts: Vec<_> = declarations
        .texture_accesses
        .iter()
        .map(|access| access.texture.resource)
        .collect();
    assert_eq!(
        texture_facts,
        vec![textures[0].resource, textures[1].resource]
    );
}

#[test]
fn nonempty_scope_selection_rejects_duplicate_overlap_and_out_of_range_indices() {
    let mut graph = GraphBuilder::default();
    let mut reads = nonempty_reads(&mut graph);
    assert!(reads.select(&[0, 0]).is_err());
    let indices = reads.select(&[0]).expect("first draw claims scope");
    assert!(reads.select(&[0]).is_err());
    assert!(reads.select(&[2]).is_err());
    graph
        .add_pass("finish-first-scope", |pass| {
            let mut scope = ForwardReadScope {
                pass,
                state: &mut reads,
                indices,
                declared: 0,
            };
            complete_scope(&mut scope);
            scope.finish().expect("first scope finishes");
            Ok(())
        })
        .expect("Graph transaction succeeds");
}

#[test]
fn nonempty_scope_finish_rejects_each_omitted_read_class() {
    for omitted in [VERTEX_READS, UNIFORM_READS, TEXTURE_READS] {
        let mut graph = GraphBuilder::default();
        let mut reads = nonempty_reads(&mut graph);
        graph
            .add_pass("incomplete-forward-scope", |pass| {
                let indices = reads.select(&[0]).expect("scope starts");
                let mut scope = ForwardReadScope {
                    pass,
                    state: &mut reads,
                    indices,
                    declared: 0,
                };
                if omitted != VERTEX_READS {
                    scope.read_vertex_streams().expect("vertex declaration");
                }
                if omitted != UNIFORM_READS {
                    scope.read_frame_uniforms().expect("uniform declaration");
                }
                if omitted != TEXTURE_READS {
                    scope.read_sampled_textures().expect("texture declaration");
                }
                assert!(scope.finish().is_err());
                Ok(())
            })
            .expect("incomplete scope is observed without Graph failure");
        assert!(!reads.complete(), "omitted class {omitted}");
    }
}

#[test]
fn nonempty_scoped_and_global_modes_reject_mixing_in_both_directions() {
    let mut graph = GraphBuilder::default();
    let mut scoped_then_global = nonempty_reads(&mut graph);
    scoped_then_global.select(&[0]).expect("scope starts");
    assert!(scoped_then_global.begin_global_reads().is_err());

    let mut global_then_scoped = nonempty_reads(&mut graph);
    global_then_scoped
        .begin_global_reads()
        .expect("global mode starts");
    assert!(global_then_scoped.select(&[0]).is_err());
}
