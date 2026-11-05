//! Private Deferred graph declaration facts, independent of GPU execution.

use super::*;
use crate::graph::{
    compiler::GraphCompiler,
    definition::{GraphBuilder, TextureRange},
    state::StatePlan,
};

fn texture(format: TextureFormat, usage: TextureUsage) -> fluxel_rhi::api::TextureDesc {
    fluxel_rhi::api::TextureDesc {
        label: None,
        dimension: fluxel_rhi::api::TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage,
        view_formats: Vec::new(),
    }
}

fn graph() -> (GraphBuilder, [GraphPass; 3]) {
    let mut graph = GraphBuilder::default();
    let albedo = graph
        .create_texture(texture(
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        ))
        .unwrap();
    let normal = graph
        .create_texture(texture(
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        ))
        .unwrap();
    let depth = graph
        .create_texture(texture(
            TextureFormat::Depth32Float,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        ))
        .unwrap();
    let target = graph
        .create_texture(texture(
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        ))
        .unwrap();
    let readback = graph
        .create_buffer_desc(fluxel_rhi::api::BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_DST,
            memory: fluxel_rhi::api::MemoryClass::Readback,
        })
        .unwrap();
    let (geometry, [albedo, normal, depth]) = graph
        .add_pass("geometry", |pass| {
            let albedo = pass.write_texture_with_use(
                albedo,
                whole_color(),
                color_write().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            let normal = pass.write_texture_with_use(
                normal,
                whole_color(),
                color_write().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            let depth = pass.write_texture_with_use(
                depth,
                whole_depth(),
                depth_write().uses,
                ImageLayout::DepthStencilAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            Ok([albedo, normal, depth])
        })
        .unwrap();
    let (lighting, target) = graph
        .add_pass("lighting", |pass| {
            pass.read_texture_with_use(
                albedo,
                whole_color(),
                sampled_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.read_texture_with_use(
                normal,
                whole_color(),
                sampled_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.read_texture_with_use(
                depth,
                whole_depth(),
                sampled_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.write_texture_with_use(
                target,
                whole_color(),
                color_write().uses,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )
        })
        .unwrap();
    let (readback_pass, ()) = graph
        .add_pass("readback", |pass| {
            pass.read_texture_with_use(
                target,
                whole_color(),
                texture_copy_read().uses,
                ImageLayout::CopySrc,
                TextureUsage::COPY_SRC,
            )?;
            pass.write_with_use(readback, 0..2048, copy_write().uses, BufferUsage::COPY_DST)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    (graph, [geometry, lighting, readback_pass])
}

#[test]
fn deferred_gbuffer_dependency_and_sampled_depth_are_real_graph_facts() {
    let (graph, [geometry, lighting, readback]) = graph();
    let compiled = GraphCompiler.compile(&graph).unwrap();
    assert_eq!(compiled.passes(), &[geometry, lighting, readback]);
    assert_eq!(compiled.pass_dependencies(1), &[0]);
    assert_eq!(compiled.pass_dependencies(2), &[1]);

    let lighting_accesses = &graph.passes[lighting.1].texture_accesses;
    assert_eq!(lighting_accesses.len(), 4);
    let depth = &lighting_accesses[2];
    assert_eq!(depth.range, whole_depth());
    assert_eq!(depth.contract.unwrap(), sampled_read());
}

#[test]
fn deferred_state_plan_transitions_gbuffer_writes_to_lighting_reads() {
    let (graph, [geometry, lighting, _]) = graph();
    let plan = StatePlan::compile(&graph, &[geometry, lighting]).unwrap();
    let geometry_transitions = plan.pass(geometry).unwrap();
    assert_eq!(geometry_transitions.textures.len(), 3);
    assert!(geometry_transitions.textures.iter().any(|transition| {
        transition.range == whole_depth()
            && transition.layout.1 == ImageLayout::DepthStencilAttachment
            && transition.to == depth_write().uses
    }));
    let lighting_transitions = plan.pass(lighting).unwrap();
    assert_eq!(lighting_transitions.textures.len(), 4);
    assert!(lighting_transitions.textures.iter().any(|transition| {
        transition.range == whole_depth()
            && transition.layout.0 == ImageLayout::DepthStencilAttachment
            && transition.layout.1 == ImageLayout::ShaderReadOnly
            && transition.to == sampled_read().uses
    }));
}

#[test]
fn deferred_sampled_depth_keeps_depth_aspect_distinct_from_color() {
    let (graph, [_, lighting, _]) = graph();
    let depth = &graph.passes[lighting.1].texture_accesses[2];
    let color = &graph.passes[lighting.1].texture_accesses[0];
    assert_eq!(
        depth.range,
        TextureRange::new(TextureAspects::DEPTH, 0..1, 0..1)
    );
    assert_eq!(
        color.range,
        TextureRange::new(TextureAspects::COLOR, 0..1, 0..1)
    );
    assert_ne!(depth.range, color.range);
}

// Canonical declaration used only to build actual-owner admission fixtures.
impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn compile_with<F>(
        device: &Device<B>,
        id: PlanId,
        frame: &DeferredFrame<B>,
        declare: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        let declaration = Self::declare_with(device, frame, declare)?;
        Self::compile_declared(device, id, declaration)
    }

    pub(crate) fn compile(
        device: &Device<B>,
        id: PlanId,
        frame: &DeferredFrame<B>,
    ) -> Result<Self> {
        Self::compile_with(device, id, frame, |context| {
            context.geometry(|pass| {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_gbuffer()
            })?;
            context.lighting(|pass| {
                pass.read_gbuffer()?;
                pass.write_main_color()
            })
        })
    }
}
