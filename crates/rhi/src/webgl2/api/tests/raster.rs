//! Contract tests for render pass, pipeline and draw domain recording.

use super::*;

#[test]
fn mock_records_render_pass_pipeline_and_draw_domains() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(mock_texture_desc())
        .expect("texture");
    let view = GlTextureView {
        target: GlAttachmentTarget::Texture(texture),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 1,
        height: 1,
        sample_count: 1,
    };
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    let program = api
        .create_program(&GlProgramDescriptor {
            kind: GlProgramKind::Raster {
                vertex: GlShaderSource {
                    stage: GlShaderStage::Vertex,
                    dialect: GlShaderDialect::Embedded { version: 300 },
                    entry_point: "main".into(),
                    source_hash: ShaderSourceHash([1; 32]),
                    text: "v".into(),
                    debug_name: None,
                },
                fragment: GlShaderSource {
                    stage: GlShaderStage::Fragment,
                    dialect: GlShaderDialect::Embedded { version: 300 },
                    entry_point: "main".into(),
                    source_hash: ShaderSourceHash([2; 32]),
                    text: "f".into(),
                    debug_name: None,
                },
            },
            layout: GlPipelineLayout { bindings: vec![] },
            debug_name: None,
        })
        .expect("program")
        .0;
    let vao = api
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .expect("vao");
    api.begin_render_pass(&GlRenderPassDescriptor {
        framebuffer,
        color_attachments: vec![GlColorAttachment {
            view,
            resolve_target: None,
            load: GlLoadOp::Clear,
            store: GlStoreOp::Store,
            clear: GlColorClearValue {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            },
        }],
        depth_stencil_attachment: None,
    })
    .expect("pass");
    api.set_raster_pipeline(&GlRasterPipeline {
        program,
        vertex_array: vao,
        state: GlRasterState {
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u32::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        },
    })
    .expect("pipeline");
    api.draw_raster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    }))
    .expect("draw");
    api.end_render_pass().expect("end pass");
    assert!(api.calls().ends_with(&[
        MockCall::BeginRenderPass(framebuffer),
        MockCall::SetRasterPipeline {
            program,
            vertex_array: vao
        },
        MockCall::DrawRaster(GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex: 0,
            vertex_count: 3,
            instance_count: 1
        })),
        MockCall::EndRenderPass,
    ]));
}
