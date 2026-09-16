//! Contract tests for the framebuffer domain: the resolve word,
//! renderbuffer kind facts, draw-buffer selection, and the agreement
//! between a render pass and the framebuffer descriptor it targets.

use super::*;

#[test]
fn mock_blit_is_the_resolve_word_and_rejects_incompatible_requests() {
    let mut api =
        MockGlFamilyApi::from_discovery(snapshot_with_fact(GlFormatResourceKind::Texture, 4));
    let multisample_texture = api
        .create_texture_resource(texture_desc(4))
        .expect("multisample texture");
    let multisample_framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![texture_view(multisample_texture, 4)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("multisample framebuffer");
    let texture = api
        .create_texture_resource(texture_desc(1))
        .expect("texture");
    let framebuffer = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![texture_view(texture, 1)],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer");
    let region = GlBlitRegion {
        src_offset: [0; 2],
        src_extent: [1, 1],
        dst_offset: [0; 2],
        dst_extent: [1, 1],
    };
    let color = GlBlitMask {
        color: true,
        depth: false,
        stencil: false,
    };

    // The one legal MSAA path: resolve with nearest filtering.
    assert!(
        api.blit_framebuffer(
            multisample_framebuffer,
            framebuffer,
            region,
            GlFilterMode::Linear,
            color
        )
        .is_err()
    );
    api.blit_framebuffer(
        multisample_framebuffer,
        framebuffer,
        region,
        GlFilterMode::Nearest,
        color,
    )
    .expect("resolve blit");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::BlitFramebuffer { .. })
    ));

    assert!(
        api.blit_framebuffer(
            framebuffer,
            framebuffer,
            region,
            GlFilterMode::Nearest,
            color
        )
        .is_err(),
        "identical source and destination are rejected"
    );
    assert!(
        api.blit_framebuffer(
            framebuffer,
            multisample_framebuffer,
            region,
            GlFilterMode::Nearest,
            GlBlitMask {
                color: false,
                depth: false,
                stencil: false,
            },
        )
        .is_err(),
        "an empty mask selects no plane"
    );
    assert!(
        api.blit_framebuffer(
            framebuffer,
            multisample_framebuffer,
            GlBlitRegion {
                src_extent: [0, 1],
                ..region
            },
            GlFilterMode::Nearest,
            color,
        )
        .is_err()
    );
    let unknown = FramebufferId::new(api.context_stamp(), 9_999, 0);
    assert!(
        api.blit_framebuffer(unknown, framebuffer, region, GlFilterMode::Nearest, color)
            .is_err()
    );
}

#[test]
fn mock_renderbuffer_allocation_uses_renderbuffer_kind_facts() {
    let mut api =
        MockGlFamilyApi::from_discovery(snapshot_with_fact(GlFormatResourceKind::Renderbuffer, 4));
    let desc = GlRenderBufferDesc {
        format: GlFormat::Rgba8Unorm,
        width: 8,
        height: 8,
        samples: 4,
    };
    let renderbuffer = api.create_render_buffer(desc).expect("renderbuffer");
    api.destroy_render_buffer(renderbuffer)
        .expect("destroy renderbuffer");
    assert!(
        api.calls()
            .contains(&MockCall::CreateRenderBuffer(renderbuffer))
    );
    assert!(
        api.calls()
            .contains(&MockCall::DestroyRenderBuffer(renderbuffer))
    );

    assert!(
        api.create_render_buffer(GlRenderBufferDesc { samples: 8, ..desc })
            .is_err(),
        "sample count must stay within the discovered max_samples"
    );
    assert!(
        api.create_render_buffer(GlRenderBufferDesc { width: 0, ..desc })
            .is_err()
    );

    let mut without_facts = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let trace_len = without_facts.calls().len();
    assert!(
        without_facts.create_render_buffer(desc).is_err(),
        "a texture-only format table cannot authorize a renderbuffer"
    );
    assert!(
        !without_facts.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CreateRenderBuffer(_)))
    );
}

#[test]
fn mock_framebuffer_draw_buffers_selection_is_validated_and_applied() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(texture_desc(1))
        .expect("texture");
    let view = texture_view(texture, 1);
    let descriptor = |color_count: usize, draw_buffers: Vec<u32>| GlFramebufferDescriptor {
        color_attachments: vec![view; color_count],
        depth_stencil_attachment: None,
        draw_buffers,
    };

    assert_eq!(
        api.create_framebuffer(&descriptor(1, vec![1])),
        Err(GlError::Validation {
            operation: "create-framebuffer",
            message: "invalid framebuffer descriptor".into(),
        })
    );
    assert!(api.create_framebuffer(&descriptor(1, vec![0, 0])).is_err());
    assert!(
        api.create_framebuffer(&descriptor(4, vec![0, 1, 2, 3, 0]))
            .is_err()
    );
    let mrt = api
        .create_framebuffer(&descriptor(2, vec![1, 0]))
        .expect("explicit MRT selection");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::CreateFramebuffer(created)) if *created == mrt
    ));
    let stored = api
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("default selection stays legal");
    assert_ne!(stored, mrt);
}

#[test]
fn mock_render_pass_must_match_the_stored_framebuffer_descriptor() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = api
        .create_texture_resource(mock_texture_desc())
        .expect("renderable texture");
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
    let mut mismatched = view;
    mismatched.width = 2;
    let trace_len = api.calls().len();
    assert!(
        api.begin_render_pass(&GlRenderPassDescriptor {
            framebuffer,
            color_attachments: vec![GlColorAttachment {
                view: mismatched,
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
        .is_err()
    );
    assert!(
        !api.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::BeginRenderPass(_)))
    );
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
    .expect("failed validation did not begin a pass");
}
