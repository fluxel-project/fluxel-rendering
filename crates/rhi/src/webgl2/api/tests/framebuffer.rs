//! Contract tests for the framebuffer domain: the resolve word,
//! renderbuffer kind facts, the renderbuffer attachment arm, draw-buffer
//! selection, and the agreement between a render pass and the framebuffer
//! descriptor it targets.

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

/// A renderbuffer is a first-class attachment, held to the same rule sequence
/// as a texture.
///
/// The gap this closes is a missing *route*, not a missing rule: both providers
/// can allocate a renderbuffer after the 0.15 allocation work, so before the
/// renderbuffer variant existed a caller could allocate one and then find it had
/// nowhere to attach. The rule sequence is the texture arm's, restated in
/// renderbuffer terms -- a renderbuffer addresses no mip level and no layer, so
/// its only legal view is level 0, layer 0, exactly one layer, at the
/// allocation's own extent and sample count -- and the test walks each rule in
/// the arm's order so an arm that reordered or dropped one fails here.
///
/// The last rule both arms end with, renderable evidence for the view's
/// storage class, runs through one shared helper on the accepting path as well:
/// the fixture records exactly one renderbuffer row, at the allocation's own
/// sample count, so an arm looking the row up under any other key finds nothing
/// and the accepted case below fails.
#[test]
fn mock_renderbuffer_attaches_through_the_same_rule_sequence_as_a_texture() {
    let mut api =
        MockGlFamilyApi::from_discovery(snapshot_with_fact(GlFormatResourceKind::Renderbuffer, 4));
    let renderbuffer = api
        .create_render_buffer(GlRenderBufferDesc {
            format: GlFormat::Rgba8Unorm,
            width: 8,
            height: 8,
            samples: 4,
        })
        .expect("renderbuffer");
    let view = GlTextureView {
        target: GlAttachmentTarget::Renderbuffer(renderbuffer),
        format: GlFormat::Rgba8Unorm,
        mip_level: 0,
        array_layer: 0,
        layer_count: 1,
        width: 8,
        height: 8,
        sample_count: 4,
    };
    let descriptor = |view| GlFramebufferDescriptor {
        color_attachments: vec![view],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };

    let framebuffer = api
        .create_framebuffer(&descriptor(view))
        .expect("a renderbuffer attaches");
    assert!(matches!(
        api.calls().last(),
        Some(MockCall::CreateFramebuffer(created)) if *created == framebuffer
    ));

    let validation = |message: &'static str| GlError::Validation {
        operation: "create-framebuffer",
        message: message.into(),
    };
    let rejected: [(GlTextureView, GlError); 6] = [
        (
            GlTextureView {
                format: GlFormat::Rgba8Srgb,
                ..view
            },
            validation("attachment view format does not match the allocation"),
        ),
        (
            GlTextureView {
                mip_level: 1,
                ..view
            },
            validation("attachment mip level is invalid"),
        ),
        (
            GlTextureView { width: 4, ..view },
            validation("attachment view extent does not match the allocation extent"),
        ),
        (
            GlTextureView {
                array_layer: 1,
                ..view
            },
            GlError::Unsupported {
                operation: "create-framebuffer",
                reason: "layered attachments are not part of this framebuffer slice",
            },
        ),
        (
            GlTextureView {
                sample_count: 1,
                ..view
            },
            validation("attachment sample count does not match the allocation"),
        ),
        // A second layer is refused as a view count this context never proved
        // rather than as a layered attachment: the descriptor's multiview gate
        // reads `layer_count` and runs before the per-view arms here. The arm's
        // own layered rule covers the other coordinate -- `array_layer`, which
        // the view-count gate never reads, and which is the case above.
        (
            GlTextureView {
                layer_count: 2,
                ..view
            },
            validation("multiview view count is not proved"),
        ),
    ];
    for (malformed, expected) in rejected {
        let trace_len = api.calls().len();
        assert_eq!(
            api.create_framebuffer(&descriptor(malformed)),
            Err(expected),
            "the renderbuffer arm rejects {malformed:?} as the texture arm would"
        );
        assert!(
            !api.calls()[trace_len..]
                .iter()
                .any(|call| matches!(call, MockCall::CreateFramebuffer(_))),
            "a rejected attachment never creates a framebuffer"
        );
    }

    // The route is only closed if the pass can name the same view: a caller
    // that could build a framebuffer but not draw into it would still have half
    // a feature.
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
    .expect("a pass draws into a renderbuffer attachment");
    assert_eq!(
        api.calls().last(),
        Some(&MockCall::BeginRenderPass(framebuffer))
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
