//! Evidence for the multiview gate at both framebuffer call sites.
//!
//! The gate is a capability fact rather than a descriptor shape rule, and both
//! executable providers apply it, so the recorder has to apply it at the same
//! point in its own order.  Both halves are needed: a case above the recorded
//! limit proves the limit is read, and a case at the limit proves the gate is a
//! limit rather than a refusal of the whole domain.

use super::*;

#[test]
fn the_multiview_gate_rejects_a_view_count_the_context_never_proved() {
    // Above the recorded limit: two views are this context's whole capacity, and
    // nothing else about this view is wrong, so the count is the only reason
    // left to refuse it.
    let mut api = webgl2_recorder();
    let view = target_view(&mut api, 3);
    api.clear_calls();
    let error = api
        .create_framebuffer(&framebuffer_descriptor(view))
        .expect_err("three views exceed the proved count of two");
    assert!(matches!(
        &error,
        GlError::Validation { operation, .. } if *operation == "create-framebuffer"
    ));
    assert_rejected(api.calls());

    // And on a context that proved no view count at all, where even the smallest
    // multiview request is refused.  Without this half, a gate that only
    // compared against a limit would accept multiview on a context that has
    // none, which is exactly the hole the missing gate left open.
    let mut api = recorder();
    let view = target_view(&mut api, 2);
    api.clear_calls();
    assert!(
        api.create_framebuffer(&framebuffer_descriptor(view))
            .is_err(),
        "two views are not available where no multiview route was proved"
    );
    assert_rejected(api.calls());
}

#[test]
fn the_multiview_gate_accepts_the_view_count_the_context_proved() {
    let mut api = webgl2_recorder();
    let view = target_view(&mut api, 2);
    let framebuffer = api
        .create_framebuffer(&framebuffer_descriptor(view))
        .expect("two views are exactly the proved count");
    // A gate that refused the limit itself would be indistinguishable from one
    // that refused multiview outright, so the accept case is what proves the
    // recorded limit is read rather than only the capability.
    assert_eq!(
        api.calls().last(),
        Some(&MockCall::CreateFramebuffer(framebuffer))
    );
    // The pass is where a multiview request is actually obeyed, so the same
    // count has to survive the pass gate too; a recorder gating only creation
    // would let the pass render one layer and silently drop the rest.
    assert_eq!(
        api.begin_render_pass(&pass_descriptor(framebuffer, view)),
        Ok(())
    );
    assert_eq!(
        api.calls().last(),
        Some(&MockCall::BeginRenderPass(framebuffer))
    );
}
