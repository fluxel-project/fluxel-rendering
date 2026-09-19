//! Tests for the integration point: the wiring the domain modules are written
//! against, exercised through one domain so that a change to the domain contract
//! shows up here before it shows up in seven other modules.
//!
//! The required domains are reached through one of them here; the two roles whose
//! entry points carry the optional bound are in [`optional`], which needs a
//! machine over a different mock and therefore a file of its own.

use super::*;
use crate::webgl2::api::tests::{snapshot, texture_desc, texture_view};
use crate::webgl2::api::{
    GlColorAttachment, GlColorClearValue, GlFamilyProfile, GlFramebufferApi, GlLoadOp,
    GlResourceApi, GlStoreOp, MockCall, MockGlFamilyApi,
};

/// A machine over the mock and a pass that names a framebuffer the machine
/// built for it, which is the ordering a real caller uses.
fn machine_with_a_pass() -> (
    GlStateMachine<MockGlFamilyApi>,
    GlFramebufferDescriptor,
    GlRenderPassDescriptor,
) {
    machine_with_a_pass_in(ExecutionMode::Optimized)
}

/// The same machine, in a named mode.
///
/// Every other helper here builds the optimized machine, because that is the
/// one production runs.  The oracle is built explicitly where a test is about
/// what the oracle does differently.
fn machine_with_a_pass_in(
    mode: ExecutionMode,
) -> (
    GlStateMachine<MockGlFamilyApi>,
    GlFramebufferDescriptor,
    GlRenderPassDescriptor,
) {
    let mut machine = GlStateMachine::with_mode(
        MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2)),
        mode,
    );
    let texture = machine
        .backend()
        .create_texture_resource(texture_desc(1))
        .expect("attachment texture");
    let descriptor = GlFramebufferDescriptor {
        color_attachments: vec![texture_view(texture, 1)],
        depth_stencil_attachment: None,
        draw_buffers: vec![],
    };
    let (framebuffer, _) = machine
        .framebuffer_for(&descriptor)
        .expect("the machine derives the framebuffer");
    let pass = GlRenderPassDescriptor {
        framebuffer,
        color_attachments: vec![GlColorAttachment {
            view: texture_view(texture, 1),
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
    };
    (machine, descriptor, pass)
}

fn calls_of(
    machine: &mut GlStateMachine<MockGlFamilyApi>,
    matched: fn(&MockCall) -> bool,
) -> usize {
    machine
        .backend()
        .calls()
        .iter()
        .filter(|call| matched(call))
        .count()
}

/// Applies a boundary whose effect the test does not assert.
///
/// [`SessionEffects`] is `#[must_use]` because a caller that ignores a pass
/// that ended will skip the pipeline it needs.  A setup step that
/// deliberately does not care states that here rather than at each call.
fn apply(machine: &mut GlStateMachine<MockGlFamilyApi>, pass: Option<GlRenderPassDescriptor>) {
    let boundary = match pass {
        Some(pass) => machine.begin_pass(pass),
        None => machine.end_pass(),
    };
    let _ = boundary.expect("the boundary applies");
}

#[test]
fn the_machine_derives_one_framebuffer_and_opens_the_pass_over_it() {
    let (mut machine, descriptor, pass) = machine_with_a_pass();

    let effects = machine.begin_pass(pass.clone()).expect("the pass opens");
    assert!(effects.pass_began);
    // The derived framebuffer is the machine's to destroy, and asking for
    // the same attachment set again returns the same object.
    let (again, owned) = machine
        .framebuffer_for(&descriptor)
        .expect("the framebuffer is served from the cache");
    assert_eq!(again, pass.framebuffer);
    assert!(!owned);
    assert_eq!(
        calls_of(&mut machine, |call| matches!(
            call,
            MockCall::CreateFramebuffer(_)
        )),
        1
    );

    apply(&mut machine, None);
    assert_eq!(machine.session().framebuffer_records(), 1);
}

#[test]
fn shutdown_destroys_what_the_machine_derived() {
    let (mut machine, _, pass) = machine_with_a_pass();
    apply(&mut machine, Some(pass));
    apply(&mut machine, None);
    assert_eq!(
        calls_of(&mut machine, |call| matches!(
            call,
            MockCall::DestroyFramebuffer(_)
        )),
        0
    );

    machine.shutdown();

    assert_eq!(
        calls_of(&mut machine, |call| matches!(
            call,
            MockCall::DestroyFramebuffer(_)
        )),
        1
    );
    assert_eq!(machine.session().framebuffer_records(), 0);
    assert_eq!(
        machine.counters().lifecycle.driver_errors,
        0,
        "the mock accepts every deletion"
    );
}

/// The oracle must not skip, and that is the whole point of it.
///
/// This is the test the mode's own accessor test was standing in for: it
/// asserts the property on a *machine*, through a domain, rather than
/// asserting that `ExecutionMode` reports itself.  A redundant request is the
/// case that distinguishes the two modes, so it is the case that is run: an
/// optimized machine re-opening the pass it already has skips the boundary,
/// and an oracle machine re-establishes it.  If the oracle ever skipped, its
/// trace would be a trace no mirror-free machine could have produced, and the
/// differential comparison it exists for would prove nothing.
#[test]
fn an_oracle_machine_emits_a_redundant_boundary_the_optimized_one_skips() {
    let (mut optimized, _, pass) = machine_with_a_pass();
    apply(&mut optimized, Some(pass.clone()));
    let before = calls_of(&mut optimized, |call| {
        matches!(call, MockCall::EndRenderPass)
    });
    apply(&mut optimized, Some(pass.clone()));
    let after = calls_of(&mut optimized, |call| {
        matches!(call, MockCall::EndRenderPass)
    });
    assert_eq!(
        after, before,
        "the optimized machine proved the boundary redundant and skipped it"
    );
    assert_eq!(optimized.session().mode(), ExecutionMode::Optimized);

    let (mut oracle, _, pass) = machine_with_a_pass_in(ExecutionMode::Oracle);
    apply(&mut oracle, Some(pass.clone()));
    let before = calls_of(&mut oracle, |call| matches!(call, MockCall::EndRenderPass));
    apply(&mut oracle, Some(pass));
    let after = calls_of(&mut oracle, |call| matches!(call, MockCall::EndRenderPass));
    assert_eq!(
        after,
        before + 1,
        "the oracle re-established the boundary instead of skipping it"
    );
    assert_eq!(oracle.session().mode(), ExecutionMode::Oracle);
}

mod optional;
