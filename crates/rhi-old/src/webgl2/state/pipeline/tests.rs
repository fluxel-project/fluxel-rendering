//! Tests for the installed-pipeline mirror and the program records.
//!
//! Every test here runs the domain against the mock provider rather than a
//! hand-written double, because what is being tested is the *trace* -- which
//! calls were emitted, in what order, and how many -- and a double written for
//! these tests would answer with whatever trace the test expected.  The mock is
//! the same provider the Layer 1 contract suites run against, so a change in
//! Layer 1's own call surface shows up here as a failing trace rather than as a
//! mirror that quietly agrees with itself.
//!
//! This file owns only fixtures and assertions.  It declares no state of its
//! own, so a change to the domain's shape fails a test here rather than
//! requiring an edit here first.

use super::*;
use crate::webgl2::api::GlFamilyApi;
use crate::webgl2::api::tests::{snapshot, texture_desc, texture_view};
use crate::webgl2::api::{
    GlColorAttachment, GlColorClearValue, GlColorTargetState, GlCullMode, GlError, GlFamilyProfile,
    GlFramebufferApi, GlFramebufferDescriptor, GlFrontFace, GlLoadOp, GlMultisampleState,
    GlPipelineLayout, GlPrimitiveTopology, GlProgramKind, GlRasterState, GlRenderPassDescriptor,
    GlResourceApi, GlShaderApi, GlShaderDialect, GlShaderSource, GlShaderStage, GlStoreOp,
    GlVertexApi, GlVertexLayout, GlViewport, MockCall, MockGlFamilyApi, ShaderSourceHash,
    VertexArrayId,
};
use crate::webgl2::state::cache::DEFAULT_BUDGET;
use crate::webgl2::state::event::ScopedRawAccess;
use crate::webgl2::state::knowledge::DirtyDomains;

/// A backend with a pass open over a one-pixel attachment.
///
/// Layer 1 refuses a pipeline install outside a pass and the mock enforces
/// that the way both providers do, so a fixture that expects an install to
/// succeed has to open one.
fn backend_with_a_pass() -> MockGlFamilyApi {
    let mut backend = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let texture = backend
        .create_texture_resource(texture_desc(1))
        .expect("attachment texture");
    let view = texture_view(texture, 1);
    let framebuffer = backend
        .create_framebuffer(&GlFramebufferDescriptor {
            color_attachments: vec![view],
            depth_stencil_attachment: None,
            draw_buffers: vec![],
        })
        .expect("framebuffer over the attachment set");
    backend
        .begin_render_pass(&GlRenderPassDescriptor {
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
        .expect("the pass opens");
    backend
}

fn source(stage: GlShaderStage, seed: u8, text: &str) -> GlShaderSource {
    GlShaderSource {
        stage,
        dialect: GlShaderDialect::Embedded { version: 300 },
        entry_point: "main".into(),
        source_hash: ShaderSourceHash([seed; 32]),
        text: text.into(),
        debug_name: None,
    }
}

/// A raster program descriptor whose fragment source is the discriminator.
fn descriptor(fragment_text: &str) -> GlProgramDescriptor {
    GlProgramDescriptor {
        kind: GlProgramKind::Raster {
            vertex: source(GlShaderStage::Vertex, 1, "v"),
            fragment: source(GlShaderStage::Fragment, 2, fragment_text),
        },
        layout: GlPipelineLayout { bindings: vec![] },
        debug_name: None,
    }
}

fn vertex_array(backend: &mut MockGlFamilyApi) -> VertexArrayId {
    backend
        .create_vertex_array(&GlVertexLayout {
            buffers: vec![],
            attributes: vec![],
        })
        .expect("vertex array")
}

fn pipeline(
    program: ProgramId,
    vertex_array: VertexArrayId,
    cull_mode: GlCullMode,
) -> GlRasterPipeline {
    GlRasterPipeline {
        program,
        vertex_array,
        state: GlRasterState {
            topology: GlPrimitiveTopology::Triangles,
            cull_mode,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![GlColorTargetState {
                write_mask: 0x0f,
                blend: None,
            }],
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
    }
}

fn pipeline_state(mode: ExecutionMode) -> PipelineState {
    PipelineState::new(DEFAULT_BUDGET, mode)
}

/// Applies a pipeline whose effect the test does not assert.
///
/// [`PipelineEffects`] is `#[must_use]` because a caller that ignores an
/// install will let the geometry mirror skip a binding it needs.  A setup
/// step that deliberately does not care states that here, once, instead of
/// at each call site.
fn apply(state: &mut PipelineState, backend: &mut MockGlFamilyApi, counters: &mut StateCounters) {
    let _ = state
        .reconcile(backend, counters)
        .expect("the install applies");
}

fn calls_of(backend: &MockGlFamilyApi, matched: fn(&MockCall) -> bool) -> usize {
    backend.calls().iter().filter(|call| matched(call)).count()
}

fn installs(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| {
        matches!(call, MockCall::SetRasterPipeline { .. })
    })
}

fn links(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| matches!(call, MockCall::CreateProgram(_)))
}

fn destroys(backend: &MockGlFamilyApi) -> usize {
    calls_of(backend, |call| matches!(call, MockCall::DestroyProgram(_)))
}

#[test]
fn a_pipeline_that_is_already_installed_is_not_installed_again() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    let wanted = pipeline(program, vao, GlCullMode::None);

    state.set_pipeline(&wanted);
    let first = state
        .reconcile(&mut backend, &mut counters)
        .expect("the first install emits");
    assert_eq!(
        first,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(state.applied_pipeline(), Some(&wanted));

    // Re-installing the pipeline already in the driver is the common case in
    // a renderer that draws several objects with one pipeline.
    state.set_pipeline(&wanted);
    let second = state
        .reconcile(&mut backend, &mut counters)
        .expect("an unchanged pipeline is not a failure");
    assert_eq!(second, PipelineEffects::NONE);
    assert_eq!(installs(&backend), 1);
    // Two calls, not one: the link and the install are both this domain's
    // driver traffic, and only the second was skippable here.  The skip is
    // the whole point of the domain.
    assert_eq!(counters.domain_counts(StateDomain::Pipeline).emitted, 2);
    assert_eq!(counters.domain_counts(StateDomain::Pipeline).skipped, 1);
    assert_eq!(
        counters
            .domain_counts(StateDomain::Pipeline)
            .unknown_recoveries,
        1,
        "the first install recovered a pipeline the mirror had no belief about"
    );
}

#[test]
fn an_oracle_emits_the_redundant_install_the_optimized_mirror_skips() {
    // A skip is two decisions -- the mirror agrees, and this mode may act on
    // that -- and a domain that made only the first would produce one trace in
    // both modes.  The oracle exists to be the trace a machine with no mirror at
    // all would have produced, so a skip it should not have made is invisible to
    // the comparison it exists for.  Both instances are given the same two
    // requests: one install, and then the request that repeats it.
    let mut optimized_backend = backend_with_a_pass();
    let optimized_vao = vertex_array(&mut optimized_backend);
    let mut optimized = pipeline_state(ExecutionMode::Optimized);
    let mut optimized_counters = StateCounters::default();
    let (program, _, _) = optimized
        .program_for(
            &mut optimized_backend,
            &descriptor("f"),
            &mut optimized_counters,
        )
        .expect("the program links");
    let wanted = pipeline(program, optimized_vao, GlCullMode::None);

    let mut oracle_backend = backend_with_a_pass();
    let oracle_vao = vertex_array(&mut oracle_backend);
    let mut oracle = pipeline_state(ExecutionMode::Oracle);
    let mut oracle_counters = StateCounters::default();
    let (oracle_program, _, _) = oracle
        .program_for(&mut oracle_backend, &descriptor("f"), &mut oracle_counters)
        .expect("the oracle still links a usable program");
    let repeated = pipeline(oracle_program, oracle_vao, GlCullMode::None);

    optimized.set_pipeline(&wanted);
    let first = optimized
        .reconcile(&mut optimized_backend, &mut optimized_counters)
        .expect("the first install emits");
    assert_eq!(
        first,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    optimized.set_pipeline(&wanted);
    let skipped = optimized
        .reconcile(&mut optimized_backend, &mut optimized_counters)
        .expect("a redundant install is not a failure");

    oracle.set_pipeline(&repeated);
    let oracle_first = oracle
        .reconcile(&mut oracle_backend, &mut oracle_counters)
        .expect("the oracle installs the pipeline as well");
    assert_eq!(
        oracle_first,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    oracle.set_pipeline(&repeated);
    let re_emitted = oracle
        .reconcile(&mut oracle_backend, &mut oracle_counters)
        .expect("the oracle re-installs rather than skipping");

    // First observable: the trace.  Two requests cost the driver one install
    // under the optimized mirror and two under the oracle.
    assert_eq!(skipped, PipelineEffects::NONE);
    assert_eq!(installs(&optimized_backend), 1);
    assert_eq!(
        re_emitted,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&oracle_backend), 2);

    // Second observable, and the discriminating one: this domain's own tally.
    // It is what a report reads to say how much the mirror saved, and the trace
    // is what says whether a call reached the driver -- so an ungated skip would
    // show up as a saving the oracle claimed for a call the optimized instance
    // is supposed to be the only one allowed to drop.  The counts are asserted
    // per instance rather than as a difference, because the oracle's zero is the
    // half that fails when the mode is ignored.
    assert_eq!(
        optimized_counters
            .domain_counts(StateDomain::Pipeline)
            .skipped,
        1,
        "the optimized mirror proved the repeated install redundant"
    );
    assert_eq!(
        oracle_counters.domain_counts(StateDomain::Pipeline).skipped,
        0,
        "the oracle has nothing to report as skipped: it emitted every required call"
    );
}

#[test]
fn a_changed_pipeline_is_installed_again() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");

    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    // The redundancy unit is the whole pipeline, so a change to any one of
    // its rasterization values makes the request a different one.
    let changed = pipeline(program, vao, GlCullMode::Back);
    state.set_pipeline(&changed);
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("a changed pipeline installs");

    assert_eq!(
        effects,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&backend), 2);
    assert_eq!(state.applied_pipeline(), Some(&changed));
    assert_eq!(
        counters
            .domain_counts(StateDomain::Pipeline)
            .unknown_recoveries,
        1,
        "a changed request is not a recovery: the mirror had a belief"
    );
}

#[test]
fn a_reconcile_with_nothing_asked_for_emits_nothing() {
    // A pass that only clears, or a compute-only frame, runs the domains
    // without ever asking this one for a pipeline.
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("an empty desire is not a failure");

    assert_eq!(effects, PipelineEffects::NONE);
    assert_eq!(installs(&backend), 0);
    assert_eq!(counters.domain_counts(StateDomain::Pipeline).requests, 1);
    assert_eq!(counters.domain_counts(StateDomain::Pipeline).skipped, 1);
}

#[test]
fn a_pass_that_ended_makes_the_next_reconcile_install_again() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    let wanted = pipeline(program, vao, GlCullMode::None);
    state.set_pipeline(&wanted);
    apply(&mut state, &mut backend, &mut counters);

    // Layer 1's `end_render_pass` clears the provider's record of the
    // installed pipeline.  The session reports that and the machine applies
    // it here, so the mirror must stop claiming an install it can no longer
    // justify.
    state.pass_ended();
    assert!(!state.pipeline_installed());
    assert_eq!(state.applied_pipeline(), None);
    assert_eq!(
        state.desired_pipeline(),
        Some(&wanted),
        "the caller still wants this pipeline; only the driver's state was lost"
    );

    let again = state
        .reconcile(&mut backend, &mut counters)
        .expect("the install is repeated");
    assert_eq!(
        again,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&backend), 2);
    assert_eq!(
        counters
            .domain_counts(StateDomain::Pipeline)
            .unknown_recoveries,
        2,
        "the pass end left the mirror with no belief, and the re-install is a recovery"
    );
}

#[test]
fn a_program_is_linked_once_and_served_from_the_cache_afterwards() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    let (first, reflection, owned) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the first request links");
    assert!(
        !owned,
        "a retained program belongs to the cache, not to the caller"
    );
    // The mock reflects an empty program unless a test injects one, which
    // is what an un-decorated descriptor links to.
    assert!(reflection.assignments.is_empty() && reflection.vertex_inputs.is_empty());
    let (second, _, owned) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the second request is served from the cache");
    assert_eq!(first, second);
    assert!(!owned);

    assert_eq!(links(&backend), 1);
    assert_eq!(state.program_records(), 1);
    assert_eq!(counters.caches.misses, 1);
    assert_eq!(counters.caches.hits, 1);
    assert_eq!(counters.caches.created, 1);
    assert_eq!(counters.domain_counts(StateDomain::Pipeline).emitted, 1);
    // Two retained values on the first request -- the lookup key and the
    // record's own reflection -- and two more on the hit, where the key is
    // rebuilt and the reflection is cloned out of the table.
    assert_eq!(counters.steady_state_allocations, 4);
}

#[test]
fn a_program_for_a_different_descriptor_is_linked_separately() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();

    let (first, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the first program links");
    let (second, _, _) = state
        .program_for(&mut backend, &descriptor("g"), &mut counters)
        .expect("the second program links");

    assert_ne!(first, second);
    assert_eq!(links(&backend), 2);
    assert_eq!(state.program_records(), 2);
    assert_eq!(counters.caches.misses, 2);
    assert_eq!(counters.caches.hits, 0);
}

#[test]
fn linking_a_program_forgets_the_installed_pipeline() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the first program links");
    let wanted = pipeline(program, vao, GlCullMode::None);
    state.set_pipeline(&wanted);
    apply(&mut state, &mut backend, &mut counters);
    assert!(state.pipeline_installed());

    // Layer 1 links with the new program bound and unbinds it afterwards, so
    // the driver no longer holds the pipeline this mirror believes in.
    state
        .program_for(&mut backend, &descriptor("g"), &mut counters)
        .expect("the second program links");
    assert!(!state.pipeline_installed());

    state.set_pipeline(&wanted);
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the install is repeated");
    assert_eq!(
        effects,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&backend), 2);
}

#[test]
fn deleting_a_program_drops_its_record_and_the_pipeline_that_named_it() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    let destroyed = destroys(&backend);

    state.invalidate(
        &mut backend,
        &StateEvent::ProgramDeleted(program),
        &mut counters,
    );

    assert_eq!(state.program_records(), 0);
    assert_eq!(counters.caches.invalidated, 1);
    assert!(
        !state.pipeline_installed(),
        "an installed pipeline naming a deleted program is not installable"
    );
    assert_eq!(
        destroys(&backend),
        destroyed,
        "the event says the owner is deleting the program, so destroying it here would be the second deletion"
    );
    assert_eq!(
        counters.lifecycle.domain_invalidations, 0,
        "a single deletion is not a whole-domain invalidation"
    );

    // The next install needs a program, and the record that named the old
    // one is gone, so the domain links again rather than trusting it.
    let (replacement, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links again");
    assert_ne!(replacement, program);
    state.set_pipeline(&pipeline(replacement, vao, GlCullMode::None));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the install emits");
    assert_eq!(
        effects,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(links(&backend), 2);
    assert_eq!(installs(&backend), 2);
}

#[test]
fn deleting_a_vertex_array_forgets_the_pipeline_that_named_it() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let unrelated = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(installs(&backend), 1);
    let destroyed = destroys(&backend);

    // The negative control first: an array this pipeline does not name leaves
    // the claim alone.  Without it the test would pass against an arm that
    // invalidated on every array deletion, which is a different -- and wrong --
    // statement about what the mirror knows.
    state.invalidate(
        &mut backend,
        &StateEvent::VertexArrayDeleted(unrelated),
        &mut counters,
    );
    assert!(
        state.pipeline_installed(),
        "the install this mirror believes in does not name that array"
    );

    state.invalidate(
        &mut backend,
        &StateEvent::VertexArrayDeleted(vao),
        &mut counters,
    );

    // The claim covers the whole verb, and the verb binds this array.  Layer 1
    // unbinds it when the object is deleted, so the program and the rasterization
    // values being still installed does not make the claim true: one component of
    // its effect is gone.
    assert!(
        !state.pipeline_installed(),
        "an install whose vertex array was deleted is not what the driver holds"
    );
    assert_eq!(
        counters.lifecycle.domain_invalidations, 0,
        "a single deletion is not a whole-domain invalidation"
    );
    assert_eq!(
        destroys(&backend),
        destroyed,
        "the event says the owner is deleting the array, so destroying it here would be the second deletion"
    );

    // And the whole point of dropping the claim: the same request the mirror
    // would otherwise have skipped is emitted again, which is the install the
    // deletion made necessary.
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    let effects = state
        .reconcile(&mut backend, &mut counters)
        .expect("the install is emitted again");
    assert_eq!(
        effects,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&backend), 2);
}

#[test]
fn a_deleted_shader_object_is_not_a_program_cache_event() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let shader = backend
        .create_shader(&source(GlShaderStage::Vertex, 1, "v"))
        .expect("shader object");
    state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");

    state.invalidate(
        &mut backend,
        &StateEvent::ShaderDeleted(shader),
        &mut counters,
    );

    // Layer 1 links from lowered source *content* and its providers delete
    // the shader objects as soon as the link succeeds, so a deleted shader
    // object names nothing this domain's records hold.
    assert_eq!(state.program_records(), 1);
    assert_eq!(counters.caches.invalidated, 0);
}

#[test]
fn a_refused_install_reports_the_domain_and_leaves_no_installed_pipeline() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    backend.fail_next(GlError::OutOfMemory {
        operation: "set-raster-pipeline",
    });

    let error = state
        .reconcile(&mut backend, &mut counters)
        .expect_err("the backend refuses the install");

    assert_eq!(error.domain(), StateDomain::Pipeline);
    assert_eq!(error.operation(), "set-raster-pipeline");
    assert!(
        !state.pipeline_installed(),
        "a failed install must not be mirrored as an installed pipeline"
    );
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(installs(&backend), 0);

    let retry = state
        .reconcile(&mut backend, &mut counters)
        .expect("the retry emits");
    assert_eq!(
        retry,
        PipelineEffects {
            pipeline_installed: true,
        }
    );
    assert_eq!(installs(&backend), 1);
}

#[test]
fn a_refused_link_reports_the_domain_and_operation() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    backend.fail_next(GlError::OutOfMemory {
        operation: "create-program",
    });

    let error = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect_err("the backend refuses the link");

    assert_eq!(error.domain(), StateDomain::Pipeline);
    assert_eq!(error.operation(), "create-program");
    assert_eq!(counters.lifecycle.driver_errors, 1);
    assert_eq!(
        state.program_records(),
        0,
        "a refused link leaves no record behind"
    );
    assert_eq!(links(&backend), 0);
}

#[test]
fn a_context_loss_purges_program_records_without_destroying_them() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    let destroyed = destroys(&backend);
    backend.context_lost().expect("the mock records the loss");

    state.invalidate(&mut backend, &StateEvent::ContextLost, &mut counters);

    assert_eq!(state.program_records(), 0);
    assert_eq!(
        destroys(&backend),
        destroyed,
        "an identity from a lost epoch is not callable, so nothing may be destroyed through it"
    );
    assert_eq!(counters.caches.purged, 1);
    assert_eq!(counters.lifecycle.domain_invalidations, 1);
    // The mirror must stop claiming a pipeline the context took with it,
    // while keeping what the caller asked for: restoration installs it.
    assert!(!state.pipeline_installed());
    assert!(state.desired_pipeline().is_some());
}

#[test]
fn a_raw_scope_that_touched_this_domain_drops_the_programs_it_can_still_delete() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    let destroyed = destroys(&backend);

    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Pipeline,
        ))),
        &mut counters,
    );

    assert_eq!(state.program_records(), 0);
    assert_eq!(
        destroys(&backend),
        destroyed + 1,
        "the context is alive, so this layer deletes what it linked rather than leaking it"
    );
    assert_eq!(
        counters.caches.purged, 0,
        "a scope is not an epoch change: these identities are still callable"
    );
    assert_eq!(counters.lifecycle.domain_invalidations, 1);
    assert!(!state.pipeline_installed());
}

#[test]
fn a_raw_scope_that_declared_another_domain_leaves_this_one_alone() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");

    state.invalidate(
        &mut backend,
        &StateEvent::ScopedRawAccess(ScopedRawAccess::declaring(DirtyDomains::of(
            StateDomain::Textures,
        ))),
        &mut counters,
    );

    assert_eq!(state.program_records(), 1);
    assert_eq!(counters.lifecycle.domain_invalidations, 0);
}

#[test]
fn shutdown_destroys_the_programs_this_domain_linked() {
    let mut backend = backend_with_a_pass();
    let vao = vertex_array(&mut backend);
    let mut state = pipeline_state(ExecutionMode::Optimized);
    let mut counters = StateCounters::default();
    let (program, _, _) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the program links");
    state.set_pipeline(&pipeline(program, vao, GlCullMode::None));
    apply(&mut state, &mut backend, &mut counters);
    assert_eq!(destroys(&backend), 0);

    state.shutdown(&mut backend, &mut counters);

    assert_eq!(destroys(&backend), 1);
    assert_eq!(state.program_records(), 0);
    assert!(!state.pipeline_installed());
    assert!(state.desired_pipeline().is_none());
    assert_eq!(
        counters.lifecycle.driver_errors, 0,
        "the mock accepts every deletion"
    );
}

#[test]
fn an_oracle_never_reuses_a_program_and_hands_ownership_back() {
    let mut backend = backend_with_a_pass();
    let mut state = pipeline_state(ExecutionMode::Oracle);
    let mut counters = StateCounters::default();

    let (_, _, owned) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the oracle still links a usable program");
    assert!(
        owned,
        "the oracle retains nothing, so the caller destroys it"
    );
    let (_, _, owned) = state
        .program_for(&mut backend, &descriptor("f"), &mut counters)
        .expect("the oracle links again");
    assert!(owned);

    assert_eq!(links(&backend), 2);
    assert_eq!(state.program_records(), 0);
    assert_eq!(counters.caches.hits, 0);
    assert_eq!(counters.caches.created, 0);
    assert_eq!(
        counters.steady_state_allocations, 0,
        "the oracle retains nothing, so it pays for no lookup key"
    );
}
