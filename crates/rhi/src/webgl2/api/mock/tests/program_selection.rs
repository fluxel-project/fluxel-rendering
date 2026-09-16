//! Contract tests for the driver's single current program.
//!
//! GL has exactly one current program, and three verbs can change it: a link,
//! which reflects through a bind scope and leaves none selected; a raster
//! pipeline install; and a compute program install.  A dispatch or a draw
//! therefore cannot trust that the program it recorded is still the one the
//! driver holds, so the recorder models one *slot* rather than one belief per
//! verb, and a verb that uses a program restores it.
//!
//! These tests are the reason `set_compute_program` is not allowed to record an
//! intention: a recorder that only recorded it would accept a dispatch the real
//! provider cannot perform, which is how the native omission stayed invisible to
//! every gate (the plan records it as P1-16).
//!
//! Honest limit of that evidence: the recorder is not a driver, so what these
//! tests fix is the contract the executable providers are held to -- which
//! program each verb selects, and that a use restores the one it needs.  The
//! native provider cannot be constructed without a live GL context, so its half
//! is held by reading `NativeGlProvider::ensure_program`'s call sites rather than
//! by a failing case here; the same limit `tests/indirect.rs` states for its own
//! domain.

use super::*;

#[test]
fn installing_a_compute_program_selects_it() {
    let mut recorder = desktop_recorder();
    let (program, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    let from = api.calls().len();

    api.set_compute_program(program).expect("install");

    // The verb's name and its whole contract say it installs, so the trace has to
    // show the driver call and not only the record of it.  Without the selection a
    // dispatch runs whatever program is current, which on a real profile is the
    // raster program a preceding pass installed or, after any link, no program at
    // all -- and nothing reports it.
    assert_trace(
        &api.calls()[from..],
        &[
            MockCall::SelectProgram(program),
            MockCall::SetComputeProgram(program),
        ],
    );
}

#[test]
fn a_dispatch_after_a_raster_install_selects_the_compute_program_again() {
    let mut recorder = desktop_recorder();
    let (compute, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let (raster, _) = recorder
        .create_program(&raster_program())
        .expect("raster program");
    let vao = recorder
        .create_vertex_array(&empty_vertex_layout())
        .expect("vertex array");
    begin_pass(&mut recorder);

    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    api.set_compute_program(compute).expect("install");

    // The raster install takes the one slot, so the compute program the caller
    // installed is no longer what the driver holds.  This is the interaction the
    // omission hid: the slot moved and nothing said so.
    let mut recorder = api.into_inner();
    recorder
        .set_raster_pipeline(&pipeline(raster, vao))
        .expect("pipeline");
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture still proves both optional rows");

    let groups = GlDispatchGroups([1, 1, 1]);
    let from = api.calls().len();
    api.dispatch(groups).expect("dispatch");

    // The dispatch restores the program it was told to run rather than running
    // the one that happens to be selected.
    assert_trace(
        &api.calls()[from..],
        &[MockCall::SelectProgram(compute), MockCall::Dispatch(groups)],
    );
}

#[test]
fn a_draw_after_a_compute_install_selects_the_raster_program_again() {
    let mut recorder = desktop_recorder();
    let (compute, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let (raster, _) = recorder
        .create_program(&raster_program())
        .expect("raster program");
    let vao = recorder
        .create_vertex_array(&empty_vertex_layout())
        .expect("vertex array");
    begin_pass(&mut recorder);
    recorder
        .set_raster_pipeline(&pipeline(raster, vao))
        .expect("pipeline");

    // The same fact from the other side, which is why it is a separate arm rather
    // than a third case of the same test: the compute install is what takes the
    // slot here, and a draw that did not restore its program would draw with the
    // compute program still current.
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    api.set_compute_program(compute).expect("install");
    let mut recorder = api.into_inner();

    let draw = GlDrawCommand::NonIndexed(GlNonIndexedDraw {
        first_vertex: 0,
        vertex_count: 3,
        instance_count: 1,
    });
    let from = recorder.calls().len();
    recorder.draw_raster(draw).expect("draw");

    assert_trace(
        &recorder.calls()[from..],
        &[MockCall::SelectProgram(raster), MockCall::DrawRaster(draw)],
    );
}

#[test]
fn a_second_dispatch_selects_nothing() {
    let mut recorder = desktop_recorder();
    let (program, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    api.set_compute_program(program).expect("install");
    let groups = GlDispatchGroups([1, 1, 1]);
    api.dispatch(groups).expect("first dispatch");

    let from = api.calls().len();
    api.dispatch(groups).expect("second dispatch");

    // The restoral is a comparison and not a re-install: a dispatch whose program
    // is still current selects nothing, which is what keeps the extra call off the
    // steady-state path.
    assert_trace(&api.calls()[from..], &[MockCall::Dispatch(groups)]);
}
