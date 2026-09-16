//! Evidence for the optional command domains the recorder implements.
//!
//! Each domain gets an accepted operation recorded under its own variant and a
//! rejection recorded as an error with no accepted call beside it, because those
//! two halves are what the differential contract needs: the accepted half proves
//! the trace names the shape that ran, the rejected half proves the guard runs
//! before the side effect.  Two optional domains have no executable provider in
//! this contract (no family binds a multi-draw-indirect or advanced-offset entry
//! point yet), so the rules those halves enforce are the ones the descriptors and
//! capability rows define rather than the ones a provider call site would.

use super::*;

#[test]
fn an_advanced_draw_records_the_offset_the_context_proved() {
    let mut api = recorder();
    begin_pass(&mut api);
    let plain = GlAdvancedDrawCommand {
        draw: non_indexed(),
        base_vertex: 0,
        first_instance: 0,
    };
    api.clear_calls();
    // Zero is not an offset, so the draw every profile already has stays legal
    // on a context that proved nothing optional: the gate is about the offsets,
    // not about the domain, and a recorder that refused zero would reject a draw
    // both providers accept.
    assert_eq!(api.draw_advanced_raster(plain), Ok(()));
    assert_trace(api.calls(), &[MockCall::DrawAdvancedRaster(plain)]);

    // A nonzero offset is accepted only where this context proved it, so the
    // seam that injects the proof has to change the answer; without this half
    // the test would pass against a mock that recorded the domain unconditionally.
    api.set_advanced_raster_capabilities(GlAdvancedRasterCapabilities {
        base_vertex: true,
        first_instance: false,
    });
    let offset = GlAdvancedDrawCommand {
        draw: non_indexed(),
        base_vertex: 4,
        first_instance: 0,
    };
    api.clear_calls();
    assert_eq!(api.draw_advanced_raster(offset), Ok(()));
    assert_trace(api.calls(), &[MockCall::DrawAdvancedRaster(offset)]);
}

#[test]
fn an_unproved_advanced_offset_is_rejected_before_the_draw() {
    let mut api = recorder();
    begin_pass(&mut api);
    api.clear_calls();
    let draw = GlAdvancedDrawCommand {
        draw: non_indexed(),
        base_vertex: 0,
        first_instance: 2,
    };
    // The default proves nothing, so an unconfigured recorder must refuse the
    // offset rather than accept one no provider ever proved.  The failure is an
    // unsupported domain rather than a malformed command: the draw is well
    // formed, this context simply never earned the command, and a caller that
    // cannot tell those apart would keep repairing a request that is already
    // correct.
    let error = api
        .draw_advanced_raster(draw)
        .expect_err("first_instance was never proved");
    assert!(matches!(
        &error,
        GlError::Unsupported { operation, .. } if *operation == "draw-advanced-raster"
    ));
    assert_rejected(api.calls());
}

#[test]
fn every_indirect_row_a_context_never_proved_rejects_before_any_buffer_is_read() {
    let mut api = recorder();
    let buffer = buffer_with(&mut api, 32, GlBufferUsage::UNIFORM);
    // Deliberately no pass and no indirect role: the row gate precedes every
    // other condition, so each rejection names the capability rather than the
    // missing pass or the buffer's role, and a trace with no accepted command
    // beside the errors proves nothing was bound before the guard ran.
    api.clear_calls();
    let error = api
        .draw_indirect(command_range(buffer, 16, 1))
        .expect_err("the indirect-draw row was never proved");
    assert!(matches!(
        &error,
        GlError::Unsupported { operation, .. } if *operation == "draw-indirect"
    ));
    let error = api
        .multi_draw_indirect(command_range(buffer, 16, 1))
        .expect_err("the multi-draw-indirect row was never proved");
    assert!(matches!(
        &error,
        GlError::Unsupported { operation, .. } if *operation == "multi-draw-indirect"
    ));
    let error = api
        .multi_draw_indirect_count(command_range(buffer, 16, 1), count_range(buffer, 4, 1))
        .expect_err("the multi-draw-indirect row was never proved");
    assert!(matches!(
        &error,
        GlError::Unsupported { operation, .. } if *operation == "multi-draw-indirect-count"
    ));
    assert_eq!(
        api.calls().len(),
        3,
        "one rejection per verb and nothing else"
    );
    assert!(
        api.calls()
            .iter()
            .all(|call| matches!(call, MockCall::Error(_))),
        "no indirect command is recorded beside a rejection: {:?}",
        api.calls()
    );
}

#[test]
fn an_indirect_command_is_recorded_as_one_read_command() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 16, GlBufferUsage::INDIRECT);
    begin_pass(&mut api);
    api.clear_calls();
    let command = command_range(buffer, 16, 1);
    assert_eq!(api.draw_indirect(command), Ok(()));
    assert_trace(api.calls(), &[MockCall::DrawIndirect(command)]);
}

#[test]
fn an_indirect_command_on_a_buffer_without_the_role_is_rejected() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 16, GlBufferUsage::UNIFORM);
    begin_pass(&mut api);
    api.clear_calls();
    // The role is what lets a provider read the bytes as records instead of as
    // caller data, so a live buffer created for another purpose is not a command
    // buffer.  The range's own arithmetic is legal here, which is what makes this
    // rejection name the role rather than the layout.
    let error = api
        .draw_indirect(command_range(buffer, 16, 1))
        .expect_err("a uniform buffer is not a command buffer");
    assert!(matches!(
        &error,
        GlError::Validation { message, .. } if message.contains("indirect usage")
    ));
    assert_rejected(api.calls());
}

#[test]
fn a_batch_read_is_recorded_as_one_indirect_batch() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 16, GlBufferUsage::INDIRECT);
    begin_pass(&mut api);
    api.clear_calls();
    let commands = command_range(buffer, 16, 1);
    assert_eq!(api.multi_draw_indirect(commands), Ok(()));
    assert_trace(api.calls(), &[MockCall::MultiDrawIndirect(commands)]);
}

#[test]
fn a_batch_read_beyond_the_queried_count_limit_is_rejected() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 32, GlBufferUsage::INDIRECT);
    begin_pass(&mut api);
    api.clear_calls();
    // Two records fit the range and are well formed, so the only reason left is
    // the queried limit of one: the recorder must refuse a batch asking for more
    // draws than this context ever claimed it could read, or a test could prove
    // a submission shape the hardware would never have accepted.
    let error = api
        .multi_draw_indirect(command_range(buffer, 32, 2))
        .expect_err("two records exceed the queried count limit of one");
    assert!(matches!(
        &error,
        GlError::Validation { message, .. } if message.contains("multi-draw-indirect limit")
    ));
    assert_rejected(api.calls());
}

#[test]
fn a_counted_batch_records_both_ranges_it_read() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 16, GlBufferUsage::INDIRECT);
    begin_pass(&mut api);
    api.clear_calls();
    let commands = command_range(buffer, 16, 1);
    let count = count_range(buffer, 4, 1);
    assert_eq!(api.multi_draw_indirect_count(commands, count), Ok(()));
    // Both ranges are part of the recorded command: a trace keeping only the
    // record range would hide which count word decided how many draws ran, and
    // that word is the whole difference between this verb and the batch one.
    assert_trace(
        api.calls(),
        &[MockCall::MultiDrawIndirectCount { commands, count }],
    );
}

#[test]
fn a_counted_batch_whose_count_word_is_out_of_range_is_rejected() {
    let mut api = desktop_recorder();
    let buffer = buffer_with(&mut api, 16, GlBufferUsage::INDIRECT);
    begin_pass(&mut api);
    api.clear_calls();
    // One whole u32 word must fit at the offset the count names; asking for it
    // past the end of the range would make the provider read whatever follows.
    let count = GlIndirectCountRange {
        range: GlBufferRange {
            buffer,
            offset: 0,
            size: 4,
        },
        count_offset: 4,
        max_draw_count: 1,
    };
    let error = api
        .multi_draw_indirect_count(command_range(buffer, 16, 1), count)
        .expect_err("the count word does not fit its range");
    // The count range's validator owns this error and names its own operation,
    // so the recorder reports the module's spelling rather than this call's.
    // That is exactly what a provider propagates, and rewriting it here would
    // make the recorder's trace disagree with the provider it mirrors.
    assert!(matches!(
        &error,
        GlError::Validation { operation, .. } if *operation == "multi_draw_indirect_count"
    ));
    assert_rejected(api.calls());
}

#[test]
fn dispatch_indirect_records_the_record_it_read() {
    let mut recorder = desktop_recorder();
    let buffer = buffer_with(&mut recorder, 12, GlBufferUsage::INDIRECT);
    let (program, _) = recorder
        .create_program(&compute_program())
        .expect("compute program");
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    api.set_compute_program(program).expect("install");
    let from = api.calls().len();
    let command = dispatch_command(buffer, 12);
    assert_eq!(api.dispatch_indirect(command), Ok(()));
    assert_trace(&api.calls()[from..], &[MockCall::DispatchIndirect(command)]);
}

#[test]
fn an_indirect_dispatch_without_an_installed_program_is_rejected() {
    let mut recorder = desktop_recorder();
    let buffer = buffer_with(&mut recorder, 12, GlBufferUsage::INDIRECT);
    let mut api = recorder
        .try_with_compute_storage()
        .expect("the fixture proved both optional rows");
    let from = api.calls().len();
    // The installed program is what makes the record's work-group triple
    // meaningful, so dispatching without one must fail before the buffer is read
    // rather than after the driver has already decoded the record.
    let error = api
        .dispatch_indirect(dispatch_command(buffer, 12))
        .expect_err("no compute program is installed");
    assert!(matches!(
        &error,
        GlError::Validation { operation, .. } if *operation == "dispatch-indirect"
    ));
    assert_rejected(&api.calls()[from..]);
}

#[test]
fn a_batch_is_recorded_as_one_command_only_where_that_command_was_proved() {
    let mut api = webgl2_recorder();
    begin_pass(&mut api);
    let batch =
        GlMultiDraw::new(vec![non_indexed_at(0, 3), non_indexed_at(3, 6)]).expect("homogeneous");
    api.clear_calls();
    assert_eq!(api.multi_draw(&batch), Ok(()));
    // This context proved the combined command, so issuing the batch as one
    // command is what a provider does and what the trace must say: a recorder
    // that only ever decomposed would make the two routes indistinguishable and
    // a state-machine test could no longer see which one ran.
    assert_trace(api.calls(), &[MockCall::MultiDraw(batch.clone())]);
}

#[test]
fn a_batch_without_the_combined_command_is_recorded_as_its_single_draws() {
    let mut api = recorder();
    begin_pass(&mut api);
    let batch =
        GlMultiDraw::new(vec![non_indexed_at(0, 3), non_indexed_at(3, 6)]).expect("homogeneous");
    api.clear_calls();
    assert_eq!(api.multi_draw(&batch), Ok(()));
    // This context proved no combined command, so the batch has to be reported
    // as the single draws it really is, in submission order.  Both providers
    // take the same fallback, so the equal-but-distinct vertex offsets here are
    // what makes a reordering or a merge visible instead of silent.
    assert_trace(
        api.calls(),
        &[
            MockCall::DrawRaster(non_indexed_at(0, 3)),
            MockCall::DrawRaster(non_indexed_at(3, 6)),
        ],
    );
}
