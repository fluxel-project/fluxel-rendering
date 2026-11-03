use std::{
    cell::{Cell, RefCell},
    vec::Vec,
};

use super::{
    ResizeWaitOutcome, await_resize_confirmation, requested_resize_event, resize_is_confirmed,
};
use fluxel_host::WindowEvent;

#[test]
fn resize_confirmation_requires_a_matching_host_event_and_client_extent() {
    let target = [256, 144];
    let events = [WindowEvent::Resized {
        width: target[0],
        height: target[1],
    }];
    let event = requested_resize_event(&events, target);

    assert_eq!(event, Some("resized"));
    assert!(resize_is_confirmed(event, target, target));
    assert!(!resize_is_confirmed(event, [320, 180], target));
}

#[test]
fn client_extent_without_a_matching_event_does_not_confirm_resize() {
    let target = [256, 144];
    let events = [WindowEvent::Resized {
        width: 320,
        height: 180,
    }];

    assert_eq!(requested_resize_event(&events, target), None);
    assert!(!resize_is_confirmed(None, target, target));
}

#[test]
fn restored_event_can_confirm_the_requested_extent() {
    let target = [320, 180];
    let events = [WindowEvent::Restored {
        width: target[0],
        height: target[1],
    }];
    let event = requested_resize_event(&events, target);

    assert_eq!(event, Some("restored"));
    assert!(resize_is_confirmed(event, target, target));
}

#[test]
fn resize_waits_through_empty_polls_until_host_and_client_agree() {
    let target = [256, 144];
    let samples = RefCell::new(
        vec![
            (Vec::new(), [320, 180]),
            (Vec::new(), target),
            (
                vec![WindowEvent::Resized {
                    width: target[0],
                    height: target[1],
                }],
                target,
            ),
        ]
        .into_iter(),
    );
    let polls = Cell::new(0);
    let pauses = Cell::new(0);

    let outcome = await_resize_confirmation(
        target,
        || {
            polls.set(polls.get() + 1);
            Ok(samples
                .borrow_mut()
                .next()
                .expect("test provides each expected poll"))
        },
        || polls.get() >= 3,
        || pauses.set(pauses.get() + 1),
    )
    .expect("polling fixture must not fail");

    match outcome {
        ResizeWaitOutcome::Confirmed { fact, state } => {
            assert_eq!(fact.extent, target);
            assert_eq!(fact.event, "resized");
            assert_eq!(state.observed_events.len(), 1);
        }
        ResizeWaitOutcome::TimedOut(_) => panic!("delayed matching event must confirm resize"),
    }
    assert_eq!(polls.get(), 3);
    assert_eq!(pauses.get(), 2);
}

#[test]
fn resize_wait_timeout_retains_last_actual_state() {
    let target = [256, 144];
    let samples = RefCell::new(
        vec![
            (Vec::new(), [320, 180]),
            (
                vec![WindowEvent::Resized {
                    width: 320,
                    height: 180,
                }],
                [320, 180],
            ),
        ]
        .into_iter(),
    );
    let polls = Cell::new(0);

    let outcome = await_resize_confirmation(
        target,
        || {
            polls.set(polls.get() + 1);
            Ok(samples
                .borrow_mut()
                .next()
                .expect("test provides each expected poll"))
        },
        || polls.get() >= 2,
        || {},
    )
    .expect("polling fixture must not fail");

    match outcome {
        ResizeWaitOutcome::Confirmed { .. } => panic!("wrong host event must not confirm resize"),
        ResizeWaitOutcome::TimedOut(state) => {
            assert_eq!(state.selected_event, None);
            assert_eq!(state.last_extent, [320, 180]);
            assert_eq!(state.observed_events.len(), 1);
        }
    }
    assert_eq!(polls.get(), 2);
}
