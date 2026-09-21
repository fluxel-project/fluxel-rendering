//! Timer-query evidence: the acquisition oracle, the recorded counter width,
//! and the disjoint-interval classification.
//!
//! Three of these four tests run against the live context and assert that the
//! capability row, the retained commands, the extension ledger, and the
//! recorded counter width are four views of one acquisition. The command oracle
//! is additionally driven against a JS stand-in, because the half this browser
//! cannot supply is the browser's own answer; the reflection and width-reading
//! code under test is the real one.

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

use super::*;

#[wasm_bindgen_test]
fn timer_capability_row_matches_the_recorded_counter_width() {
    let mut provider = provider();
    let width = provider.snapshot().limits().query_counter_bits;
    let supports = provider
        .snapshot()
        .capabilities()
        .supports(GlCapability::TimerQuery);
    let commands = provider.timer.is_some();
    let acquired = provider
        .snapshot()
        .extensions()
        .provenance(GlKnownExtension::ExtDisjointTimerQueryWebgl2)
        == Some(ExtensionProvenance::Acquired);
    // The row, the retained commands, the ledger, and the recorded width are
    // four views of one acquisition and must agree; a row that reported a
    // width while no command set existed would make the capability row lie.
    assert_eq!(
        supports,
        width != 0,
        "timer row and recorded counter width disagree"
    );
    assert_eq!(
        acquired, commands,
        "timer ledger and retained commands disagree"
    );
    assert_eq!(
        supports,
        commands && width != 0,
        "timer row enabled without an acquired, callable command set"
    );
    // Which half of this contract ran is environment-dependent, so it is logged
    // rather than inferred: the fail-closed half is asserted below, and the
    // end-to-end half only runs in `timer_measurement_runs_end_to_end...`.
    web_sys::console::log_1(
        &format!("browser timer route: capability={supports} counter_bits={width}").into(),
    );

    // A context with no acquired timer object must record no width at all.
    assert_eq!(exec_timer::recorded_counter_width(&provider.raw, None), 0);

    let query = provider.create_query().expect("create query");
    let error = provider.begin_elapsed_query(query).unwrap_err();
    // An enabled row must not reject the interval it claims to support; a
    // failure here means the entry points cannot really be called.
    assert!(
        !supports,
        "timer capability enabled but begin failed: {error:?}"
    );
    assert!(
        is_unsupported(&error, "begin-elapsed-query"),
        "a disabled timer row must fail closed: {error:?}"
    );
    assert!(is_unsupported(
        &provider.end_elapsed_query().unwrap_err(),
        "end-elapsed-query"
    ));
    assert!(is_unsupported(
        &provider.query_timestamp(query).unwrap_err(),
        "query-timestamp"
    ));
    // Fail-closed means no side effect, so the query still records nothing and
    // a result read is refused as "never measured" rather than answered.
    assert!(is_validation(
        &provider.query_result(query).unwrap_err(),
        "query-result"
    ));
    provider.destroy_query(query).expect("destroy query");
}

/// The five commands one acquired timer object must expose, by their JS names.
const TIMER_COMMANDS: [&str; 5] = [
    "beginQueryEXT",
    "endQueryEXT",
    "queryCounterEXT",
    "getQueryEXT",
    "getQueryParameterEXT",
];

/// Builds a JS stand-in for the acquired timer object.
///
/// This is not a simulation of a driver: the browser's own answers are what
/// these tests cannot supply (this Chrome does not expose the timer route at
/// all), so the stand-in supplies only that half and the real reflection and
/// width-reading code runs unchanged against it.
fn synthetic_timer(commands: &[&str], counter_bits_body: &str) -> JsValue {
    let object = js_sys::Object::new();
    for name in commands {
        let function = if *name == "getQueryEXT" {
            js_sys::Function::new_with_args("target, pname", counter_bits_body)
        } else {
            js_sys::Function::new_with_args("a, b", "")
        };
        js_sys::Reflect::set(&object, &(*name).into(), &function).expect("set a timer command");
    }
    object.into()
}

#[wasm_bindgen_test]
fn timer_command_oracle_refuses_a_partial_object_and_a_widthless_route() {
    let provider = provider();
    // Both targets answer, with different widths: the recorded width is the
    // smaller, because the one row covers intervals and timestamps alike.
    let widths = "if (pname === 0x8864) { return target === 0x88BF ? 32 : 40; } return null;";
    let object = synthetic_timer(&TIMER_COMMANDS, widths);
    let timer =
        exec_timer::BrowserTimerQuery::acquire(&object).expect("a complete command set acquires");
    assert_eq!(timer.elapsed_counter_bits(), Some(32));
    assert_eq!(timer.timestamp_counter_bits(), Some(40));
    assert_eq!(
        exec_timer::recorded_counter_width(&provider.raw, Some(&timer)),
        32
    );

    // A reported object missing any one command must not acquire: a row enabled
    // by an object whose command path would throw at the next call is exactly
    // the lie the oracle exists to prevent.
    for missing in TIMER_COMMANDS {
        let partial: Vec<&str> = TIMER_COMMANDS
            .iter()
            .copied()
            .filter(|name| *name != missing)
            .collect();
        let object = synthetic_timer(&partial, widths);
        assert!(
            exec_timer::BrowserTimerQuery::acquire(&object).is_none(),
            "a partial object acquired without {missing}"
        );
        assert_eq!(exec_timer::recorded_counter_width(&provider.raw, None), 0);
    }

    // A route that answers no width for one target, a fractional width, or no
    // width at all records zero, which is what disables the capability row.
    for body in [
        "if (pname === 0x8864) { return target === 0x88BF ? 32 : 0; } return null;",
        "if (pname === 0x8864) { return target === 0x88BF ? 32.5 : 40; } return null;",
        "if (pname === 0x8864) { return target === 0x88BF ? 8 : null; } return null;",
        "return null;",
    ] {
        let object = synthetic_timer(&TIMER_COMMANDS, body);
        let timer = exec_timer::BrowserTimerQuery::acquire(&object).expect("complete set");
        assert_eq!(
            exec_timer::recorded_counter_width(&provider.raw, Some(&timer)),
            0,
            "an unusable width answered as a usable one: {body}"
        );
    }

    // A wider elapsed counter than the timestamp counter records the smaller.
    let object = synthetic_timer(
        &TIMER_COMMANDS,
        "if (pname === 0x8864) { return target === 0x88BF ? 64 : 36; } return null;",
    );
    let timer = exec_timer::BrowserTimerQuery::acquire(&object).expect("complete set");
    assert_eq!(
        exec_timer::recorded_counter_width(&provider.raw, Some(&timer)),
        36
    );
}

#[wasm_bindgen_test]
fn timer_measurement_classification_covers_the_disjoint_matrix() {
    // Availability is the outer decision: a measurement that is not ready has
    // no disruption answer to consult, and an unreadable answer is unknown.
    assert_eq!(
        exec_timer::classify_measurement(None, None, None),
        GlQueryResult::Unknown
    );
    assert_eq!(
        exec_timer::classify_measurement(Some(false), None, None),
        GlQueryResult::Pending
    );
    // A ready interval whose disruption flag could not be read is unknown, not
    // available: handing out the number would claim the interval was intact.
    assert_eq!(
        exec_timer::classify_measurement(Some(true), None, Some(7)),
        GlQueryResult::Unknown
    );
    assert_eq!(
        exec_timer::classify_measurement(Some(true), Some(true), Some(7)),
        GlQueryResult::Disjoint
    );
    assert_eq!(
        exec_timer::classify_measurement(Some(true), Some(false), Some(7)),
        GlQueryResult::Available(7)
    );
    assert_eq!(
        exec_timer::classify_measurement(Some(true), Some(false), None),
        GlQueryResult::Unknown
    );

    // The target word decides which domain owns a recorded measurement, and it
    // must not claim the occlusion counter as a timer.
    assert!(exec_timer::is_timer_target(0x88BF));
    assert!(exec_timer::is_timer_target(0x8E28));
    assert!(!exec_timer::is_timer_target(0));
    assert!(!exec_timer::is_timer_target(format_map::SAMPLES_PASSED));
}

#[wasm_bindgen_test]
fn timer_measurement_runs_end_to_end_when_the_context_proved_it() {
    let mut provider = provider();
    if !provider
        .snapshot()
        .capabilities()
        .supports(GlCapability::TimerQuery)
    {
        // Nothing to run here: the fail-closed half is asserted above, and the
        // honest evidence for an unproved context is that it stays closed.
        return;
    }
    let query = provider.create_query().expect("create query");
    provider
        .begin_elapsed_query(query)
        .expect("begin elapsed interval");
    provider.end_elapsed_query().expect("end elapsed interval");
    let result = provider
        .query_result(query)
        .expect("a proved timer query must answer without a driver error");
    // A disrupted interval cannot be provoked on demand, so the assertion is
    // that the answer is one of the four classifications and never an error.
    // `Disjoint` is asserted as reachable only by its pure classification.
    assert!(matches!(
        result,
        GlQueryResult::Pending
            | GlQueryResult::Available(_)
            | GlQueryResult::Disjoint
            | GlQueryResult::Unknown
    ));
    // A timestamp is a single write and must not occupy the active slot.
    provider.destroy_query(query).expect("destroy query");
    let stamp = provider.create_query().expect("create timestamp query");
    provider.query_timestamp(stamp).expect("write a timestamp");
    provider
        .query_timestamp(stamp)
        .expect("a timestamp is not an active interval");
    provider
        .destroy_query(stamp)
        .expect("destroy timestamp query");
}
