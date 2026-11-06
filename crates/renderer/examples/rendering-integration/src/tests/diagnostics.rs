use super::*;

fn entry(source: u32, kind: u32, severity: u32) -> String {
    format!(
        "DebugMessageLogEntry {{ source: {source}, msg_type: {kind}, id: {}, severity: {severity}, message: {:?} }}",
        0x4658_454c_u32,
        WglObserver::ACTIVATION_MARKER
    )
}

#[test]
fn only_decoded_application_notifications_are_instrumentation() {
    for kind in [
        glow::DEBUG_TYPE_MARKER,
        glow::DEBUG_TYPE_PUSH_GROUP,
        glow::DEBUG_TYPE_POP_GROUP,
    ] {
        assert!(is_application_marker(&entry(
            glow::DEBUG_SOURCE_APPLICATION,
            kind,
            glow::DEBUG_SEVERITY_NOTIFICATION
        )));
    }
    assert!(!is_application_marker(&entry(
        glow::DEBUG_SOURCE_API,
        glow::DEBUG_TYPE_MARKER,
        glow::DEBUG_SEVERITY_NOTIFICATION
    )));
    assert!(!is_application_marker(&entry(
        glow::DEBUG_SOURCE_APPLICATION,
        glow::DEBUG_TYPE_ERROR,
        glow::DEBUG_SEVERITY_HIGH
    )));
    assert!(!is_application_marker(&entry(
        glow::DEBUG_SOURCE_API,
        glow::DEBUG_TYPE_PERFORMANCE,
        glow::DEBUG_SEVERITY_LOW
    )));
    assert!(!is_application_marker("unrecognized log representation"));
}

#[test]
fn activation_proof_requires_exact_metadata_and_message_not_substring() {
    let marker = entry(
        glow::DEBUG_SOURCE_APPLICATION,
        glow::DEBUG_TYPE_MARKER,
        glow::DEBUG_SEVERITY_NOTIFICATION,
    );
    assert!(is_activation_marker(&marker));
    assert!(!is_activation_marker(&(marker.clone() + " extra text")));
    assert!(!is_activation_marker(
        &marker.replace("id: 1180190028", "id: 0")
    ));
    assert!(!is_activation_marker(&entry(
        glow::DEBUG_SOURCE_API,
        glow::DEBUG_TYPE_MARKER,
        glow::DEBUG_SEVERITY_NOTIFICATION
    )));
}

#[test]
fn core_debug_output_version_check_is_numeric_not_a_fixed_prefix_list() {
    assert!(gl_version_at_least("4.3 Core Profile", 4, 3));
    assert!(gl_version_at_least("4.10 vendor text", 4, 3));
    assert!(gl_version_at_least("5.0", 4, 3));
    assert!(!gl_version_at_least("4.2", 4, 3));
    assert!(!gl_version_at_least("vendor 4.6", 4, 3));
    assert!(!gl_version_at_least("unparseable", 4, 3));
}
