//! Browser vendor/renderer identity evidence, including the unmasked route.
//!
//! A WebGL2 context answers `VENDOR` and `RENDERER` with strings the browser
//! itself chooses, so a hardware-evidence record built from them alone cannot
//! name the driver that actually ran the commands. Some contexts expose a debug
//! route that answers the real pair instead; it is optional, and it is the only
//! source of that pair. The values are therefore recorded as provider-specific
//! flag markers — the same evidence home the native provider uses for its own
//! driver-specific extras — rather than overwriting the context's verbatim
//! answers, so both facts stay visible and distinguishable.
//!
//! Nothing here ever synthesizes an identity: an answer that is absent,
//! non-string, or blank is marked unavailable instead of guessed at, because a
//! fabricated driver string would silently invalidate every record quoting it.
//!
//! This module does not own the context-attribute flags (that is `discovery.rs`)
//! and never decides whether a capability enables.

use web_sys::WebGl2RenderingContext as Gl;

use super::super::api::GlContextFlags;

/// `UNMASKED_VENDOR_WEBGL` and `UNMASKED_RENDERER_WEBGL` of the debug
/// renderer-info route. The bindings name neither parameter, so both are
/// spelled from the registry like the other unnamed constants in this tree.
const UNMASKED_VENDOR: u32 = 0x9245;
const UNMASKED_RENDERER: u32 = 0x9246;

/// The registry spelling of that route.
///
/// It is a literal because the typed extension registry (`api/extensions.rs`)
/// has no variant for it. That absence is a recorded decision rather than an
/// omission: the identity stays a free-form platform fact until a consumer needs
/// it as a value, and the registry section names that trigger. A typed variant
/// is the change that would let this constant disappear.
const IDENTITY_ROUTE: &str = "WEBGL_debug_renderer_info";

/// Reads the unmasked vendor/renderer pair, if the context really exposes it.
///
/// Returns `None` unless the route object exists and both strings are present
/// and non-blank. A partially answered route is not half an identity: the
/// caller records one unavailable marker instead, so a reader can never mistake
/// a masked browser string for a driver name.
pub(super) fn unmasked_identity(raw: &Gl) -> Option<(String, String)> {
    let identity = read_identity(raw);
    // Asking for an optional route may be answered with an error instead of an
    // object. That error belongs to this question, and leaving it pending would
    // make the next unrelated provider call report it as its own failure.
    let _ = raw.get_error();
    identity
}

fn read_identity(raw: &Gl) -> Option<(String, String)> {
    raw.get_extension(IDENTITY_ROUTE).ok().flatten()?;
    let vendor = raw.get_parameter(UNMASKED_VENDOR).ok()?.as_string();
    let renderer = raw.get_parameter(UNMASKED_RENDERER).ok()?.as_string();
    identity_from_strings(vendor, renderer)
}

/// Accepts the pair only when both halves are really there.
///
/// Separate from the reads so the rule is decided in one pure place: no read
/// error, no non-string parameter, and no blank value can become an identity.
pub(super) fn identity_from_strings(
    vendor: Option<String>,
    renderer: Option<String>,
) -> Option<(String, String)> {
    let vendor = vendor?;
    let renderer = renderer?;
    if vendor.trim().is_empty() || renderer.trim().is_empty() {
        return None;
    }
    Some((vendor, renderer))
}

/// Records the identity evidence as context flag markers.
///
/// Exactly one shape is always written, so a later reader can tell a context
/// whose route answered from one recorded before the route was asked at all.
pub(super) fn record_markers(flags: &mut GlContextFlags, identity: Option<(String, String)>) {
    match identity {
        Some((vendor, renderer)) => {
            flags
                .other
                .insert(format!("webgl.unmasked-vendor={vendor}"));
            flags
                .other
                .insert(format!("webgl.unmasked-renderer={renderer}"));
        }
        None => {
            flags
                .other
                .insert("webgl.unmasked-identity=unavailable".into());
        }
    }
}
