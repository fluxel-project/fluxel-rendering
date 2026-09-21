//! GLES family hardware evidence, collected on a real EGL pbuffer context.
//!
//! This fixture is the offscreen half of the GL-family evidence pair, and it is
//! the one that can run on a device rather than on the machine that builds it.
//! `fluxel-rhi`'s doc-hidden conformance entry owns opening the context and
//! reading the driver, because that needs the GL family's private identities and
//! no out-of-workspace caller can be given them without widening a contract this
//! series froze.  What is left here is argument parsing and a report.
//!
//! It prints the same JSON object `examples/windows-gl4` prints, key for key, so
//! that the two evidences can be diffed and one checker can read both.  That is
//! the reason this file restates the report writer rather than sharing one: the
//! two fixtures are out-of-workspace crates by design, and a shared crate between
//! them would be an edge each of them would have to carry.  A reader comparing
//! the two should find them near-identical on purpose.
//!
//! # What it cannot be asked for, and why
//!
//! There is no observe-only run.  The desktop fixture can print a context reading
//! with no frame behind it, because WGL has a separate entry for that; here the
//! reading and the run come from one call, and `drive_gles_pbuffer_draws` refuses
//! a zero-draw workload rather than measuring only its own setup.  So `--draws`
//! defaults to one rather than to zero, and there is no `--frames`: an offscreen
//! context has no message queue to pump and no window that can be closed.
//!
//! # Why the extent is a parameter when the target is not
//!
//! The workload renders into a fixed four-by-four texture; the extent this
//! fixture takes is the *pbuffer's* size, which is a fact about the surface the
//! frame runs on.  They are separate on purpose -- a pbuffer has no window to
//! adopt an extent from, so the caller has to state it, and the report echoes it
//! back so that the surface and the frame it carried can be told apart.
//!
//! ```text
//! adb shell /data/local/tmp/fluxel-android-gles-harness \
//!     --extent 64x64 --draws 1 --readback /data/local/tmp/colour.rgba
//! ```
//!
//! To test a GLES compatibility floor rather than discovering the highest
//! profile the driver accepts, add `--gles-version 3.0`, `3.1`, or `3.2`. A
//! 3.1 case accepts an observed 3.1 or 3.2 context, but rejects 3.0. The JSON
//! records both `requested_minimum_version` and the observed `version`.
//!
//! # The picture
//!
//! With `--readback PATH` the run also reads its colour target back and writes
//! the raw bytes to `PATH`, so that the picture a real GLES context produced can
//! be *looked at* and not only counted.  The pixels are reported in the same JSON
//! by value as well, because the target is a fixed four-by-four and a reviewer
//! checking coverage and orientation should not have to decode a file to do it.
//!
//! The bytes are written in the order the family produced them and the report
//! says which order that is; nothing here flips them.  A fixture that reversed
//! rows would be inventing an interpretation, and the one thing a consumer of
//! this output must not have to guess is whether the image is upside down.

use std::path::PathBuf;
use std::process::ExitCode;

use fluxel_rhi::test_support::{
    ColourReadback, EglGlesVersion, NativeGlContextReport, NativeGlDrawReport,
    drive_gles_pbuffer_draws,
};

/// The default pbuffer extent, and the extent the context is opened for.
///
/// Small on purpose: the frame this drives renders into a four-by-four target
/// whatever the pbuffer is, so a larger surface buys nothing but allocation.  It
/// is not four-by-four itself, because a pbuffer that exactly matched the target
/// would make a surface that failed to be created at the requested size
/// indistinguishable from one that was.
const DEFAULT_EXTENT: [u32; 2] = [64, 64];

/// The identity the context is opened with.
///
/// Carried through the common device identity rather than being a GL name: the
/// adapter's resources are keyed by it, and a zero here is refused before
/// anything is opened.
const CONTEXT_IDENTITY: u64 = 1;

fn main() -> ExitCode {
    let mut extent = DEFAULT_EXTENT;
    let mut draws = 1_u32;
    let mut mode = String::from("optimized");
    let mut readback: Option<PathBuf> = None;
    // `None` means highest-supported discovery (3.2 -> 3.1 -> 3.0). A value
    // is a minimum profile assertion: a newer observed context is valid.
    let mut gles_version = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--extent" => match arguments.next().as_deref().and_then(parse_extent) {
                Some(parsed) => extent = parsed,
                None => {
                    eprintln!("--extent wants WIDTHxHEIGHT, for example 64x64");
                    return ExitCode::from(2);
                }
            },
            "--draws" => match arguments.next().as_deref().and_then(|raw| raw.parse().ok()) {
                Some(0) | None => {
                    eprintln!("--draws wants a positive count");
                    return ExitCode::from(2);
                }
                Some(parsed) => draws = parsed,
            },
            "--mode" => match arguments.next() {
                Some(parsed) => mode = parsed,
                None => {
                    eprintln!("--mode wants `optimized` or `oracle`");
                    return ExitCode::from(2);
                }
            },
            "--gles-version" => match arguments.next().as_deref().and_then(parse_gles_version) {
                Some(parsed) => gles_version = Some(parsed),
                None => {
                    eprintln!("--gles-version wants one of: 3.0, 3.1, 3.2");
                    return ExitCode::from(2);
                }
            },
            "--readback" => match arguments.next() {
                Some(path) => readback = Some(PathBuf::from(path)),
                None => {
                    eprintln!("--readback wants the path to write the colour target to");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }

    let wanted = readback.is_some();
    let report = match drive_gles_pbuffer_draws(
        extent,
        CONTEXT_IDENTITY,
        &mode,
        draws,
        wanted,
        gles_version,
    ) {
        Ok(report) => report,
        Err(error) => return report_failure(&error),
    };

    // Written before the report is printed, so that a document naming a file
    // cannot be produced by a run whose file was never written.  A failure here
    // is a failure of the run: the evidence this flag exists for is the pair.
    if let (Some(path), Some(colour)) = (readback.as_ref(), report.colour.as_ref()) {
        if let Err(error) = std::fs::write(path, &colour.bytes) {
            eprintln!(
                "the colour target did not reach {}: {error}",
                path.display()
            );
            return ExitCode::FAILURE;
        }
    }

    println!("{}", render(&report.context, Some(&report)));
    ExitCode::SUCCESS
}

/// Reports a refused context both ways, and says how the process ended.
///
/// Machine-readable as well as human-readable, because a gate needs to record
/// what went wrong and not only that the process exited nonzero: the object is
/// the same `opened: false` shape the desktop fixture's failure produces, so a
/// checker reads one document whichever fixture ran.
fn report_failure(error: &str) -> ExitCode {
    println!("{{\"opened\":false,\"error\":{}}}", json_string(error));
    eprintln!("{error}");
    ExitCode::FAILURE
}

/// `WIDTHxHEIGHT`, both nonzero.
fn parse_extent(raw: &str) -> Option<[u32; 2]> {
    let (width, height) = raw.split_once('x')?;
    let width: u32 = width.parse().ok()?;
    let height: u32 = height.parse().ok()?;
    (width > 0 && height > 0).then_some([width, height])
}

/// Parses the fixture's deliberately small exact-profile vocabulary.
///
/// The CLI has no `latest` spelling: omitting the option is the only way to
/// request discovery fallback, so a matrix invocation cannot accidentally turn
/// a minimum-version case into an unrecorded discovery run.
fn parse_gles_version(raw: &str) -> Option<EglGlesVersion> {
    match raw {
        "3.0" => Some(EglGlesVersion::V3_0),
        "3.1" => Some(EglGlesVersion::V3_1),
        "3.2" => Some(EglGlesVersion::V3_2),
        _ => None,
    }
}

/// The report as one JSON object.
///
/// Hand-written rather than derived: this fixture is a consumer of a doc-hidden
/// contract and adding a serialization dependency to it would make the evidence
/// depend on a third runtime for no gain.  The shape is flat and the checker
/// reads it by name, so there is nothing here a derive would do better.
///
/// `workload` is the one nested object, and it is nested rather than flattened
/// so that the two fixtures' documents have the same keys in the same places: a
/// gate that reads one reads the other, and a reviewer diffing them sees the
/// context reading and nothing else diverge.
fn render(report: &NativeGlContextReport, workload: Option<&NativeGlDrawReport>) -> String {
    let mut fields = vec![
        ("opened".to_owned(), "true".to_owned()),
        (
            "requested_minimum_version".to_owned(),
            report
                .requested_minimum_version
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_owned()),
        ),
        ("profile".to_owned(), json_string(&report.profile)),
        ("version".to_owned(), json_string(&report.version)),
        (
            "shading_language_version".to_owned(),
            json_string(&report.shading_language_version),
        ),
        ("vendor".to_owned(), json_string(&report.vendor)),
        ("renderer".to_owned(), json_string(&report.renderer)),
        (
            "driver_or_browser".to_owned(),
            json_string(&report.driver_or_browser),
        ),
        ("debug".to_owned(), report.debug.to_string()),
        (
            "forward_compatible".to_owned(),
            report.forward_compatible.to_string(),
        ),
        ("robust_access".to_owned(), report.robust_access.to_string()),
        ("no_error".to_owned(), report.no_error.to_string()),
        ("other_flags".to_owned(), json_strings(&report.other_flags)),
        (
            "reported_extension_count".to_owned(),
            report.reported_extension_count.to_string(),
        ),
        (
            "reported_extensions".to_owned(),
            json_strings(&report.reported_extensions),
        ),
        (
            "typed_extensions".to_owned(),
            json_pairs(&report.typed_extensions),
        ),
        ("capabilities".to_owned(), json_flags(&report.capabilities)),
        ("limits".to_owned(), json_pairs(&report.limits)),
        (
            "surface_facts".to_owned(),
            json_string(&report.surface_facts),
        ),
        (
            "drawable_extent".to_owned(),
            format!(
                "[{}, {}]",
                report.drawable_extent[0], report.drawable_extent[1]
            ),
        ),
        ("owner_thread".to_owned(), json_string(&report.owner_thread)),
    ];
    if let Some(workload) = workload {
        fields.push(("workload".to_owned(), json_workload(workload)));
    }
    fields.sort_by(|left, right| left.0.cmp(&right.0));
    let body = fields
        .into_iter()
        .map(|(name, value)| format!("  {}: {}", json_string(&name), value))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{body}\n}}")
}

/// What one driven run cost, as a nested JSON object.
///
/// Every field is a `u64` counter or a small string, and each one is named after
/// the report field it came from, so the mapping from this document back to
/// `NativeGlDrawReport` needs no table.
///
/// The three submission tallies this family never writes are absent here for the
/// same reason they are absent from the report: a `draws` field that is
/// structurally zero would read as "no draws were emitted", and the emitted work
/// is recorded per domain instead.
fn json_workload(report: &NativeGlDrawReport) -> String {
    // Joined rather than printed with a separator after each row, because a
    // trailing comma before `]` is not JSON and a gate reading this document
    // would refuse the whole run for a reason that has nothing to do with the
    // context.
    let rows = report
        .domains
        .iter()
        .map(|domain| {
            format!(
                "    {{\"domain\": {}, \"requests\": {}, \"emitted\": {}, \
                 \"skipped\": {}, \"unknown_recoveries\": {}}}",
                json_string(&domain.domain),
                domain.requests,
                domain.emitted,
                domain.skipped,
                domain.unknown_recoveries
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let domains = format!("[\n{rows}\n  ]");
    let mut fields = vec![
        ("mode".to_owned(), json_string(&report.mode)),
        (
            "draws_requested".to_owned(),
            report.draws_requested.to_string(),
        ),
        ("passes".to_owned(), report.passes.to_string()),
        ("pass_loads".to_owned(), report.pass_loads.to_string()),
        ("pass_stores".to_owned(), report.pass_stores.to_string()),
        ("cache_hits".to_owned(), report.cache_hits.to_string()),
        ("cache_misses".to_owned(), report.cache_misses.to_string()),
        ("cache_created".to_owned(), report.cache_created.to_string()),
        ("cache_evicted".to_owned(), report.cache_evicted.to_string()),
        (
            "cache_live_entries".to_owned(),
            report.cache_live_entries.to_string(),
        ),
        (
            "cache_live_bytes".to_owned(),
            report.cache_live_bytes.to_string(),
        ),
        (
            "steady_state_allocations".to_owned(),
            report.steady_state_allocations.to_string(),
        ),
        (
            "binding_bytes_copied".to_owned(),
            report.binding_bytes_copied.to_string(),
        ),
        ("submit_nanos".to_owned(), report.submit_nanos.to_string()),
        ("total_nanos".to_owned(), report.total_nanos.to_string()),
        (
            "drawable_extent".to_owned(),
            format!(
                "[{}, {}]",
                report.drawable_extent[0], report.drawable_extent[1]
            ),
        ),
    ];
    if let Some(colour) = report.colour.as_ref() {
        fields.push(("colour".to_owned(), json_colour(colour)));
    }
    let body = fields
        .into_iter()
        .map(|(name, value)| format!("    {}: {}", json_string(&name), value))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{body},\n    \"domains\": {domains}\n  }}")
}

/// The frame's colour target, as a nested JSON object.
///
/// The pixels are stated by value and in the order the file holds them, which is
/// the order the family produced: `row_order` says which order that is and the
/// grid below it is read in that order, left to right.  Stating them twice --
/// once as bytes on disk and once as hex here -- is the point rather than a
/// duplication: the file is what a reviewer looks at, and this is what a checker
/// asserts on without decoding it.
///
/// The whole grid is written out because the workload's target is a fixed
/// four-by-four.  A run that could render an arbitrary extent would write the
/// file and summarise here instead; there is nothing to summarise at sixteen
/// pixels, and a summary is the one form in which a wrong pixel can hide.
fn json_colour(colour: &ColourReadback) -> String {
    let pixels = colour
        .bytes
        .chunks_exact(4)
        .map(|pixel| {
            format!(
                "\"{:02x}{:02x}{:02x}{:02x}\"",
                pixel[0], pixel[1], pixel[2], pixel[3]
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let fields = [
        (
            "extent".to_owned(),
            format!("[{}, {}]", colour.extent[0], colour.extent[1]),
        ),
        ("row_order".to_owned(), json_string(colour.row_order)),
        ("bytes".to_owned(), colour.bytes.len().to_string()),
        ("pixels_rgba8".to_owned(), format!("[{pixels}]")),
    ];
    let body = fields
        .into_iter()
        .map(|(name, value)| format!("      {}: {}", json_string(&name), value))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{body}\n    }}")
}

fn json_strings(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn json_pairs(pairs: &[(String, String)]) -> String {
    format!(
        "[{}]",
        pairs
            .iter()
            .map(|(name, value)| format!("[{}, {}]", json_string(name), json_string(value)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn json_flags(flags: &[(String, bool)]) -> String {
    format!(
        "[{}]",
        flags
            .iter()
            .map(|(name, value)| format!("[{}, {}]", json_string(name), value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// A JSON string literal for `raw`.
///
/// Driver strings reach this function untouched, so the escaping has to be real
/// rather than reassuring: a vendor string containing a quote or a control
/// character must not be able to end the literal early and produce a document
/// the gate then reads as something else.
fn json_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    out.push('"');
    for character in raw.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control < ' ' => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
