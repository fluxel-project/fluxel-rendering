//! Desktop GL 4.x hardware evidence, collected on a real context.
//!
//! This fixture is the window half of a split and nothing else.  `fluxel-host`
//! owns the native window and its message pump; `fluxel-rhi`'s doc-hidden
//! conformance entry owns opening the context and reading the driver, because
//! that needs the GL family's private identities and no out-of-workspace caller
//! can be given them without widening a contract this series froze.  The two
//! halves meet at the standard raw-handle traits and nowhere else.
//!
//! It prints one JSON object describing exactly what the driver answered, which
//! is the evidence a gate consumes; it does not decide whether the answer is
//! good.  Verdicts belong to the checking script, so that a run's output is a
//! record rather than an opinion.
//!
//! # The second thing it can be asked for
//!
//! With `--draws N` it drives a measured workload instead -- one raster pass
//! issuing `N` indexed draws through the compatibility adapter -- and reports
//! what that run cost alongside what the context turned out to be.  Both
//! readings come from the same call because they have to: `SetPixelFormat` may
//! be called once per window, so a process gets exactly one WGL context over a
//! given drawable, and observing-then-reopening is the second open the driver
//! refuses with `PixelFormatAlreadyConfigured`.  The cost run therefore carries
//! the context reading with it, and a run that asks for no workload prints
//! exactly the object it printed before the flag existed.
//!
//! # The third thing it can be asked for
//!
//! With `--readback PATH` the workload run also reads its colour target back and
//! writes the raw bytes to `PATH`, so that the picture a real desktop GL4 context
//! produced can be *looked at* and not only counted.  The pixels are reported in
//! the same JSON by value as well, because the target is a fixed four-by-four and
//! a reviewer checking coverage and orientation should not have to decode a file
//! to do it.
//!
//! The bytes are written in the order the family produced them and the report
//! says which order that is; nothing here flips them.  A fixture that reversed
//! rows would be inventing an interpretation, and the one thing a consumer of
//! this output must not have to guess is whether the image is upside down.
//!
//! ```powershell
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --frames 1
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --gl-version 4.0
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --gl-version 4.6 --draws 1
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --draws 2000 --mode oracle
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --draws 1 --readback target/evidence/gl4.rgba
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use fluxel_host::{Window, WindowConfig};
use fluxel_rhi::test_support::{
    ColourReadback, DesktopGlVersion, NativeGlContextReport, NativeGlDrawReport,
    drive_desktop_gl4_draws_minimum_version, observe_desktop_gl4_context_minimum_version,
};

/// The default client extent, and the extent the context is opened for.
///
/// Small on purpose: this fixture collects context evidence rather than
/// pixels, so the drawable only has to be a real one the driver will accept.
const DEFAULT_EXTENT: [u32; 2] = [640, 480];

/// The identity the context is opened with, whichever reading is asked for.
///
/// One number for both paths because there is one context: a run either
/// observes it or drives it, and never does both over the same drawable.
const CONTEXT_IDENTITY: u64 = 1;

fn main() -> ExitCode {
    let mut extent = DEFAULT_EXTENT;
    let mut frames = 1_u32;
    let mut draws = 0_u32;
    let mut mode = String::from("optimized");
    let mut mode_given = false;
    let mut readback: Option<PathBuf> = None;
    // `None` intentionally means the normal production-like WGL path: request
    // the highest available core context, falling back to the v13 GL 4.0
    // floor. A concrete value is a minimum-version fixture request, useful for
    // walking the 4.0--4.6 compatibility matrix on one GL 4.6-capable machine.
    let mut gl_version: Option<DesktopGlVersion> = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--extent" => match arguments.next().as_deref().and_then(parse_extent) {
                Some(parsed) => extent = parsed,
                None => {
                    eprintln!("--extent wants WIDTHxHEIGHT, for example 640x480");
                    return ExitCode::from(2);
                }
            },
            "--frames" => match arguments.next().as_deref().and_then(|raw| raw.parse().ok()) {
                Some(parsed) => frames = parsed,
                None => {
                    eprintln!("--frames wants a non-negative count");
                    return ExitCode::from(2);
                }
            },
            "--draws" => match arguments.next().as_deref().and_then(|raw| raw.parse().ok()) {
                Some(parsed) => draws = parsed,
                None => {
                    eprintln!("--draws wants a positive count");
                    return ExitCode::from(2);
                }
            },
            "--mode" => match arguments.next() {
                Some(parsed) => {
                    mode = parsed;
                    mode_given = true;
                }
                None => {
                    eprintln!("--mode wants `optimized` or `oracle`");
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
            "--gl-version" => match arguments.next().as_deref().and_then(parse_gl_version) {
                Some(parsed) => gl_version = Some(parsed),
                None => {
                    eprintln!("--gl-version wants one of 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, or 4.6");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    if readback.is_some() && draws == 0 {
        // There is no frame to read a picture out of, and the alternative --
        // opening a context, driving nothing and writing a file of zeroes --
        // would be evidence of a frame that never happened.
        eprintln!("--readback only means something with --draws: without a workload there is no frame to read");
        return ExitCode::from(2);
    }
    if mode_given && draws == 0 {
        // The entry refuses a zero-draw workload for the same reason, so saying
        // it here turns a silent no-op into a usage error.
        eprintln!("--mode only means something with --draws: without a workload there is nothing to run");
        return ExitCode::from(2);
    }

    let config = match WindowConfig::new("fluxel desktop GL4 evidence", extent[0], extent[1]) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("the window configuration was refused: {error}");
            return ExitCode::FAILURE;
        }
    };
    let window = match Window::new(config) {
        Ok(window) => window,
        Err(error) => {
            eprintln!("the window did not open: {error}");
            return ExitCode::FAILURE;
        }
    };

    // The host owns the pump, so pumping is this loop's job rather than the
    // context entry's: an unfed Win32 queue is a window the driver may block on.
    if let Err(error) = window.poll_events() {
        eprintln!("pumping the window failed: {error}");
        return ExitCode::FAILURE;
    }

    // One context, one reading: the cost run opens the context and reports what
    // it found on the way in, because the driver grants this process exactly one
    // pixel format for this drawable.  Asking for both by opening twice is what
    // `PixelFormatAlreadyConfigured` refuses, so the flag picks a path rather
    // than adding one.
    let (report, workload) = if draws == 0 {
        match observe_desktop_gl4_context_minimum_version(
            &window,
            extent,
            CONTEXT_IDENTITY,
            gl_version,
        ) {
            Ok(report) => (report, None),
            Err(error) => return report_failure(&error),
        }
    } else {
        let wanted = readback.is_some();
        match drive_desktop_gl4_draws_minimum_version(
            &window,
            extent,
            CONTEXT_IDENTITY,
            &mode,
            draws,
            wanted,
            gl_version,
        ) {
            Ok(report) => (report.context.clone(), Some(report)),
            Err(error) => return report_failure(&error),
        }
    };

    // Written before the report is printed, so that a document naming a file
    // cannot be produced by a run whose file was never written.  A failure here
    // is a failure of the run: the evidence this flag exists for is the pair.
    if let (Some(path), Some(workload)) = (readback.as_ref(), workload.as_ref()) {
        if let Some(colour) = workload.colour.as_ref() {
            if let Err(error) = std::fs::write(path, &colour.bytes) {
                eprintln!("the colour target did not reach {}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }

    for _ in 1..frames {
        if let Err(error) = window.poll_events() {
            eprintln!("pumping the window failed: {error}");
            return ExitCode::FAILURE;
        }
    }

    println!("{}", render(&report, workload.as_ref()));
    if window.close_requested() {
        // Not a failure: the window closing is the fixture's normal ending.
        eprintln!("the window was closed during the run");
    }
    ExitCode::SUCCESS
}

/// Reports a refused context both ways, and says how the process ended.
///
/// Machine-readable as well as human-readable, because a gate needs to record
/// what went wrong and not only that the process exited nonzero: the object is
/// the same `opened: false` shape either entry's failure produces, so a checker
/// reads one document whether the context refused to open or refused to run.
fn report_failure(error: &str) -> ExitCode {
    println!("{{\"opened\":false,\"error\":{}}}", json_string(error));
    eprintln!("{error}");
    ExitCode::FAILURE
}

/// `WIDTHxHEIGHT`, both nonzero.
fn parse_extent(raw: &str) -> Option<[u32; 2]> {    let (width, height) = raw.split_once('x')?;
    let width: u32 = width.parse().ok()?;
    let height: u32 = height.parse().ok()?;
    (width > 0 && height > 0).then_some([width, height])
}

/// A v13 desktop GL minimum version for the WGL fixture.
fn parse_gl_version(raw: &str) -> Option<DesktopGlVersion> {
    let (major, minor) = raw.split_once('.')?;
    let major = major.parse().ok()?;
    let minor = minor.parse().ok()?;
    DesktopGlVersion::new(major, minor)
}

/// The report as one JSON object.
///
/// Hand-written rather than derived: this fixture is a consumer of a
/// doc-hidden contract and adding a serialization dependency to it would make
/// the evidence depend on a third runtime for no gain.  The shape is flat and
/// the checker reads it by name, so there is nothing here a derive would do
/// better.
///
/// `workload` is the one nested object, and it is nested rather than flattened
/// so that the keys of a run that asked for no workload are byte-for-byte the
/// keys it printed before the flag existed -- a gate reading this document by
/// name cannot be affected by a field it does not ask for, but an *absence* it
/// already tolerates is not something to start relying on.
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
        // `version` remains for existing consumers. `observed_version` makes
        // the minimum-request/actual-context distinction explicit in a matrix
        // record, where WGL may report a newer core context than requested.
        ("version".to_owned(), json_string(&report.version)),
        (
            "observed_version".to_owned(),
            json_string(&report.version),
        ),
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
        (
            "robust_access".to_owned(),
            report.robust_access.to_string(),
        ),
        ("no_error".to_owned(), report.no_error.to_string()),
        (
            "other_flags".to_owned(),
            json_strings(&report.other_flags),
        ),
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
        (
            "capabilities".to_owned(),
            json_flags(&report.capabilities),
        ),
        ("limits".to_owned(), json_pairs(&report.limits)),
        (
            "surface_facts".to_owned(),
            json_string(&report.surface_facts),
        ),
        (
            "drawable_extent".to_owned(),
            format!("[{}, {}]", report.drawable_extent[0], report.drawable_extent[1]),
        ),
        (
            "owner_thread".to_owned(),
            json_string(&report.owner_thread),
        ),
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
/// Every field is a `u64` counter or a small string, and each one is named
/// after the report field it came from, so the mapping from this document back
/// to `NativeGlDrawReport` needs no table.
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
        ("draws_requested".to_owned(), report.draws_requested.to_string()),
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
/// grid below it is read in that order, left to right.  Stating them twice -- once
/// as bytes on disk and once as hex here -- is the point rather than a
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
/// Driver strings reach this function untouched, so the escaping has to be
/// real rather than reassuring: a vendor string containing a quote or a control
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
