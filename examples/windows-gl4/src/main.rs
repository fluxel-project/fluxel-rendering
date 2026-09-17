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
//! ```powershell
//! cargo run --manifest-path examples/windows-gl4/Cargo.toml -- --frames 1
//! ```

use std::process::ExitCode;

use fluxel_host::{Window, WindowConfig};
use fluxel_rhi::test_support::{DesktopGl4ContextReport, observe_desktop_gl4_context};

/// The default client extent, and the extent the context is opened for.
///
/// Small on purpose: this fixture collects context evidence rather than
/// pixels, so the drawable only has to be a real one the driver will accept.
const DEFAULT_EXTENT: [u32; 2] = [640, 480];

fn main() -> ExitCode {
    let mut extent = DEFAULT_EXTENT;
    let mut frames = 1_u32;
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
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
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

    let report = match observe_desktop_gl4_context(&window, extent, 1) {
        Ok(report) => report,
        Err(error) => {
            // Machine-readable failure too: a gate needs to record what went
            // wrong, not only that the process exited nonzero.
            println!("{{\"opened\":false,\"error\":{}}}", json_string(&error));
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    for _ in 1..frames {
        if let Err(error) = window.poll_events() {
            eprintln!("pumping the window failed: {error}");
            return ExitCode::FAILURE;
        }
    }

    println!("{}", render(&report));
    if window.close_requested() {
        // Not a failure: the window closing is the fixture's normal ending.
        eprintln!("the window was closed during the run");
    }
    ExitCode::SUCCESS
}

/// `WIDTHxHEIGHT`, both nonzero.
fn parse_extent(raw: &str) -> Option<[u32; 2]> {
    let (width, height) = raw.split_once('x')?;
    let width: u32 = width.parse().ok()?;
    let height: u32 = height.parse().ok()?;
    (width > 0 && height > 0).then_some([width, height])
}

/// The report as one JSON object.
///
/// Hand-written rather than derived: this fixture is a consumer of a
/// doc-hidden contract and adding a serialization dependency to it would make
/// the evidence depend on a third runtime for no gain.  The shape is flat and
/// the checker reads it by name, so there is nothing here a derive would do
/// better.
fn render(report: &DesktopGl4ContextReport) -> String {
    let mut fields = vec![
        ("opened".to_owned(), "true".to_owned()),
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
    fields.sort_by(|left, right| left.0.cmp(&right.0));
    let body = fields
        .into_iter()
        .map(|(name, value)| format!("  {}: {}", json_string(&name), value))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{body}\n}}")
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
