//! Driving a real GLES context over an EGL pbuffer through the compatibility
//! adapter, and reporting what the run cost.
//!
//! # Why this is not the WGL entry with a different context
//!
//! The workload, the report vocabulary and the request parsing are shared --
//! they are in [`super`] because they are facts about a *run*.  What differs is
//! everything about how the context is obtained, and the difference is not
//! cosmetic:
//!
//! - **There is no host.**  [`super::drive_desktop_gl4_draws`] takes a
//!   `HasWindowHandle + HasDisplayHandle` because WGL needs a drawable the
//!   caller owns.  An EGL pbuffer is offscreen by construction, so this entry
//!   takes an extent and nothing else.  A signature that still demanded a window
//!   would be demanding one it never reads.
//! - **The surface is created at a size, not observed at one.**  A window
//!   surface has whatever extent the Host's window has; a pbuffer has exactly
//!   the extent it was created with, which is why the caller states it and why
//!   the report echoes it back.
//! - **It is not presentable.**  `EglGlesContext::present` refuses a pbuffer
//!   rather than pretending offscreen work was shown, so nothing here presents
//!   and the report carries no presentation fact.
//!
//! # The seam with the provider, and why it is shaped this way
//!
//! `scripts/check_gl_architecture.py` allows `compat` to name `api` but forbids
//! it from naming `glow`, and forbids `api/egl` from naming `compat`.  So this
//! file may hold an `EglGlesContext` but may not write a `glow` type, and the
//! provider may not learn what a `NativeGlDrawReport` is.  The join is
//! [`EglGlesContext::with_current`], which hands a closure a `&glow::Context`
//! whose type this file never spells -- exactly how [`super`] already drives WGL.
//!
//! # What this file cannot prove, and does not claim
//!
//! A run here says a GLES implementation accepted a 3.1 pbuffer context and
//! executed the frame.  It says nothing about the *device* behind that
//! implementation: an emulator's guest GL strings are a presented profile rather
//! than hardware.  (`CLAUDE.md` §4.5 -- a driver string is not a correctness
//! proof, and the series plan's condition on a physical GLES 3.1 device stands
//! separately.)

use crate::webgl2::api::{EglGlesContext, EglGlesVersion, EglPbufferSize};

use super::super::{readback, workload};
use super::{ColourReadback, NativeGlDrawReport, parse_request};
use crate::webgl2::api::NativeGlProvider;
use crate::webgl2::conformance;

/// The GLES version this entry asks for.
///
/// Exactly 3.1, and not "3.0 or better".  The profile ledger distinguishes the
/// embedded versions, and a provider that quietly downgraded would answer a
/// different question than the one the evidence is collected for -- so a driver
/// that cannot give 3.1 fails here rather than reporting a 3.0 run under a 3.1
/// heading.  `EglGlesContext::new_inner` refuses an exact request the driver
/// cannot honour and does not fall back on its own.
const VERSION: EglGlesVersion = EglGlesVersion::V3_1;

/// Opens a real GLES context over an EGL pbuffer and drives `draws` indexed
/// draws through the compatibility adapter.
///
/// `mode` is `"optimized"` or `"oracle"`; `draws` must be nonzero, and
/// `identity` must be nonzero.  `extent` is the pbuffer's size, in pixels.  The
/// context is made current on the calling thread, driven, and destroyed before
/// this returns; nothing here borrows a caller's object, because there is none
/// to borrow.
///
/// `read_colour` asks for the frame's own picture as well as its cost, under the
/// same rule and for the same reason as the WGL entry: the readback has to
/// happen inside the one call that still holds the frame's lease, so a caller
/// who wants both cannot get them from two calls.
///
/// # Errors
///
/// The `Err` is a rendered description, for the reason the WGL entry gives: the
/// typed errors here are crate-private, and what a hardware gate needs from a
/// failure is the message.
pub fn drive_gles_pbuffer_draws(
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
    read_colour: bool,
) -> Result<NativeGlDrawReport, String> {
    let (mode, stamp) = parse_request(mode, identity, draws)?;
    let size = EglPbufferSize {
        width: extent[0],
        height: extent[1],
    };

    let mut context = EglGlesContext::new_pbuffer(stamp, size, VERSION)
        .map_err(|error| format!("the EGL pbuffer context did not open: {error:?}"))?;
    // The reading is taken from the snapshot the provider gathered and validated
    // while opening, so this cannot disagree with the context the frame runs on.
    let snapshot = context
        .discover()
        .map_err(|error| format!("the context opened but its discovery is unreadable: {error:?}"))?
        .clone();
    let reading = conformance::report(&snapshot, extent, &context.owner_thread());

    // Where the readback lands, when one was asked for: a local rather than a
    // return value, because the hook that fills it runs *inside* the frame and
    // the report is assembled after the context is gone.
    let mut readback: Option<readback::TexturePixels> = None;
    let outcome = {
        let pixels = &mut readback;
        context.with_current("drive the representative workload", |gl| {
            // SAFETY: `with_current` made this context current on this thread and
            // owns it for the whole call, and `snapshot` is the evidence this
            // exact context produced during its own discovery -- the pair of
            // conditions `from_discovered` requires.
            let backend = unsafe { NativeGlProvider::from_discovered(gl, snapshot.clone()) };
            Ok(workload::drive_with(
                backend,
                mode,
                draws,
                extent,
                // This surface has no host clock to borrow: the entry is not
                // windowed and nothing above it owns an epoch for it.  `Instant`
                // is available here because a GLES build is never wasm, and the
                // epoch is taken once so the closure hands out plain nanoseconds.
                {
                    let epoch = std::time::Instant::now();
                    move || epoch.elapsed().as_nanos() as u64
                },
                move |device, texture| {
                    // A run that asked for no picture reads nothing, so its
                    // clocked path is the one every other measurement takes.
                    if !read_colour {
                        return Ok(());
                    }
                    *pixels = Some(device.read_texture(texture).map_err(|error| {
                        format!("the frame's colour target was not readable: {error:?}")
                    })?);
                    Ok(())
                },
            ))
        })
    };

    // Teardown is reported rather than swallowed: a context that refuses to
    // dispose has left EGL objects behind, and a caller reading a successful
    // report would have no way to know.  It runs on every path below, including
    // the failing ones, so the leak cannot depend on whether the run worked.
    let disposed = context.dispose();

    let report = match outcome {
        Err(error) => Err(format!(
            "the context refused to run the workload: {error:?}"
        )),
        // The workload's own refusals keep their own message: they are statements
        // about this frame rather than about the context, and prefixing them with
        // a context failure would point a reader at the driver.
        Ok(Err(inner)) => Err(inner),
        Ok(Ok(cost)) => Ok(NativeGlDrawReport::from_cost(
            reading,
            cost,
            readback.map(ColourReadback::from_pixels),
        )),
    };
    disposed.map_err(|error| format!("the EGL context did not dispose cleanly: {error:?}"))?;
    report
}
