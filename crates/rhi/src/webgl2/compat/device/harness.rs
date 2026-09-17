//! Driving a real DesktopGl4 context through the compatibility adapter, and
//! reporting what the run cost.
//!
//! # Why this exists, and why it is not [`crate::webgl2::conformance`]
//!
//! `conformance` answers *what is this context*.  It opens one, reads the
//! discovery snapshot, drops it, and returns a projection -- observe and drop in
//! one call, which is the whole reason it could exist before the question "may a
//! caller open a GL-family device?" was answered.  That shape cannot measure
//! anything a *frame* does: a run has to keep the provider stack alive across a
//! compiled graph, an executor, a submission and a completion, and it has to
//! report what the run emitted rather than what the driver claimed about itself.
//!
//! This is the second entry of that family.  It takes the same narrow host
//! protocol (a raw drawable through the standard handle traits, plus the extent
//! the caller says it has) and returns plain data, but its data is a cost: the
//! per-domain tallies Layer 2 keeps, the cache traffic behind them, and the two
//! durations the layer deliberately does not record.  Nothing here is public API
//! and nothing here becomes one -- see the note on `mode` below.
//!
//! # What is left here once the workload moved out
//!
//! Everything that is a fact about *this* surface: opening a WGL context over
//! the caller's drawable, taking the desktop reading that only this surface can
//! take, and projecting both into one report.  The workload itself -- the graph,
//! the draw loop, the counters -- is [`super::workload`], shared with the browser
//! surface so that the cached-versus-uncached differential compares two contexts
//! rather than two frames.
//!
//! # The mode is a string, and that is not a style preference
//!
//! The differential this entry exists for is the same frame driven twice, once
//! with Layer 2 allowed to skip a redundant call and once with it forced to emit
//! every one.  That knob is [`ExecutionMode`], and it is `pub(crate)`: which mode
//! a *renderer* runs is not a choice the common contract offers, so it must not
//! appear in a signature an out-of-workspace caller can name.  The caller
//! therefore spells the mode, this module parses it, and the crate-private type
//! stays crate-private.  An unrecognized spelling is refused rather than
//! defaulted, because a differential whose two halves silently ran the same mode
//! is worse than one that failed.

use std::time::Instant;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::readback;
use super::workload::{self, DrawCost};
use crate::webgl2::api::{
    ContextEpoch, ContextStamp, DeviceIdentity, NativeGlProvider, WglContextSurface,
};
use crate::webgl2::conformance::{self, DesktopGl4ContextReport};
use crate::webgl2::state::ExecutionMode;

pub use super::workload::DomainTally;

/// The pixels the frame left in its exported colour target.
///
/// Deliberately plain data with no accessor: the consumer is an out-of-workspace
/// fixture that writes them to a file, and every type on the path that produced
/// them is crate-private.
///
/// The row order is carried rather than left to the reader because it is the one
/// property of these bytes that a consumer can silently get wrong and still
/// produce a plausible-looking image.  The family's copy verbs follow the GL
/// bottom-left convention, so the first row here is the *bottom* one and an
/// image written straight out would be upside down.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColourReadback {
    /// The pixel extent that was read.
    pub extent: [u32; 2],
    /// Four eight-bit channels per pixel, `extent[0] * extent[1] * 4` bytes.
    pub bytes: Vec<u8>,
    /// The row order [`Self::bytes`] is in.
    pub row_order: &'static str,
}

impl ColourReadback {
    /// The readback as this report states it.
    fn from_pixels(pixels: readback::TexturePixels) -> Self {
        Self {
            extent: pixels.extent,
            bytes: pixels.bytes,
            row_order: "gl-bottom-left",
        }
    }
}

/// What one driven run of the workload cost, together with the context it ran on.
///
/// The context reading is carried rather than left to a second call, and the
/// reason is a driver fact this entry learned by being run: `SetPixelFormat` may
/// be called **once** per window, so a process cannot open two WGL contexts over
/// the same drawable -- the second is refused with `PixelFormatAlreadyConfigured`
/// before anything is measured.  What a caller would otherwise do is observe the
/// context, drop it, then reopen it to drive the frame, and that second open is
/// exactly the call the driver refuses.  So the reading rides along with the run
/// that produced it, and [`crate::test_support::observe_desktop_gl4_context`]
/// stays what it is: the entry for callers who want the reading and no run.
///
/// # Three of the submission tallies are absent, and why
///
/// `StateCounters::submissions` carries twelve tallies because the other
/// backends need all twelve.  This family writes **three** of them -- `passes`,
/// `pass_loads` and `pass_stores`, all from its session domain -- and never
/// increments `draws`, `clears`, `presents` or the rest.  A report that carried
/// those would be reporting a field that is zero because nothing writes it, which
/// is a worse answer than no field: it reads as "no draws were emitted".  So they
/// are absent here, and the emitted work is read where it is actually recorded,
/// per domain, as `emitted` against `requests`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesktopGl4DrawReport {
    /// What the context this run drove answered when it was asked.
    pub context: DesktopGl4ContextReport,
    /// The execution mode the run actually used, as it was parsed.
    pub mode: String,
    /// How many draws the caller asked for.
    ///
    /// What was *asked*, not what was counted: this family has no tally of draws
    /// issued, so pairing this with a "draws emitted" would be an invention.  The
    /// load-bearing number next to it is the per-domain `emitted`.
    pub draws_requested: u32,
    /// Raster passes begun.
    pub passes: u64,
    /// Attachments loaded rather than cleared.
    pub pass_loads: u64,
    /// Attachments stored at pass end.
    pub pass_stores: u64,
    /// Derived-cache hits.
    pub cache_hits: u64,
    /// Derived-cache misses.
    pub cache_misses: u64,
    /// Derived objects the cache created.
    pub cache_created: u64,
    /// Derived objects it evicted.
    pub cache_evicted: u64,
    /// Derived objects it still holds.
    pub cache_live_entries: u64,
    /// Bytes those entries account for.
    pub cache_live_bytes: u64,
    /// Steady-state heap allocations made while applying state.
    pub steady_state_allocations: u64,
    /// Bytes copied while applying state.
    pub binding_bytes_copied: u64,
    /// Every domain's tallies, in the layer's own stable order.
    pub domains: Vec<DomainTally>,
    /// How long the executor call took, excluding context creation.
    pub submit_nanos: u64,
    /// How long the whole in-context run took, including object creation.
    pub total_nanos: u64,
    /// The extent the context was opened for.
    pub drawable_extent: [u32; 2],
    /// What the frame left in its exported colour target, when the caller asked
    /// for it.
    ///
    /// `None` means the caller asked for a cost and no picture, and it is the
    /// only thing that distinguishes such a run: the readback is a GL call on the
    /// clocked path, so a run that does not want the pixels must not pay for
    /// them, and one that does gets them measured inside `total_nanos`.
    pub colour: Option<ColourReadback>,
}

impl DesktopGl4DrawReport {
    /// This surface's context reading, over a cost the shared workload measured.
    ///
    /// Written out field by field rather than derived, because the two structs
    /// are deliberately not the same type: the cost is what *both* surfaces
    /// measure, and this report is what only the native surface can add to it.
    /// A `Deref` or an embedded field would tie the public report's shape to the
    /// shared one, and the shared one is free to change when the funnel adds
    /// variety to the workload.
    fn from_cost(
        context: DesktopGl4ContextReport,
        cost: DrawCost,
        colour: Option<ColourReadback>,
    ) -> Self {
        Self {
            context,
            mode: cost.mode,
            draws_requested: cost.draws_requested,
            passes: cost.passes,
            pass_loads: cost.pass_loads,
            pass_stores: cost.pass_stores,
            cache_hits: cost.cache_hits,
            cache_misses: cost.cache_misses,
            cache_created: cost.cache_created,
            cache_evicted: cost.cache_evicted,
            cache_live_entries: cost.cache_live_entries,
            cache_live_bytes: cost.cache_live_bytes,
            steady_state_allocations: cost.steady_state_allocations,
            binding_bytes_copied: cost.binding_bytes_copied,
            domains: cost.domains,
            submit_nanos: cost.submit_nanos,
            total_nanos: cost.total_nanos,
            drawable_extent: cost.drawable_extent,
            colour,
        }
    }
}

/// Opens a real desktop GL context over `host`'s drawable and drives `draws`
/// indexed draws through the compatibility adapter.
///
/// `mode` is `"optimized"` or `"oracle"`; `draws` must be nonzero.  The context
/// is made current on the calling thread, driven, and destroyed before this
/// returns, so nothing borrowed from `host` outlives it.
///
/// `read_colour` asks for the frame's own picture as well as its cost.  It is a
/// parameter and not a second entry point because the two cannot be separated:
/// the readback has to happen inside the one call that still holds the frame's
/// lease, so a caller who wants both cannot get them from two calls.  A caller
/// who wants only the cost passes `false` and pays nothing for a readback, which
/// is the shape every measurement in the funnel uses -- the readback is a GL
/// command pair on the clocked path, and a cost that silently carried one would
/// not be the cost the funnel compared.
///
/// # Errors
///
/// The `Err` is a rendered description, for the reason
/// [`crate::test_support::observe_desktop_gl4_context`] gives: the typed errors
/// here are crate-private, and what a hardware gate needs from a failure is the
/// message.
pub fn drive_desktop_gl4_draws<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
    read_colour: bool,
) -> Result<DesktopGl4DrawReport, String>
where
    H: HasWindowHandle + HasDisplayHandle,
{
    let (mode, stamp) = parse_request(mode, identity, draws)?;
    let window = host
        .window_handle()
        .map_err(|error| format!("the host's window handle is not available: {error:?}"))?;
    let display = host
        .display_handle()
        .map_err(|error| format!("the host's display handle is not available: {error:?}"))?;

    // Both handles borrow `host`, and the context that adopts them is dropped
    // before this function returns, so neither outlives the window it names.
    let context = WglContextSurface::open(stamp, window, display, extent)
        .map_err(|error| format!("the WGL context did not open: {error:?}"))?;
    let snapshot = context
        .discover()
        .map_err(|error| format!("the context opened but discovery refused it: {error:?}"))?
        .clone();
    // The context reading is taken here because here is the only place it can be:
    // this context is the only one this process gets over this drawable, so a
    // caller who wants the reading and the run gets both from this call or
    // neither.
    let reading = conformance::report(&snapshot, extent, &context.owner_thread());

    // The whole run happens inside one `with_current`, and it has to: the
    // provider borrows the `glow::Context` for its entire lifetime, so the
    // context has to be current for every call the executor makes, not just for
    // the one that built it.  The callback's own error channel is the context's,
    // so the run's message-shaped failures ride out through the inner `Result`
    // rather than being forced into a driver error they are not.
    // The clock this surface hands the workload.  `Instant` is the host's, and it
    // is supplied from here rather than taken by the workload because the browser
    // surface cannot use it at all: `Instant::now` panics on
    // `wasm32-unknown-unknown`, and a module that reached for a clock itself would
    // be a module that only runs on one of its two surfaces.  The epoch is taken
    // once so the closure hands out plain nanoseconds from it.
    let epoch = Instant::now();
    let now_nanos = || epoch.elapsed().as_nanos() as u64;

    // Where the readback lands, when one was asked for.  It is a local rather
    // than a return value because the hook that fills it cannot be the thing
    // that returns: the hook runs *inside* the frame, and the report it feeds is
    // assembled after the context has closed.
    let mut readback: Option<readback::TexturePixels> = None;
    let outcome = context.with_current("drive the representative workload", |gl| {
        // SAFETY: `with_current` made this context current on this thread and
        // owns it for the whole call, and `snapshot` is the evidence this exact
        // context produced during its own discovery -- which is the pair of
        // conditions `from_discovered` requires.
        let backend = unsafe { NativeGlProvider::from_discovered(gl, snapshot.clone()) };
        let pixels = &mut readback;
        Ok(workload::drive_with(
            backend,
            mode,
            draws,
            extent,
            now_nanos,
            move |device, texture| {
                // A run that asked for no picture reads nothing, so its clocked
                // path is the one every other measurement takes.
                if !read_colour {
                    return Ok(());
                }
                *pixels = Some(device.read_texture(texture).map_err(|error| {
                    format!("the frame's colour target was not readable: {error:?}")
                })?);
                Ok(())
            },
        ))
    });
    match outcome {
        Err(error) => Err(format!(
            "the context refused to run the workload: {error:?}"
        )),
        // The workload's own refusals keep their own message: they are statements
        // about this frame rather than about the context, and prefixing them with
        // a context failure would point a reader at the driver.
        Ok(Err(inner)) => Err(inner),
        Ok(Ok(cost)) => Ok(DesktopGl4DrawReport::from_cost(
            reading,
            cost,
            readback.map(ColourReadback::from_pixels),
        )),
    }
}

/// Everything [`drive_desktop_gl4_draws`] can refuse before it opens anything.
///
/// Split out for the same reason the context is opened last: a request that is
/// going to be refused should be refused while it is still a request.  It also
/// makes the refusals testable without a window, which matters because the rest
/// of this module cannot be.
fn parse_request(
    mode: &str,
    identity: u64,
    draws: u32,
) -> Result<(ExecutionMode, ContextStamp), String> {
    let mode = parse_mode(mode)?;
    if draws == 0 {
        return Err(
            "a workload of zero draws measures only the setup, so it is refused".to_owned(),
        );
    }
    let device = DeviceIdentity::new(identity)
        .ok_or_else(|| "a device identity has to be nonzero".to_owned())?;
    Ok((mode, ContextStamp::new(device, ContextEpoch::INITIAL)))
}

fn parse_mode(mode: &str) -> Result<ExecutionMode, String> {
    match mode {
        "optimized" => Ok(ExecutionMode::Optimized),
        "oracle" => Ok(ExecutionMode::Oracle),
        other => Err(format!(
            "an execution mode is `optimized` or `oracle`, and `{other}` is neither"
        )),
    }
}

#[cfg(test)]
mod tests;
