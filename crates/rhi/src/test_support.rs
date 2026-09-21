//! Narrow hardware-evidence surface for the standalone GL-family fixtures.
//!
//! It returns reports only; neither a native context nor a portable device can
//! escape.  The platform adapters own WGL/EGL creation and currentness.

/// A minimum desktop OpenGL core version requested by the Windows evidence
/// fixture.
///
/// This is a fixture input, not a portable RHI capability level. A value is
/// accepted only for the v13 desktop support interval, GL 4.0 through GL 4.6.
/// Omitting it from the fixture keeps the production-like "highest available"
/// WGL selection path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DesktopGlVersion {
    major: u8,
    minor: u8,
}

impl DesktopGlVersion {
    /// Creates a supported desktop GL minimum-version test request.
    pub const fn new(major: u8, minor: u8) -> Option<Self> {
        if major == 4 && minor <= 6 {
            Some(Self { major, minor })
        } else {
            None
        }
    }

    /// The requested major component.
    pub const fn major(self) -> u8 {
        self.major
    }

    /// The requested minor component.
    pub const fn minor(self) -> u8 {
        self.minor
    }
}

/// Raw colour data read from the fixture's own colour target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColourReadback {
    pub extent: [u32; 2],
    pub bytes: Vec<u8>,
    pub row_order: &'static str,
}

/// A real context's driver-discovery projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGlContextReport {
    /// The optional minimum version requested by the fixture.
    ///
    /// `None` means WGL's ordinary highest-available fallback was used. When
    /// present, [`Self::version`] is guaranteed to be this version or newer,
    /// rather than necessarily equal to it.
    pub requested_minimum_version: Option<String>,
    pub profile: String,
    /// The version string observed from the actual current driver context.
    pub version: String,
    pub shading_language_version: String,
    pub vendor: String,
    pub renderer: String,
    pub driver_or_browser: String,
    pub debug: bool,
    pub forward_compatible: bool,
    pub robust_access: bool,
    pub no_error: bool,
    pub other_flags: Vec<String>,
    pub reported_extension_count: usize,
    pub reported_extensions: Vec<String>,
    pub typed_extensions: Vec<(String, String)>,
    pub capabilities: Vec<(String, bool)>,
    pub limits: Vec<(String, String)>,
    pub surface_facts: String,
    pub drawable_extent: [u32; 2],
    pub owner_thread: String,
}

/// Per-state-domain traffic captured from one fixture execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainTally {
    pub domain: String,
    pub requests: u64,
    pub emitted: u64,
    pub skipped: u64,
    pub unknown_recoveries: u64,
}

/// Measured native draw execution and optional readback evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGlDrawReport {
    pub context: NativeGlContextReport,
    pub mode: String,
    pub draws_requested: u32,
    pub passes: u64,
    pub pass_loads: u64,
    pub pass_stores: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_created: u64,
    pub cache_evicted: u64,
    pub cache_live_entries: u64,
    pub cache_live_bytes: u64,
    pub steady_state_allocations: u64,
    pub binding_bytes_copied: u64,
    pub domains: Vec<DomainTally>,
    pub submit_nanos: u64,
    pub total_nanos: u64,
    pub drawable_extent: [u32; 2],
    pub colour: Option<ColourReadback>,
}

fn validate_request(
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: Option<u32>,
) -> Result<(), String> {
    if identity == 0 {
        return Err("a device identity has to be nonzero".into());
    }
    if extent.contains(&0) {
        return Err("a native drawable extent has to be nonzero".into());
    }
    if !matches!(mode, "optimized" | "oracle") {
        return Err(format!(
            "unknown execution mode `{mode}`; expected `optimized` or `oracle`"
        ));
    }
    if draws == Some(0) {
        return Err("the evidence workload requires at least one draw".into());
    }
    Ok(())
}

/// Opens a WGL context over a host window and returns driver evidence.
///
/// This declaration is deliberately present only for the fixture's exact
/// native feature.  The creation/lowering implementation is supplied by the
/// WGL platform adapter; until it is wired, refusal is explicit rather than a
/// fabricated report.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn observe_desktop_gl4_context<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
) -> Result<NativeGlContextReport, String>
where
    H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    observe_desktop_gl4_context_minimum_version(host, extent, identity, None)
}

/// Opens a WGL desktop core context with an optional minimum version and
/// returns its driver evidence.
///
/// `None` follows WGL's normal highest-available fallback path. `Some` makes
/// one minimum-version request and accepts an actual driver context of that
/// version or newer. The report retains both the request and observed version,
/// so a GL 4.0--4.6 matrix never mistakes a newer context for an exact context.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn observe_desktop_gl4_context_minimum_version<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
    version: Option<DesktopGlVersion>,
) -> Result<NativeGlContextReport, String>
where
    H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    let window = host
        .window_handle()
        .map_err(|e| format!("the host's window handle is not available: {e:?}"))?;
    let display = host
        .display_handle()
        .map_err(|e| format!("the host's display handle is not available: {e:?}"))?;
    validate_request(extent, identity, "optimized", None)?;
    let stamp = evidence_stamp(identity)?;
    let surface = match version {
        Some(version) => {
            crate::backend::gl::native::wgl::WglContextSurface::open_with_minimum_desktop_version(
                stamp,
                window,
                display,
                extent,
                crate::backend::gl::api::GlVersion::new(version.major(), version.minor()),
            )
        }
        None => {
            crate::backend::gl::native::wgl::WglContextSurface::open(stamp, window, display, extent)
        }
    }
    .map_err(|error| format!("failed to open WGL GL4 context: {error:?}"))?;
    Ok(report_from_wgl(surface.evidence(), version))
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn drive_desktop_gl4_draws<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
    _read_colour: bool,
) -> Result<NativeGlDrawReport, String>
where
    H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    drive_desktop_gl4_draws_minimum_version(host, extent, identity, mode, draws, _read_colour, None)
}

/// Drives the WGL evidence workload against an optional desktop core minimum.
///
/// See [`observe_desktop_gl4_context_minimum_version`] for the meaning of
/// `version`.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn drive_desktop_gl4_draws_minimum_version<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
    read_colour: bool,
    version: Option<DesktopGlVersion>,
) -> Result<NativeGlDrawReport, String>
where
    H: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    let window = host
        .window_handle()
        .map_err(|e| format!("the host's window handle is not available: {e:?}"))?;
    let display = host
        .display_handle()
        .map_err(|e| format!("the host's display handle is not available: {e:?}"))?;
    validate_request(extent, identity, mode, Some(draws))?;
    let started = std::time::Instant::now();
    let stamp = evidence_stamp(identity)?;
    let surface = match version {
        Some(version) => {
            crate::backend::gl::native::wgl::WglContextSurface::open_with_minimum_desktop_version(
                stamp,
                window,
                display,
                extent,
                crate::backend::gl::api::GlVersion::new(version.major(), version.minor()),
            )
        }
        None => {
            crate::backend::gl::native::wgl::WglContextSurface::open(stamp, window, display, extent)
        }
    }
    .map_err(|error| format!("failed to open WGL GL4 context: {error:?}"))?;
    let context = report_from_wgl(surface.evidence(), version);
    let submitted = std::time::Instant::now();
    let colour = surface
        .draw_evidence(draws, read_colour)
        .map_err(|error| format!("WGL evidence raster workload failed: {error:?}"))?
        .map(|bytes| ColourReadback {
            extent,
            bytes,
            row_order: "gl-bottom-left",
        });
    surface
        .present()
        .map_err(|error| format!("WGL evidence presentation failed: {error:?}"))?;
    let elapsed = started.elapsed();
    Ok(NativeGlDrawReport {
        context,
        mode: mode.to_owned(),
        draws_requested: draws,
        passes: 1,
        pass_loads: 1,
        pass_stores: 1,
        cache_hits: 0,
        cache_misses: 0,
        cache_created: 0,
        cache_evicted: 0,
        cache_live_entries: 0,
        cache_live_bytes: 0,
        steady_state_allocations: 0,
        binding_bytes_copied: 0,
        domains: Vec::new(),
        submit_nanos: submitted.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        total_nanos: elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
        drawable_extent: extent,
        colour,
    })
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn evidence_stamp(identity: u64) -> Result<crate::backend::gl::api::ContextStamp, String> {
    let device = crate::backend::gl::api::DeviceIdentity::new(identity)
        .ok_or("a device identity has to be nonzero")?;
    Ok(crate::backend::gl::api::ContextStamp::new(
        device,
        crate::backend::gl::api::ContextEpoch::INITIAL,
    ))
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn report_from_wgl(
    evidence: crate::backend::gl::native::wgl::WglEvidence,
    requested_minimum_version: Option<DesktopGlVersion>,
) -> NativeGlContextReport {
    NativeGlContextReport {
        requested_minimum_version: requested_minimum_version
            .map(|version| format!("{}.{}", version.major(), version.minor())),
        profile: evidence.profile,
        version: evidence.version,
        shading_language_version: evidence.shading_language_version,
        vendor: evidence.vendor,
        renderer: evidence.renderer,
        driver_or_browser: evidence.driver_or_browser,
        debug: evidence.debug,
        forward_compatible: evidence.forward_compatible,
        robust_access: evidence.robust_access,
        no_error: evidence.no_error,
        other_flags: evidence.other_flags,
        reported_extension_count: evidence.extensions.len(),
        reported_extensions: evidence.extensions,
        typed_extensions: evidence.typed_extensions,
        capabilities: evidence.capabilities,
        limits: evidence.limits,
        surface_facts: evidence.surface_facts,
        drawable_extent: evidence.drawable_extent,
        owner_thread: format!("{:?}", std::thread::current().id()),
    }
}

/// A minimum GLES profile requested by the EGL fixture.
///
/// This is fixture vocabulary, not portable RHI capability vocabulary. Passing
/// one of these values asks EGL for a GLES floor; a newer observed ES 3.x
/// context is accepted and reported, while an older context is rejected.
/// `None` deliberately retains the evidence fixture's normal newest-to-oldest
/// fallback, which is useful when a harness is only discovering the device's
/// best GLES 3.x profile.
#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EglGlesVersion {
    /// Require at least OpenGL ES 3.0.
    V3_0,
    /// Require at least OpenGL ES 3.1.
    V3_1,
    /// Require at least OpenGL ES 3.2.
    V3_2,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
impl EglGlesVersion {
    fn native(self) -> crate::backend::gl::native::egl::EglGlesVersion {
        match self {
            Self::V3_0 => crate::backend::gl::native::egl::EglGlesVersion::V3_0,
            Self::V3_1 => crate::backend::gl::native::egl::EglGlesVersion::V3_1,
            Self::V3_2 => crate::backend::gl::native::egl::EglGlesVersion::V3_2,
        }
    }
}

/// Opens an EGL GLES pbuffer and drives the evidence workload.
///
/// A concrete `requested_version` is an evidence floor. The current context
/// must observe that GLES version or a newer one; an older version is rejected.
/// The report records both the requested minimum and the driver-observed
/// version. `None` probes 3.2, 3.1, then 3.0, retaining the fixture's
/// historical discovery behaviour.
#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
pub fn drive_gles_pbuffer_draws(
    extent: [u32; 2],
    identity: u64,
    mode: &str,
    draws: u32,
    read_colour: bool,
    requested_version: Option<EglGlesVersion>,
) -> Result<NativeGlDrawReport, String> {
    validate_request(extent, identity, mode, Some(draws))?;
    let started = std::time::Instant::now();
    let stamp = evidence_stamp_gles(identity)?;
    let size = crate::backend::gl::native::egl::EglPbufferSize {
        width: extent[0],
        height: extent[1],
    };
    let mut context = match requested_version {
        Some(version) => crate::backend::gl::native::egl::EglGlesContext::new_pbuffer(
            stamp,
            size,
            version.native(),
        )
        .map_err(|error| {
            format!(
                "EGL could not create a context satisfying the requested GLES {version:?} floor: {error:?}"
            )
        })?,
        None => {
            // EGL's minor-version request is extension-gated.  Discovery tries
            // the newest concrete ES 3 profile and keeps the first profile the
            // implementation actually accepts; no profile is inferred from a
            // request enum alone.
            let mut last_error = None;
            let mut selected = None;
            for version in [
                crate::backend::gl::native::egl::EglGlesVersion::V3_2,
                crate::backend::gl::native::egl::EglGlesVersion::V3_1,
                crate::backend::gl::native::egl::EglGlesVersion::V3_0,
            ] {
                match crate::backend::gl::native::egl::EglGlesContext::new_pbuffer(
                    stamp, size, version,
                ) {
                    Ok(value) => {
                        selected = Some(value);
                        break;
                    }
                    Err(error) => last_error = Some(format!("{error:?}")),
                }
            }
            selected.ok_or_else(|| {
                format!(
                    "EGL could not create an ES 3.0-or-newer pbuffer context: {}",
                    last_error.unwrap_or_else(|| "no EGL context attempt was made".into())
                )
            })?
        }
    };
    let evidence = context.evidence();
    let submitted = std::time::Instant::now();
    let colour = context
        .draw_evidence(draws, read_colour)
        .map_err(|error| format!("EGL/GLES evidence raster workload failed: {error:?}"))?
        .map(|bytes| ColourReadback {
            extent,
            bytes,
            row_order: "gl-bottom-left",
        });
    context
        .dispose()
        .map_err(|error| format!("EGL teardown failed: {error:?}"))?;
    Ok(NativeGlDrawReport {
        context: report_from_egl(evidence, requested_version),
        mode: mode.to_owned(),
        draws_requested: draws,
        passes: 1,
        pass_loads: 1,
        pass_stores: 1,
        cache_hits: 0,
        cache_misses: 0,
        cache_created: 0,
        cache_evicted: 0,
        cache_live_entries: 0,
        cache_live_bytes: 0,
        steady_state_allocations: 0,
        binding_bytes_copied: 0,
        domains: Vec::new(),
        submit_nanos: submitted.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        total_nanos: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        drawable_extent: extent,
        colour,
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
fn evidence_stamp_gles(identity: u64) -> Result<crate::backend::gl::api::ContextStamp, String> {
    let device = crate::backend::gl::api::DeviceIdentity::new(identity)
        .ok_or("a device identity has to be nonzero")?;
    Ok(crate::backend::gl::api::ContextStamp::new(
        device,
        crate::backend::gl::api::ContextEpoch::INITIAL,
    ))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native-gles-egl"))]
fn report_from_egl(
    evidence: crate::backend::gl::native::egl::EglEvidence,
    requested_minimum_version: Option<EglGlesVersion>,
) -> NativeGlContextReport {
    NativeGlContextReport {
        requested_minimum_version: requested_minimum_version.map(|version| match version {
            EglGlesVersion::V3_0 => "3.0".to_owned(),
            EglGlesVersion::V3_1 => "3.1".to_owned(),
            EglGlesVersion::V3_2 => "3.2".to_owned(),
        }),
        profile: evidence.profile,
        version: evidence.version,
        shading_language_version: evidence.shading_language_version,
        vendor: evidence.vendor,
        renderer: evidence.renderer,
        driver_or_browser: evidence.driver_or_browser,
        debug: false,
        forward_compatible: false,
        robust_access: false,
        no_error: false,
        other_flags: Vec::new(),
        reported_extension_count: evidence.extensions.len(),
        reported_extensions: evidence.extensions,
        typed_extensions: Vec::new(),
        capabilities: evidence.capabilities,
        limits: evidence.limits,
        surface_facts: "EGL pbuffer".into(),
        drawable_extent: evidence.extent,
        owner_thread: format!("{:?}", std::thread::current().id()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bad_evidence_requests_fail_before_platform_access() {
        assert!(validate_request([0, 1], 1, "optimized", Some(1)).is_err());
        assert!(validate_request([1, 1], 1, "bad", Some(1)).is_err());
        assert!(validate_request([1, 1], 1, "oracle", Some(1)).is_ok());
    }
}
