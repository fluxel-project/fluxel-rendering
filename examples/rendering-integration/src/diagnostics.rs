//! Per-attempt native diagnostic observations for the integration fixture.
//!
//! The observer only records diagnostics from the exact device/context used by
//! the fixture. Diagnostic absence remains evidence only for this attempt.

use std::{error::Error, fs, path::Path};

#[cfg(all(windows, feature = "dx12"))]
pub(crate) fn record_dx12_device_open(output: &Path) -> Result<(), Box<dyn Error>> {
    let enabled = std::env::var_os("FLUXEL_DX12_DEBUG").is_some();
    fs::write(
        output.join("dx12-debug-request.json"),
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"requested\": {},\n",
                "  \"deviceOpenCompleted\": true,\n",
                "  \"activationContract\": \"fluxel-rhi b5d569feace548a6ed498901deaf31a5efcaca61 crates/dx12/src/backend/device/mod.rs: InstanceBackend::open calls D3D12GetDebugInterface and EnableDebugLayer before D3D12CreateDevice when FLUXEL_DX12_DEBUG is present\"\n",
                "}}\n"
            ),
            enabled,
        ),
    )?;
    Ok(())
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) struct WglObserver {
    version: String,
    khr_debug_extension: bool,
    core_debug_output: bool,
    activation_marker_observed: bool,
    activation_messages: Vec<String>,
    observed_messages: Vec<String>,
    log_saturated: bool,
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl WglObserver {
    const ACTIVATION_MARKER: &'static str = "fluxel-rendering WGL diagnostic observer active";

    /// Enables KHR_debug output and proves that the exact RHI-owned context
    /// accepted and returned an application marker through its debug log.
    pub(crate) fn install(
        device: &fluxel_rhi::api::Device<fluxel_rhi::gl::Gl>,
    ) -> Result<Self, Box<dyn Error>> {
        use fluxel_rhi::{api::NativeInteropRequest, gl::NativeGlApi as _};

        let interop = device
            .native_interop()
            .ok_or("WGL device did not expose native interop")?;
        // SAFETY: the guard makes the RHI-owned context current on its owner
        // thread. These calls only configure and query KHR_debug diagnostics;
        // they do not use or mutate RHI-owned resources.
        let (version, khr_debug_extension, core_debug_output, activation) = unsafe {
            let guard = interop.acquire(NativeInteropRequest::Device)?;
            let gl = &*guard
                .raw_pointer()
                .cast::<fluxel_rhi::gl::NativeGlContext>();
            let version = gl.get_parameter_string(glow::VERSION);
            let khr_debug_extension = gl.supported_extensions().contains("GL_KHR_debug");
            let core_debug_output = gl_version_at_least(&version, 4, 3);
            let available = khr_debug_extension || core_debug_output;
            if !available {
                return Err("WGL context has neither OpenGL 4.3+ nor GL_KHR_debug".into());
            }
            gl.enable(glow::DEBUG_OUTPUT);
            gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
            gl.debug_message_control(glow::DONT_CARE, glow::DONT_CARE, glow::DONT_CARE, &[], true);
            // Pass-label notifications can fill the driver's bounded log.
            // Exclude only application push/pop notifications at admission;
            // API/driver errors, warnings and performance messages stay on.
            for kind in [glow::DEBUG_TYPE_PUSH_GROUP, glow::DEBUG_TYPE_POP_GROUP] {
                gl.debug_message_control(
                    glow::DEBUG_SOURCE_APPLICATION,
                    kind,
                    glow::DEBUG_SEVERITY_NOTIFICATION,
                    &[],
                    false,
                );
            }
            gl.debug_message_insert(
                glow::DEBUG_SOURCE_APPLICATION,
                glow::DEBUG_TYPE_MARKER,
                0x4658_454c,
                glow::DEBUG_SEVERITY_NOTIFICATION,
                Self::ACTIVATION_MARKER,
            );
            (
                version,
                khr_debug_extension,
                core_debug_output,
                drain_messages(gl),
            )
        };
        let (activation_messages, log_saturated) = activation;
        let activation_marker_observed = activation_messages
            .iter()
            .any(|entry| is_activation_marker(entry));
        if !activation_marker_observed || log_saturated {
            return Err(format!("WGL KHR_debug observer did not return its application marker; raw activation queue: {activation_messages:?}").into());
        }
        Ok(Self {
            version,
            khr_debug_extension,
            core_debug_output,
            activation_marker_observed,
            activation_messages,
            observed_messages: Vec::new(),
            log_saturated,
        })
    }

    /// Drains every message present after the workloads, retaining application
    /// activation evidence separately from native diagnostics.
    pub(crate) fn write(
        &mut self,
        device: &fluxel_rhi::api::Device<fluxel_rhi::gl::Gl>,
        output: &Path,
    ) -> Result<(), Box<dyn Error>> {
        use fluxel_rhi::api::NativeInteropRequest;

        let interop = device
            .native_interop()
            .ok_or("WGL device did not expose native interop")?;
        // SAFETY: the exact context remains current only for this scoped
        // read-only diagnostic query, and no RHI object is accessed.
        let messages = unsafe {
            let guard = interop.acquire(NativeInteropRequest::Device)?;
            let gl = &*guard
                .raw_pointer()
                .cast::<fluxel_rhi::gl::NativeGlContext>();
            drain_messages(gl)
        };
        // Keep diagnostics that were already queued while the observer was
        // enabled. Filtering only the final drain would lose early driver
        // messages when a workload later returns an error.
        self.log_saturated |= messages.1;
        self.observed_messages.extend(messages.0);
        let all = self
            .activation_messages
            .iter()
            .chain(self.observed_messages.iter());
        let native = all
            .clone()
            .filter(|entry| !is_application_marker(entry))
            .cloned()
            .collect::<Vec<_>>();
        let application = all
            .filter(|entry| is_application_marker(entry))
            .cloned()
            .collect::<Vec<_>>();
        let activation = self.activation_messages.clone();
        fs::write(
            output.join("wgl-debug-observer.json"),
            format!(
                concat!(
                    "{{\n",
                    "  \"schemaVersion\": 1,\n",
                    "  \"installed\": true,\n",
                    "  \"observer\": \"KHR_debug output plus message-log drain through scoped native interop\",\n",
                    "  \"logSaturated\": {},\n",
                    "  \"applicationPushPopNotificationsAdmitted\": false,\n",
                    "  \"contextVersion\": {},\n",
                    "  \"khrDebugExtension\": {},\n",
                    "  \"coreDebugOutput\": {},\n",
                    "  \"debugOutputAvailable\": {},\n",
                    "  \"activationMarkerObserved\": {},\n",
                    "  \"activationMessages\": {},\n",
                    "  \"applicationMarkers\": {},\n",
                    "  \"nativeDiagnostics\": {}\n",
                    "}}\n"
                ),
                self.log_saturated,
                json_string(&self.version),
                self.khr_debug_extension,
                self.core_debug_output,
                self.khr_debug_extension || self.core_debug_output,
                self.activation_marker_observed,
                json_array(&activation),
                json_array(&application),
                json_array(&native),
            ),
        )?;
        Ok(())
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn gl_version_at_least(value: &str, required_major: u32, required_minor: u32) -> bool {
    let mut parts = value.splitn(3, |character: char| {
        !(character.is_ascii_digit() || character == '.')
    });
    let Some(version) = parts.next() else {
        return false;
    };
    let mut numbers = version.split('.');
    let (Some(Ok(major)), Some(Ok(minor))) = (
        numbers.next().map(str::parse::<u32>),
        numbers.next().map(str::parse::<u32>),
    ) else {
        return false;
    };
    major > required_major || (major == required_major && minor >= required_minor)
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
unsafe fn drain_messages(gl: &fluxel_rhi::gl::NativeGlContext) -> (Vec<String>, bool) {
    use fluxel_rhi::gl::NativeGlApi as _;
    let mut messages = Vec::new();
    // SAFETY: the caller holds the current-context interop lease.
    let capacity = unsafe { gl.get_parameter_i32(glow::MAX_DEBUG_LOGGED_MESSAGES) };
    let mut saturated = capacity <= 0;
    loop {
        // SAFETY: the caller holds the native interop lease for this context.
        let queued = unsafe { gl.get_parameter_i32(glow::DEBUG_LOGGED_MESSAGES) }.max(0) as u32;
        saturated |= capacity > 0 && queued >= capacity as u32;
        if queued == 0 {
            return (messages, saturated);
        }
        // SAFETY: the same lease keeps the context current during the drain.
        let batch = unsafe { gl.get_debug_message_log(queued) };
        if batch.is_empty() {
            return (messages, true);
        }
        messages.extend(batch.into_iter().map(|entry| format!("{entry:?}")));
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn is_application_marker(raw: &str) -> bool {
    // glow =0.18.0 exposes Debug but no getters for this entry type. Decode
    // only its exact numeric header; unknown representations fail closed and
    // remain native diagnostics. Message text never grants an exemption.
    let Some(header) = raw.strip_prefix("DebugMessageLogEntry { source: ") else {
        return false;
    };
    let Some((source, rest)) = header.split_once(", msg_type: ") else {
        return false;
    };
    let Some((kind, rest)) = rest.split_once(", id: ") else {
        return false;
    };
    let Some((id, rest)) = rest.split_once(", severity: ") else {
        return false;
    };
    let Some((severity, message)) = rest.split_once(", message: ") else {
        return false;
    };
    source.parse::<u32>() == Ok(glow::DEBUG_SOURCE_APPLICATION)
        && id.parse::<u32>().is_ok()
        && severity.parse::<u32>() == Ok(glow::DEBUG_SEVERITY_NOTIFICATION)
        && message.starts_with('"')
        && message.ends_with("\" }")
        && matches!(
            kind.parse::<u32>(),
            Ok(glow::DEBUG_TYPE_MARKER | glow::DEBUG_TYPE_PUSH_GROUP | glow::DEBUG_TYPE_POP_GROUP)
        )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn is_activation_marker(raw: &str) -> bool {
    raw == format!(
        "DebugMessageLogEntry {{ source: {}, msg_type: {}, id: {}, severity: {}, message: {:?} }}",
        glow::DEBUG_SOURCE_APPLICATION,
        glow::DEBUG_TYPE_MARKER,
        0x4658_454c_u32,
        glow::DEBUG_SEVERITY_NOTIFICATION,
        WglObserver::ACTIVATION_MARKER
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn json_string(value: &str) -> String {
    let mut escaped = String::from("\"");
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => escaped.push_str(&format!("\\u{:04x}", value as u32)),
            value => escaped.push(value),
        }
    }
    escaped.push('\"');
    escaped
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn json_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(all(test, windows, feature = "native-gl-wgl"))]
#[path = "tests/diagnostics.rs"]
mod tests;
