//! Metal shader-library and entry-function lowering.
//!
//! The public artifact fixes the entry-point and stage before this code runs.
//! This module deliberately retains both the library and function: Metal's
//! function normally keeps its library alive, but doing so explicitly makes the
//! ownership boundary auditable and mirrors the portable `ShaderModule` handle.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::any::Any;
use std::sync::Arc;

use objc2::{
    msg_send,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLDevice, MTLFunction, MTLLibrary};

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::shader::backend::ShaderModuleBackend;
use crate::api::shader::{ShaderArtifact, ShaderCode};

use super::device::MetalShared;

#[link(name = "System")]
unsafe extern "C" {
    fn dispatch_data_create(
        buffer: NonNull<c_void>,
        size: usize,
        queue: Option<NonNull<c_void>>,
        destructor: *mut c_void,
    ) -> *mut AnyObject;
    fn dispatch_release(object: *mut AnyObject);
}

const DISPATCH_DATA_DESTRUCTOR_DEFAULT: *mut c_void = core::ptr::null_mut();

/// One compiled Metal library function accepted by this logical device.
pub(super) struct MetalShader {
    function: Retained<ProtocolObject<dyn MTLFunction>>,
    _library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Kept last so it is dropped after its Objective-C children.
    _shared: Arc<MetalShared>,
}

// Metal objects are immutable after creation; command encoder mutation stays
// on the device submission path. This is the same confined Objective-C Send /
// Sync boundary used for resources.
unsafe impl Send for MetalShader {}
unsafe impl Sync for MetalShader {}

impl MetalShader {
    pub(super) fn function(&self) -> &ProtocolObject<dyn MTLFunction> {
        &self.function
    }
}

impl ShaderModuleBackend for MetalShader {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Compiles source or loads a metallib then resolves the artifact's one entry
/// point. Runtime compiler failures are normal structured creation errors;
/// they never become a capability claim.
pub(super) fn create_shader(
    shared: Arc<MetalShared>,
    artifact: &ShaderArtifact,
) -> RhiResult<MetalShader> {
    let library = match &artifact.code {
        ShaderCode::Msl(source) => {
            let source = NSString::from_str(source);
            shared
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| native_error("compile MSL source", &error))?
        }
        ShaderCode::Metallib(bytes) => new_library_from_metallib(&shared.device, bytes)?,
        _ => {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                "Metal accepts only MSL source or metallib shader artifacts",
            )
            .at("MetalDevice::create_shader"));
        }
    };
    let name = NSString::from_str(&artifact.entry_point);
    let function = library.newFunctionWithName(&name).ok_or_else(|| {
        RhiError::new(
            RhiErrorKind::Unsupported,
            "the Metal library does not contain the artifact entry point",
        )
        .at("MetalDevice::create_shader")
    })?;
    Ok(MetalShader {
        function,
        _library: library,
        _shared: shared,
    })
}

fn native_error(operation: &'static str, _error: &NSError) -> RhiError {
    // NSError text is intentionally not copied into the portable error: its
    // locale and lifetime are platform details. Diagnostics may attach it on
    // macOS, while this stable message remains useful in captures and tests.
    RhiError::new(RhiErrorKind::BackendFailure, operation).at("MetalDevice::create_shader")
}

/// `objc2-metal` intentionally does not require the still-evolving `dispatch2`
/// crate just to pass immutable metallib bytes. This is the narrow ownership
/// bridge recommended by wgpu-hal: dispatch copies/retains the data for the
/// call, then the temporary `dispatch_data_t` is released immediately.
fn new_library_from_metallib(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> RhiResult<Retained<ProtocolObject<dyn MTLLibrary>>> {
    if bytes.is_empty() {
        // Metal's API has no useful empty-data result, and `NonNull` would be
        // fabricated for it. Reject it at the same boundary as native loading.
        return Err(
            RhiError::new(RhiErrorKind::InvalidUsage, "a metallib artifact is empty")
                .at("MetalDevice::create_shader"),
        );
    }
    let buffer = NonNull::new(bytes.as_ptr().cast_mut())
        .ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "a non-empty metallib slice unexpectedly had a null address",
            )
            .at("MetalDevice::create_shader")
        })?
        .cast();
    let data = unsafe {
        dispatch_data_create(buffer, bytes.len(), None, DISPATCH_DATA_DESTRUCTOR_DEFAULT)
    };
    // libdispatch documents a retained non-null dispatch_data_t for valid
    // arguments. A null here is an allocation failure that Objective-C cannot
    // report through NSError, so turn it into an ordinary backend error below.
    if data.is_null() {
        return Err(RhiError::new(
            RhiErrorKind::OutOfMemory,
            "Metal could not allocate dispatch data for the metallib",
        )
        .at("MetalDevice::create_shader"));
    }
    let result = unsafe { msg_send![device, newLibraryWithData: data, error: _] };
    unsafe { dispatch_release(data) };
    result.map_err(|error| native_error("load metallib bytes", &error))
}
