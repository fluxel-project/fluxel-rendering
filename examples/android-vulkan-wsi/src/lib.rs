#![cfg(target_os = "android")]

use core::ffi::{c_char, c_void};
use std::ffi::CString;

/// Prefix of NDK `ANativeActivity`; only `callbacks` is read here.
#[repr(C)]
pub struct ANativeActivity {
    callbacks: *mut ANativeActivityCallbacks,
}

/// Exact `ANativeActivityCallbacks` field order from NDK `native_activity.h`.
/// Android allocates this table; native code may replace individual callbacks,
/// but must never replace the table pointer itself.
#[repr(C)]
struct ANativeActivityCallbacks {
    on_start: *mut c_void,
    on_resume: *mut c_void,
    on_save_instance_state: *mut c_void,
    on_pause: *mut c_void,
    on_stop: *mut c_void,
    on_destroy: *mut c_void,
    on_window_focus_changed: *mut c_void,
    on_native_window_created: Option<unsafe extern "C" fn(*mut ANativeActivity, *mut c_void)>,
    on_native_window_resized: *mut c_void,
    on_native_window_redraw_needed: *mut c_void,
    on_native_window_destroyed: Option<unsafe extern "C" fn(*mut ANativeActivity, *mut c_void)>,
    on_input_queue_created: *mut c_void,
    on_input_queue_destroyed: *mut c_void,
    on_content_rect_changed: *mut c_void,
    on_configuration_changed: *mut c_void,
    on_low_memory: *mut c_void,
}

unsafe extern "C" fn window_created(_: *mut ANativeActivity, window: *mut c_void) {
    let report = fluxel_rhi::android_vulkan_wsi::run(window)
        .unwrap_or_else(|error| format!("{{\"error\":\"{}\"}}", error.replace('"', "'")));
    log(&report);
}

unsafe extern "C" fn window_destroyed(_: *mut ANativeActivity, window: *mut c_void) {
    // This is the framework's last valid native-window lifecycle boundary.
    // It drops the private registry guard, marks the target terminal, and pairs
    // the ANativeWindow acquire held by the Vulkan backend.
    fluxel_rhi::android_vulkan_wsi::on_native_window_destroyed(window);
    log("{\"schema\":\"fluxel.android-vulkan-wsi.v1\",\"window_destroyed\":true}");
}

fn log(message: &str) {
    let tag = CString::new("FluxelVulkanWSI").expect("static Android log tag");
    let message = CString::new(message).expect("JSON contains no NUL");
    unsafe {
        __android_log_write(4, tag.as_ptr(), message.as_ptr());
    }
}

#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(priority: i32, tag: *const c_char, text: *const c_char) -> i32;
}

/// NativeActivity entry point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ANativeActivity_onCreate(
    activity: *mut ANativeActivity,
    _saved_state: *mut c_void,
    _saved_state_size: usize,
) {
    // Android owns the activity and callback table. Installing callbacks is the
    // permitted lifecycle seam; allocating or replacing `callbacks` is not.
    unsafe {
        (*(*activity).callbacks).on_native_window_created = Some(window_created);
        (*(*activity).callbacks).on_native_window_destroyed = Some(window_destroyed);
    }
}
