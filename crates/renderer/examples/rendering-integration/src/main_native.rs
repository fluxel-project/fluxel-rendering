//! Backend opening and selected integration workload dispatch.
use super::main_copy::run_copy;
use super::main_demo;
use super::main_support::*;
use super::*;

#[cfg(all(windows, feature = "dx12"))]
pub(super) fn run_dx12(output: &Path, workload: Workload) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{api::DeviceRequest, dx12::Dx12Instance};

    let native = Dx12Instance::new()?;
    let instance = native.instance()?;
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no DX12 adapter was enumerated")?;
    let adapter_name = adapter.name.clone();
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    diagnostics::record_dx12_device_open(output)?;
    write_capabilities(&device, "dx12", &adapter_name, output)?;
    if let Some(route) = main_demo::route(workload) {
        return main_demo::run_dx12(device, route, &output.join(main_demo::output_name(route)));
    }
    if let Workload::S8Material(config) = workload {
        return s8_material::run(device, &output.join("s8-material-baseline"), config);
    }
    if workload == Workload::S8Lifecycle {
        let replacement = block_on(instance.open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        }))?;
        return s8_material::run_lifecycle(
            device,
            replacement,
            &output.join("s8-material-lifecycle"),
        );
    }
    if workload == Workload::S7Deferred {
        return s7_deferred::run(device, &output.join("s7-deferred"));
    }
    if workload == Workload::S7DeferredSurface {
        return s7_surface::run_dx12(device, &output.join("s7-deferred-surface"));
    }
    if workload == Workload::S7DeviceIsolation {
        let replacement = block_on(instance.open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        }))?;
        return s7_device_isolation::run(
            device,
            replacement,
            &output.join("s7-deferred-device-isolation"),
        );
    }
    if workload == Workload::S6Composed {
        return s6_composed::run(device, &output.join("s6-composed"));
    }
    if workload == Workload::S6Surface {
        return s6_composed::run_surface_dx12(device, &output.join("s6-surface"));
    }
    if workload == Workload::S5Forward {
        return s5_forward::run(device, &output.join("s5-forward"));
    }
    if workload == Workload::S5Surface {
        return s5_forward_surface::run_dx12(device, &output.join("s5-forward-surface"));
    }
    if workload == Workload::S5DeviceIsolation {
        let replacement = block_on(instance.open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        }))?;
        return s5_forward::run_device_isolation(
            device,
            replacement,
            &output.join("s5-forward-device-isolation"),
        );
    }
    if workload == Workload::S3Material {
        return s3_material::run(device, &output.join("s3-material"));
    }
    if workload == Workload::S2Cache {
        let first_output = output.join("s2-shader");
        fs::create_dir_all(&first_output)?;
        write_capabilities(&device, "dx12", &adapter_name, &first_output)?;
        let mut s2_consumer = s2_shader::S2ShaderConsumer::new();
        s2_shader::run(&mut s2_consumer, device.clone(), &first_output, false)?;

        let same_device_output = output.join("s2-shader-same-device-cache-hit");
        fs::create_dir_all(&same_device_output)?;
        write_capabilities(&device, "dx12", &adapter_name, &same_device_output)?;
        s2_shader::run(&mut s2_consumer, device.clone(), &same_device_output, true)?;

        let replacement = block_on(instance.open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        }))?;
        if replacement.id() == device.id() {
            return Err("DX12 replacement unexpectedly retained the original DeviceId".into());
        }
        let replacement_output = output.join("s2-shader-replacement-device");
        fs::create_dir_all(&replacement_output)?;
        write_capabilities(&replacement, "dx12", &adapter_name, &replacement_output)?;
        s2_shader::run(
            &mut s2_consumer,
            replacement.clone(),
            &replacement_output,
            false,
        )?;
        verify_s2_cache_replacement(&mut s2_consumer, device.id(), replacement.id(), output)?;
        return Ok(());
    }
    let old_submission = run_copy(device.clone(), &adapter_name, output)?;
    raster::run(device.clone(), &output.join("raster"))?;
    compute::run(device.clone(), &output.join("compute"))?;
    read_set::run(device.clone(), &output.join("read-set"))?;
    let mut s2_consumer = s2_shader::S2ShaderConsumer::new();
    s2_shader::run(
        &mut s2_consumer,
        device.clone(),
        &output.join("s2-shader"),
        false,
    )?;
    s2_shader::run(
        &mut s2_consumer,
        device.clone(),
        &output.join("s2-shader-same-device-cache-hit"),
        true,
    )?;
    s2_texture::run(device.clone(), &output.join("s2-texture"))?;
    s3_material::run(device.clone(), &output.join("s3-material"))?;
    surface::run_dx12(device.clone(), &output.join("surface"))?;
    s5_forward_surface::run_dx12(device.clone(), &output.join("s5-forward-surface"))?;
    let old_pipeline = raster::make_pipeline(&device, fluxel_rhi::api::TextureFormat::Rgba8Unorm)?;
    let replacement = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    if replacement.id() == device.id() {
        return Err("DX12 replacement unexpectedly retained the original DeviceId".into());
    }
    let replacement_s2_output = output.join("s2-shader-replacement-device");
    fs::create_dir_all(&replacement_s2_output)?;
    write_capabilities(&replacement, "dx12", &adapter_name, &replacement_s2_output)?;
    s2_shader::run(
        &mut s2_consumer,
        replacement.clone(),
        &replacement_s2_output,
        false,
    )?;
    verify_s2_cache_replacement(&mut s2_consumer, device.id(), replacement.id(), output)?;
    raster::replacement_rejects_old_pipeline(old_pipeline, replacement, &old_submission, output)
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(super) fn run_wgl(output: &Path, workload: Workload) -> Result<(), Box<dyn Error>> {
    use fluxel_host::{Window, WindowConfig, WindowVisibility};
    use fluxel_rhi::{
        api::{AdapterId, DeviceRequest, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };

    if workload == Workload::S2Cache {
        return Err("--s2-cache-only is currently supported only with the dx12 feature".into());
    }
    let (window_title, window_extent) = main_demo::wgl_window_properties(workload);
    let window = Window::new(
        WindowConfig::new(window_title, window_extent[0], window_extent[1])?
            .with_visibility(WindowVisibility::Hidden),
    )?;
    // SAFETY: the window remains alive for the instance, device, and all work
    // in `run_copy`, and this function remains on its context owner thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, window_extent) }?;
    let id = InstanceId::allocate()?;
    // SAFETY: this Core instance owns the adopted native context while it opens.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL adapter was enumerated")?;
    let adapter_name = adapter.name.clone();
    let device = block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    write_capabilities(&device, "native-gl-wgl", &adapter_name, output)?;
    let mut observer = diagnostics::WglObserver::install(&device)?;
    if let Some(route) = main_demo::route(workload) {
        let result = main_demo::run_wgl(
            device.clone(),
            &window,
            route,
            &output.join(main_demo::output_name(route)),
        );
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if let Workload::S8Material(config) = workload {
        let result = s8_material::run(device.clone(), &output.join("s8-material-baseline"), config);
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S8Lifecycle {
        let replacement_window = Window::new(
            WindowConfig::new("Fluxel S8 lifecycle replacement WGL", 320, 180)?
                .with_visibility(WindowVisibility::Hidden),
        )?;
        // SAFETY: both hidden roots remain alive through the lifecycle evidence.
        let replacement_native =
            unsafe { GlInstance::from_wgl_window(&replacement_window, [320, 180]) }?;
        let replacement_id = InstanceId::allocate()?;
        // SAFETY: the replacement instance exclusively owns this WGL context.
        let replacement_instance =
            unsafe { Instance::<Gl>::from_native(replacement_id, replacement_native) };
        let replacement_adapter = replacement_instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or("no WGL S8 lifecycle replacement adapter was enumerated")?;
        let replacement = block_on(replacement_instance.open(DeviceRequest {
            adapter: AdapterId {
                instance: replacement_id,
                ordinal: replacement_adapter.id.ordinal,
            },
            required_features: Vec::new(),
        }))?;
        let mut replacement_observer = diagnostics::WglObserver::install(&replacement)?;
        let replacement_diagnostics = output.join("s8-lifecycle-replacement-diagnostics");
        fs::create_dir_all(&replacement_diagnostics)?;
        let result = s8_material::run_lifecycle(
            device.clone(),
            replacement.clone(),
            &output.join("s8-material-lifecycle"),
        );
        let replacement_result = replacement_observer.write(&replacement, &replacement_diagnostics);
        let old_result = observer.write(&device, output);
        if result.is_err() || replacement_result.is_err() || old_result.is_err() {
            std::mem::forget(replacement_window);
            std::mem::forget(window);
        }
        return result.and(replacement_result).and(old_result);
    }
    if workload == Workload::S7Deferred {
        let result = s7_deferred::run(device.clone(), &output.join("s7-deferred"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S7DeferredSurface {
        let result =
            s7_surface::run_wgl(device.clone(), &window, &output.join("s7-deferred-surface"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S7DeviceIsolation {
        let replacement_window = Window::new(
            WindowConfig::new("Fluxel S7 Deferred replacement WGL", 320, 180)?
                .with_visibility(WindowVisibility::Hidden),
        )?;
        // SAFETY: both roots survive all device work, renderer retirement, and
        // diagnostics. The workload drops its Renderers before returning.
        let replacement_native =
            unsafe { GlInstance::from_wgl_window(&replacement_window, [320, 180]) }?;
        let replacement_id = InstanceId::allocate()?;
        // SAFETY: the replacement instance exclusively owns its adopted context.
        let replacement_instance =
            unsafe { Instance::<Gl>::from_native(replacement_id, replacement_native) };
        let replacement_adapter = replacement_instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or("no WGL replacement adapter was enumerated")?;
        let replacement = block_on(replacement_instance.open(DeviceRequest {
            adapter: AdapterId {
                instance: replacement_id,
                ordinal: replacement_adapter.id.ordinal,
            },
            required_features: Vec::new(),
        }))?;
        if replacement.id() == device.id() {
            return Err("WGL S7 replacement unexpectedly retained the original DeviceId".into());
        }
        let mut replacement_observer = diagnostics::WglObserver::install(&replacement)?;
        let replacement_diagnostics =
            output.join("s7-deferred-device-isolation-replacement-diagnostics");
        fs::create_dir_all(&replacement_diagnostics)?;
        let result = s7_device_isolation::run(
            device.clone(),
            replacement.clone(),
            &output.join("s7-deferred-device-isolation"),
        );
        let replacement_diagnostic_result =
            replacement_observer.write(&replacement, &replacement_diagnostics);
        let old_diagnostic_result = observer.write(&device, output);
        if result.is_err()
            || replacement_diagnostic_result.is_err()
            || old_diagnostic_result.is_err()
        {
            std::mem::forget(replacement_window);
            std::mem::forget(window);
        }
        return result
            .and(replacement_diagnostic_result)
            .and(old_diagnostic_result);
    }
    if workload == Workload::S6Composed {
        let result = s6_composed::run(device.clone(), &output.join("s6-composed"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S6Surface {
        let result =
            s6_composed::run_surface_wgl(device.clone(), &window, &output.join("s6-surface"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S5Forward {
        let result = s5_forward::run(device.clone(), &output.join("s5-forward"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S5Surface {
        let result = s5_forward_surface::run_wgl(
            device.clone(),
            &window,
            &output.join("s5-forward-surface"),
        );
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    if workload == Workload::S5DeviceIsolation {
        // A separate WGL device requires its own HWND and native context. Keep
        // both roots alive while the old Renderer retains completed resident
        // products and the replacement Renderer consumes the same snapshots.
        let replacement_window = Window::new(
            WindowConfig::new("Fluxel S5 Forward replacement WGL", 320, 180)?
                .with_visibility(WindowVisibility::Hidden),
        )?;
        // SAFETY: both windows outlive their instances, devices, diagnostics,
        // and the completion-owned products created by this workload.
        let replacement_native =
            unsafe { GlInstance::from_wgl_window(&replacement_window, [320, 180]) }?;
        let replacement_id = InstanceId::allocate()?;
        // SAFETY: this instance exclusively owns the replacement native context.
        let replacement_instance =
            unsafe { Instance::<Gl>::from_native(replacement_id, replacement_native) };
        let replacement_adapter = replacement_instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or("no WGL replacement adapter was enumerated")?;
        let replacement = block_on(replacement_instance.open(DeviceRequest {
            adapter: AdapterId {
                instance: replacement_id,
                ordinal: replacement_adapter.id.ordinal,
            },
            required_features: Vec::new(),
        }))?;
        if replacement.id() == device.id() {
            return Err("WGL S5 replacement unexpectedly retained the original DeviceId".into());
        }
        let mut replacement_observer = diagnostics::WglObserver::install(&replacement)?;
        let replacement_diagnostics =
            output.join("s5-forward-device-isolation-replacement-diagnostics");
        fs::create_dir_all(&replacement_diagnostics)?;
        let result = s5_forward::run_device_isolation(
            device.clone(),
            replacement.clone(),
            &output.join("s5-forward-device-isolation"),
        );
        let replacement_diagnostic_result =
            replacement_observer.write(&replacement, &replacement_diagnostics);
        let old_diagnostic_result = observer.write(&device, output);
        if result.is_err()
            || replacement_diagnostic_result.is_err()
            || old_diagnostic_result.is_err()
        {
            std::mem::forget(replacement_window);
            std::mem::forget(window);
        }
        return result
            .and(replacement_diagnostic_result)
            .and(old_diagnostic_result);
    }
    if workload == Workload::S3Material {
        let result = s3_material::run(device.clone(), &output.join("s3-material"));
        let diagnostic_result = observer.write(&device, output);
        if result.is_err() || diagnostic_result.is_err() {
            std::mem::forget(window);
        }
        return result.and(diagnostic_result);
    }
    let mut old_submission = None;
    let mut s2_consumer = s2_shader::S2ShaderConsumer::new();
    let result = run_copy(device.clone(), &adapter_name, output)
        .map(|submission| old_submission = Some(submission))
        .and_then(|()| raster::run(device.clone(), &output.join("raster")))
        .and_then(|()| compute::run(device.clone(), &output.join("compute")))
        .and_then(|()| read_set::run(device.clone(), &output.join("read-set")))
        .and_then(|()| {
            s2_shader::run(
                &mut s2_consumer,
                device.clone(),
                &output.join("s2-shader"),
                false,
            )
        })
        .and_then(|()| {
            s2_shader::run(
                &mut s2_consumer,
                device.clone(),
                &output.join("s2-shader-same-device-cache-hit"),
                true,
            )
        })
        .and_then(|()| s2_texture::run(device.clone(), &output.join("s2-texture")))
        .and_then(|()| s3_material::run(device.clone(), &output.join("s3-material")))
        .and_then(|()| surface::run_wgl(device.clone(), &window, &output.join("surface")))
        .and_then(|()| {
            s5_forward_surface::run_wgl(device.clone(), &window, &output.join("s5-forward-surface"))
        });
    let diagnostic_result = observer.write(&device, output);
    if result.is_err() || diagnostic_result.is_err() {
        // Dispose cache-held reusable products while the original WGL host
        // root still owns their context. The invoked Renderer already owns
        // any submitted products until its own completion/error path ends.
        drop(s2_consumer);
        // An error can leave an indeterminate native owner retained by the
        // Renderer. The window is its WGL context's external lifetime root;
        // preserve it even if draining native diagnostics itself failed.
        std::mem::forget(window);
        return result.and(diagnostic_result);
    }

    let old_pipeline = raster::make_pipeline(&device, fluxel_rhi::api::TextureFormat::Rgba8Unorm)?;
    // A WGL replacement must own a separate HWND and native context. Reusing
    // the original context here would not exercise RHI device provenance.
    let replacement_window = Window::new(
        WindowConfig::new("Fluxel replacement WGL", 320, 180)?
            .with_visibility(WindowVisibility::Hidden),
    )?;
    // SAFETY: the replacement window and its independently-created context
    // remain alive until its replacement-device workload has completed.
    let replacement_native =
        unsafe { GlInstance::from_wgl_window(&replacement_window, [320, 180]) }?;
    let replacement_id = InstanceId::allocate()?;
    // SAFETY: this Core instance owns the newly adopted replacement context.
    let replacement_instance =
        unsafe { Instance::<Gl>::from_native(replacement_id, replacement_native) };
    let replacement_adapter = replacement_instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL replacement adapter was enumerated")?;
    let replacement_adapter_name = replacement_adapter.name.clone();
    let replacement = block_on(replacement_instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: replacement_id,
            ordinal: replacement_adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    if replacement.id() == device.id() {
        return Err("WGL replacement unexpectedly retained the original DeviceId".into());
    }
    let replacement_s2_output = output.join("s2-shader-replacement-device");
    fs::create_dir_all(&replacement_s2_output)?;
    write_capabilities(
        &replacement,
        "native-gl-wgl",
        &replacement_adapter_name,
        &replacement_s2_output,
    )?;
    let replacement_diagnostics = output.join("replacement-diagnostics");
    fs::create_dir_all(&replacement_diagnostics)?;
    let mut replacement_observer = diagnostics::WglObserver::install(&replacement)?;
    let replacement_result = s2_shader::run(
        &mut s2_consumer,
        replacement.clone(),
        &replacement_s2_output,
        false,
    )
    .and_then(|()| {
        verify_s2_cache_replacement(&mut s2_consumer, device.id(), replacement.id(), output)
    })
    .and_then(|()| {
        raster::replacement_rejects_old_pipeline(
            old_pipeline,
            replacement.clone(),
            old_submission
                .as_ref()
                .ok_or("old copy submission is missing")?,
            output,
        )
    });
    let replacement_diagnostic_result =
        replacement_observer.write(&replacement, &replacement_diagnostics);
    // Include old-context pipeline creation and product destruction used by
    // the provenance refusal, without discarding the earlier queue drain.
    let old_diagnostic_result = observer.write(&device, output);
    // Both WGL host roots are still live. Release only reusable cache entries
    // before either context can be destroyed; submitted owners were retained
    // by the local Renderers through readback completion.
    drop(s2_consumer);
    if replacement_result.is_err() || replacement_diagnostic_result.is_err() {
        // The replacement context is independently owned. Preserve this Host
        // root on a failed replacement submission rather than dropping it
        // before completion evidence exists.
        std::mem::forget(replacement_window);
    }
    if old_diagnostic_result.is_err() {
        std::mem::forget(window);
    }
    replacement_result?;
    replacement_diagnostic_result?;
    old_diagnostic_result
}
