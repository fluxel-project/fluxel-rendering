//! Tests the private native-boundary lowering, validation, synchronization, and fault hooks.

#[cfg(test)]
use super::*;

#[cfg(feature = "dx12")]
static PRESENTATION_HOLD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn fixed_surface_contract_uses_current_extent_and_rejects_missing_usage() {
    let mut capabilities = wgpu_hal::SurfaceCapabilities {
        formats: vec![wgt::SurfaceFormatCapabilities {
            format: wgt::TextureFormat::Rgba8Unorm,
            color_spaces: wgt::SurfaceColorSpaces::SRGB,
        }],
        maximum_frame_latency: 1..=3,
        current_extent: Some(wgt::Extent3d {
            width: 960,
            height: 540,
            depth_or_array_layers: 1,
        }),
        usage: wgt::TextureUses::COLOR_TARGET,
        present_modes: vec![wgt::PresentMode::Fifo],
        composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
    };
    let actual = fixed_surface_extent(&capabilities, 640, 360, "test").unwrap();
    assert_eq!((actual.width, actual.height), (960, 540));

    capabilities.usage = wgt::TextureUses::empty();
    assert!(fixed_surface_extent(&capabilities, 640, 360, "test").is_err());
}

#[cfg(feature = "dx12")]
#[test]
fn presentation_tickets_retire_independently_with_native_bundle_lifetime() {
    let tickets = NativePresentationTickets::new(3);
    let (first_acquire, first) = tickets.try_acquire().unwrap();
    drop(first_acquire);
    let (second_acquire, second) = tickets.try_acquire().unwrap();
    drop(second_acquire);
    // Independent acquired frames both prevent teardown; retiring one cannot
    // accidentally release the other.
    assert_eq!(tickets.live_count(), 2);
    drop(first);
    assert_eq!(tickets.live_count(), 1);
    assert!(tickets.any_live());
    drop(second);
    assert_eq!(tickets.live_count(), 0);

    let (unknown_acquire, unknown) = tickets.try_acquire().unwrap();
    // Accepted-unknown quarantine intentionally retains its own ticket.
    std::mem::forget((unknown_acquire, unknown));
    assert!(tickets.any_live());
}

#[cfg(feature = "dx12")]
#[test]
fn unknown_presentation_completion_drop_quarantines_its_ticket() {
    let tickets = NativePresentationTickets::new(1);
    let (acquire, lease) = tickets.try_acquire().expect("one ticket");
    drop(acquire);
    quarantine_presentation_lease(Some(lease));
    assert!(
        tickets.any_live(),
        "failure completion Drop must not make the presentable image reusable"
    );
    assert!(
        tickets.poisoned(),
        "accepted-unknown retirement must make the surface refuse later work"
    );
}

#[cfg(feature = "dx12")]
#[test]
fn presentation_completion_hold_is_fifo_and_drop_releases_it() {
    use std::sync::atomic::Ordering;

    let _serial = PRESENTATION_HOLD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear_dx12_presentation_completion_holds_for_test();
    let first = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let second = Arc::new(std::sync::atomic::AtomicBool::new(true));
    arm_next_dx12_presentation_completion_hold(&first);
    arm_next_dx12_presentation_completion_hold(&second);
    let consumed_first = take_next_dx12_presentation_completion_hold().expect("first hold");
    assert!(Arc::ptr_eq(&first, &consumed_first));
    first.store(false, Ordering::Release);
    assert!(!consumed_first.load(Ordering::Acquire));
    let consumed_second = take_next_dx12_presentation_completion_hold().expect("second hold");
    assert!(Arc::ptr_eq(&second, &consumed_second));
    clear_dx12_presentation_completion_holds_for_test();
}

#[cfg(feature = "dx12")]
#[test]
fn dropped_unconsumed_presentation_hold_is_not_taken_by_a_later_present() {
    let _serial = PRESENTATION_HOLD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear_dx12_presentation_completion_holds_for_test();
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(true));
    arm_next_dx12_presentation_completion_hold(&dropped);
    drop(dropped);
    assert!(
        take_next_dx12_presentation_completion_hold().is_none(),
        "a dropped handle must not latch a future presentation"
    );
    clear_dx12_presentation_completion_holds_for_test();
}

#[cfg(feature = "dx12")]
#[test]
fn presentation_hold_predicate_observes_drop_release_without_touching_other_work() {
    use std::sync::atomic::Ordering;

    let _serial = PRESENTATION_HOLD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear_dx12_presentation_completion_holds_for_test();
    let hold = Arc::new(std::sync::atomic::AtomicBool::new(true));
    assert!(presentation_completion_is_held(Some(&hold)));
    hold.store(false, Ordering::Release);
    assert!(!presentation_completion_is_held(Some(&hold)));
    assert!(!presentation_completion_is_held(None));

    // An armed hold remains queued until the presentation path explicitly
    // takes it; generic copy/upload completion observation never calls take.
    let queued = Arc::new(std::sync::atomic::AtomicBool::new(true));
    arm_next_dx12_presentation_completion_hold(&queued);
    assert!(
        !presentation_completion_is_held(None),
        "a non-presentation completion has no hold to observe"
    );
    assert!(Arc::ptr_eq(
        &queued,
        &take_next_dx12_presentation_completion_hold().expect("presentation consumes queued hold")
    ));
    clear_dx12_presentation_completion_holds_for_test();
}

#[cfg(feature = "dx12")]
#[test]
fn presentation_accepted_unknown_fault_is_one_shot() {
    let _serial = PRESENTATION_HOLD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    inject_dx12_presentation_accepted_unknown_once();
    assert!(take_dx12_presentation_accepted_unknown());
    assert!(!take_dx12_presentation_accepted_unknown());
}

#[cfg(test)]
#[test]
fn buffer_projection_exposes_storage_widening_exactly() {
    let read = BufferUsage::empty().with(BufferUsageKind::StorageRead);
    assert_eq!(buffer_usage_from_native(lower_buffer_usage(read)), read);

    let write = BufferUsage::empty().with(BufferUsageKind::StorageWrite);
    assert_eq!(
        buffer_usage_from_native(lower_buffer_usage(write)),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead, BufferUsageKind::StorageWrite])
    );
}

#[test]
fn texture_storage_projection_preserves_access_mode() {
    for usage in [
        TextureUsage::empty().with(TextureUsageKind::StorageRead),
        TextureUsage::empty().with(TextureUsageKind::StorageWrite),
        TextureUsage::from_kinds([
            TextureUsageKind::StorageRead,
            TextureUsageKind::StorageWrite,
        ]),
    ] {
        assert_eq!(
            texture_usage_from_native(lower_texture_usage(usage), TextureFormat::Rgba8Unorm),
            usage
        );
    }
}

#[test]
fn buffer_copy_alignment_is_checked_at_the_native_boundary() {
    let aligned = BufferCopyRegion {
        source_offset: 4,
        destination_offset: 8,
        size: 12,
    };
    assert!(validate_buffer_copy_alignment(aligned).is_ok());

    for region in [
        BufferCopyRegion {
            source_offset: 2,
            ..aligned
        },
        BufferCopyRegion {
            destination_offset: 2,
            ..aligned
        },
        BufferCopyRegion { size: 6, ..aligned },
    ] {
        assert!(validate_buffer_copy_alignment(region).is_err());
    }
}

#[test]
fn storage_binding_range_is_rechecked_at_the_native_boundary() {
    assert!(validate_native_compute_binding_range(0, 256, 256, 256, 256).is_ok());
    for (offset, size, buffer_size, alignment, maximum) in [
        (4, 256, 512, 256, 256),
        (0, 256, 512, 256, 252),
        (u64::MAX - 3, 4, u64::MAX, 4, u64::MAX),
    ] {
        assert!(
            validate_native_compute_binding_range(offset, size, buffer_size, alignment, maximum,)
                .is_err()
        );
    }
}

#[test]
fn queue_operation_lock_recovers_from_poison_without_losing_exclusion() {
    let lock = Mutex::new(());
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = lock.lock().expect("fresh queue lock");
        panic!("simulate a panicking queue caller");
    }));
    assert!(lock.is_poisoned());
    let _guard = lock_queue_operations(&lock);
    // The recovered guard is still held here, so a future caller remains
    // serialized even after an unrelated panic poisoned the mutex.
}

#[test]
fn linear_clamp_sampler_descriptor_is_closed_and_filtering() {
    let descriptor = linear_clamp_sampler_descriptor();
    assert_eq!(descriptor.address_modes, [wgt::AddressMode::ClampToEdge; 3]);
    assert_eq!(descriptor.min_filter, wgt::FilterMode::Linear);
    assert_eq!(descriptor.mag_filter, wgt::FilterMode::Linear);
    assert_eq!(descriptor.mipmap_filter, wgt::MipmapFilterMode::Nearest);
    assert_eq!(descriptor.lod_clamp, 0.0..32.0);
    assert_eq!(descriptor.compare, None);
    assert_eq!(descriptor.anisotropy_clamp, 1);
    assert_eq!(descriptor.border_color, None);
}

#[test]
fn srgb_format_lowering_is_distinct_from_unorm() {
    assert_eq!(
        lower_format(TextureFormat::Rgba8Unorm),
        wgt::TextureFormat::Rgba8Unorm
    );
    assert_eq!(
        lower_format(TextureFormat::Rgba8UnormSrgb),
        wgt::TextureFormat::Rgba8UnormSrgb
    );
    assert_ne!(
        lower_format(TextureFormat::Rgba8Unorm),
        lower_format(TextureFormat::Rgba8UnormSrgb)
    );
}

#[test]
fn srgb_sampled_usage_projection_does_not_depend_on_unorm_format() {
    let sampled = TextureUsage::empty().with(TextureUsageKind::Sampled);
    assert_eq!(
        texture_usage_from_native(lower_texture_usage(sampled), TextureFormat::Rgba8UnormSrgb,),
        sampled
    );
}

#[test]
fn submit_fault_countdown_consumes_exactly_the_requested_attempt() {
    inject_submit_rejected_after(2);
    assert_eq!(take_submit_fault(), 0);
    assert_eq!(take_submit_fault(), 0);
    assert_eq!(take_submit_fault(), 1);
    assert_eq!(take_submit_fault(), 0);

    inject_submit_accepted_unknown_after(0);
    assert_eq!(take_submit_fault(), 2);
    assert_eq!(take_submit_fault(), 0);
}
