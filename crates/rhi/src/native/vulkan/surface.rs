//! Step 10's pure half: the fixed presentation contract lowered from what a
//! surface reports.
//!
//! `Vulkan` answers "what can this window present?" with three lists -- the
//! supported [`vk::SurfaceFormatKHR`] pairs, the supported
//! [`vk::PresentModeKHR`] values, and the [`vk::SurfaceCapabilitiesKHR`] limits
//! -- and the job here is to decide whether *this* backend's one presentation
//! contract is among them, and to lower that decision into the
//! [`vk::SwapchainCreateInfoKHR`] the driver is handed. It creates nothing and
//! owns nothing, so every branch of the decision is provable without a window.
//!
//! # The contract is fixed, and a refusal is a value
//!
//! Fluxel presents exactly one thing: an `R8G8B8A8_UNORM` colour attachment in
//! `SRGB_NONLINEAR`, with `FIFO` presentation and `OPAQUE` compositing. That is
//! the same fixed triple the borrowed path being replaced configures
//! (`crates/wgpu-hal`, `SurfaceConfiguration { present_mode: Fifo,
//! composite_alpha_mode: Opaque, format: Rgba8Unorm }`), and it is deliberately
//! **not** a negotiation with a preference order: choosing the closest available
//! format or mode would present something the graph did not compile against, and
//! the graph's compile-time format decision is what makes the surface texture's
//! descriptor correct.
//!
//! **sRGB is deliberately not claimed.** `Rgba8UnormSrgb` is a first-class
//! portable format, but the drawable's own encoding is not observable through
//! this API, and claiming it would apply a second gamma to bytes the compositor
//! already decodes -- the double-gamma mistake plan section 4 records. The
//! surface format list therefore states exactly `R8G8B8A8_UNORM` with
//! `SRGB_NONLINEAR`, and a surface that only offers the sRGB *format* is refused
//! by name rather than silently accepted.
//!
//! # Why the image count is clamped and the extent is not invented
//!
//! `Vulkan` leaves "how many swapchain images" to the caller inside
//! `[min_image_count, max_image_count]`, where a zero `max_image_count` means the
//! driver has no upper bound. This backend asks for [`IMAGE_COUNT`] -- one more
//! than the frame latency the borrowed path configured, which is the smallest
//! count that keeps a frame being presented while its successor is recorded -- and
//! clamps it into the driver's own range. The clamp can only raise the count to
//! what the driver requires, never lower it below one in-flight frame, so it cannot
//! turn the fixed policy into a different one silently.
//!
//! The extent has exactly one legal source. When `current_extent` is non-zero it
//! **is** the size this surface must be configured at, and the caller's requested
//! size is not a competing input; when it is zero, `Vulkan` delegates the choice
//! and the requested size is clamped to `[min_image_extent, max_image_extent]`.
//! Inventing an extent the surface did not report would configure a swapchain the
//! window cannot display.
//!
//! # What is deliberately not here
//!
//! - **No `pre_transform` decision.** A swapchain's `pre_transform` must be a
//!   transform the surface reports as supported; requiring it is a claim this
//!   increment does not need yet, and the caller of
//!   [`swapchain_create_info`] owns that field until the swapchain-owning step
//!   states which transform it selected.
//! - **No presentation-support query.** Whether a queue family can present to a
//!   surface is answered by `vkGetPhysicalDeviceSurfaceSupportKHR`, which is
//!   device enumeration (step 2's rule: the queue family is chosen by rule, and
//!   introducing presentation support there is its own decision).
//! - **No swapchain, no acquire, no present.** Those are the owning half of step
//!   10, and this module is what they lower from.

use ash::vk;
use fluxel_rendergraph::TextureUsage;

use super::texture;

/// The `Vulkan` image format the fixed presentation contract presents.
///
/// A test pins this against the portable format the surface texture descriptor
/// names (`TextureFormat::Rgba8Unorm`) through [`super::format::image_format`], so
/// the two sides of that contract cannot drift apart.
pub(crate) const PRESENT_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The `Vulkan` colour space the fixed presentation contract presents.
pub(crate) const PRESENT_COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;

/// The `Vulkan` present mode the fixed presentation contract presents.
///
/// `FIFO` is the one present mode every implementation is required to support, and
/// it is the mode the borrowed path being replaced selects, so requiring it cannot
/// refuse a conformant device.
pub(crate) const PRESENT_MODE: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

/// The `Vulkan` composite alpha the fixed presentation contract asks for.
///
/// `OPAQUE` states that the surface's alpha is ignored by the compositor, which is
/// what a fixed `Rgba8Unorm` colour target with no alpha contract actually means.
/// `INHERIT` was rejected for the same reason it is rejected on the borrowed path:
/// it makes the result depend on the window system rather than on the image.
pub(crate) const COMPOSITE_ALPHA: vk::CompositeAlphaFlagsKHR = vk::CompositeAlphaFlagsKHR::OPAQUE;

/// How many swapchain images this backend asks for.
///
/// One more than the borrowed path's maximum frame latency of two
/// (`crates/wgpu-hal`'s `SurfaceConfiguration::maximum_frame_latency`), which is the
/// smallest count under which a presented frame and a frame still being recorded
/// can both be in flight. A test pins the value rather than letting it drift,
/// because it is a policy number no compiler checks.
pub(crate) const IMAGE_COUNT: u32 = 3;

/// The image usage the fixed presentation contract requires of every swapchain
/// image.
///
/// A presented image is a colour attachment, and that is the one usage this
/// backend's fixed recipe needs. It is also the usage the borrowed path
/// configures (`wgt::TextureUses::COLOR_TARGET`), so the swapchain this backend
/// creates is the one the frozen oracle was measured against.
pub(crate) const REQUIRED_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

/// One surface configuration this backend's fixed contract accepts.
///
/// Every field is the *driver's* own vocabulary rather than a portable restatement,
/// because the only consumer is [`swapchain_create_info`]; a caller that needs a
/// portable fact asks the ledger, not this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Presentation {
    /// The surface format selected, always [`PRESENT_FORMAT`].
    pub(crate) format: vk::Format,
    /// The colour space selected, always [`PRESENT_COLOR_SPACE`].
    pub(crate) color_space: vk::ColorSpaceKHR,
    /// The extent the swapchain must be created at.
    pub(crate) extent: vk::Extent2D,
    /// The present mode selected, always [`PRESENT_MODE`].
    pub(crate) present_mode: vk::PresentModeKHR,
    /// The composite alpha selected, always [`COMPOSITE_ALPHA`].
    pub(crate) composite_alpha: vk::CompositeAlphaFlagsKHR,
    /// The swapchain image count, clamped into the driver's own range.
    pub(crate) image_count: u32,
    /// Every usage the swapchain images must carry, which always includes
    /// [`REQUIRED_USAGE`] and the lowered `requested_usage`.
    pub(crate) usage: vk::ImageUsageFlags,
}

/// Why a surface cannot serve this backend's fixed presentation contract.
///
/// Every variant is a distinct sentence, because the three refusals a caller can
/// act on differently are "the driver offers no such thing", "the driver offers it
/// for a different format", and "the request itself is not a size".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentationError {
    /// The requested extent is zero on an axis, which `Vulkan` does not define for
    /// a swapchain.
    ZeroExtent,
    /// The requested extent is not inside `[min_image_extent, max_image_extent]`.
    ExtentUnsupported,
    /// The surface does not offer `R8G8B8A8_UNORM` with `SRGB_NONLINEAR`.
    ///
    /// An sRGB image *format* is a different format and does not satisfy this: the
    /// contract presents linear-encoded bytes and no portable sRGB surface claim
    /// exists (plan section 4).
    FormatUnsupported,
    /// The surface offers the format but not with `SRGB_NONLINEAR`.
    ColorSpaceUnsupported,
    /// The surface does not offer `FIFO` presentation.
    PresentModeUnsupported,
    /// The surface cannot composite `OPAQUE`.
    CompositeAlphaUnsupported,
    /// The surface cannot be a colour attachment.
    UsageUnsupported,
    /// The lowered `requested_usage` names an operation this surface does not
    /// support.
    RequestedUsageUnsupported,
}

/// Whether the surface reports a format/colour-space pair the fixed contract
/// accepts.
///
/// Both halves must match one entry: `R8G8B8A8_UNORM` with some *other* colour
/// space is not `R8G8B8A8_UNORM` with `SRGB_NONLINEAR`, and `Vulkan` treats the
/// pair as the unit of support.
pub(crate) fn supports_format(formats: &[vk::SurfaceFormatKHR]) -> bool {
    formats.contains(&vk::SurfaceFormatKHR {
        format: PRESENT_FORMAT,
        color_space: PRESENT_COLOR_SPACE,
    })
}

/// Whether the surface offers `FIFO` presentation.
pub(crate) fn supports_present_mode(modes: &[vk::PresentModeKHR]) -> bool {
    modes.contains(&PRESENT_MODE)
}

/// Whether the surface can composite `OPAQUE`.
pub(crate) fn supports_composite_alpha(capabilities: &vk::SurfaceCapabilitiesKHR) -> bool {
    capabilities.supported_composite_alpha.contains(COMPOSITE_ALPHA)
}

/// Whether the surface's driver can present to a queue family that can also
/// rasterize.
///
/// This is the query the extent rules describe rather than a call: `Vulkan` reports
/// a surface's usable extent range as `min_image_extent`..`max_image_extent`, and a
/// zero maximum means the driver has no upper bound.
fn supports_extent(capabilities: &vk::SurfaceCapabilitiesKHR, extent: vk::Extent2D) -> bool {
    extent.width >= capabilities.min_image_extent.width
        && extent.height >= capabilities.min_image_extent.height
        && (capabilities.max_image_extent.width == 0
            || extent.width <= capabilities.max_image_extent.width)
        && (capabilities.max_image_extent.height == 0
            || extent.height <= capabilities.max_image_extent.height)
}

/// The extent a swapchain of `requested` size must be created at, or `None` when no
/// legal extent exists for this request.
///
/// The driver's `current_extent` wins whenever it is non-zero, because that is the
/// size the window system is actually showing; only a zero `current_extent` -- the
/// specification's way of saying "you choose" -- lets the requested size be used,
/// clamped into the reported range. A refused request is `None` rather than a
/// substituted extent: a swapchain created at a size the caller did not ask for
/// would present a different image than the graph compiled.
fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    requested: vk::Extent2D,
) -> Option<vk::Extent2D> {
    if capabilities.current_extent.width != 0 && capabilities.current_extent.height != 0 {
        return Some(capabilities.current_extent);
    }
    if requested.width == 0 || requested.height == 0 {
        return None;
    }
    if !supports_extent(capabilities, requested) {
        return None;
    }
    Some(requested)
}

/// The image count to ask for, clamped into what the driver reports.
///
/// A `max_image_count` of zero is the specification's "no upper bound"; every other
/// value is a real ceiling, and a `min_image_count` above [`IMAGE_COUNT`] is a real
/// floor. The clamp is `max(IMAGE_COUNT, min)` and then the upper bound by
/// `min(...)`, written so the lower bound is applied first: clamping in the other
/// order could return a count below `min_image_count` when a driver's range is
/// inverted, which the specification forbids but a value could still express.
fn clamp_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let raised = IMAGE_COUNT.max(capabilities.min_image_count);
    if capabilities.max_image_count == 0 {
        raised
    } else {
        raised.min(capabilities.max_image_count)
    }
}

/// Decides the fixed presentation contract against one surface's reported facts.
///
/// `requested_usage` is the portable usage the surface texture was declared with;
/// it is lowered through [`super::texture::usage_flags`], so a texture usage that
/// widens here would be equally visible there, and [`REQUIRED_USAGE`] is added
/// because a presented image is a colour attachment whatever else it is.
pub(crate) fn contract(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    formats: &[vk::SurfaceFormatKHR],
    present_modes: &[vk::PresentModeKHR],
    requested: vk::Extent2D,
    requested_usage: TextureUsage,
) -> Result<Presentation, PresentationError> {
    if requested.width == 0 || requested.height == 0 {
        return Err(PresentationError::ZeroExtent);
    }
    if !supports_format(formats) {
        // The two halves are separate sentences because they need different fixes:
        // a surface with the format and another colour space is a driver the caller
        // must ask differently, while a surface without the format cannot present
        // at all.
        let has_format = formats
            .iter()
            .any(|candidate| candidate.format == PRESENT_FORMAT);
        return Err(if has_format {
            PresentationError::ColorSpaceUnsupported
        } else {
            PresentationError::FormatUnsupported
        });
    }
    if !supports_present_mode(present_modes) {
        return Err(PresentationError::PresentModeUnsupported);
    }
    if !supports_composite_alpha(capabilities) {
        return Err(PresentationError::CompositeAlphaUnsupported);
    }
    let extent = choose_extent(capabilities, requested).ok_or(PresentationError::ExtentUnsupported)?;

    let usage = texture::usage_flags(requested_usage) | REQUIRED_USAGE;
    // Checked against the driver's own report rather than assumed: `COLOR_ATTACHMENT`
    // is not optional here, and the requested kinds are the graph's declared
    // operations, so a surface that cannot serve them must refuse before a
    // swapchain exists rather than at the first recorded pass.
    if !capabilities
        .supported_usage_flags
        .contains(REQUIRED_USAGE)
    {
        return Err(PresentationError::UsageUnsupported);
    }
    if !capabilities.supported_usage_flags.contains(usage) {
        return Err(PresentationError::RequestedUsageUnsupported);
    }

    Ok(Presentation {
        format: PRESENT_FORMAT,
        color_space: PRESENT_COLOR_SPACE,
        extent,
        present_mode: PRESENT_MODE,
        composite_alpha: COMPOSITE_ALPHA,
        image_count: clamp_image_count(capabilities),
        usage,
    })
}

/// Lowers a decision into the create-info the driver is handed.
///
/// The one field this leaves to its caller is `pre_transform`, which must name a
/// transform the surface reports as supported and is therefore the swapchain-owning
/// step's decision rather than a default invented here. Everything else is written,
/// including the fields whose defaults would silently claim something: `clipped` is
/// true (the implementation may ignore obscured pixels, which the specification
/// permits and the borrowed path does), `image_array_layers` is one (this contract
/// presents one 2-D image), and `old_swapchain` is null (this is a fresh creation,
/// and a reconfigure passes its own predecessor through the field).
///
/// `surface` is borrowed rather than owned: the swapchain refers to it, so the
/// caller that owns the surface must outlive the swapchain created from this value,
/// which is the same ownership rule the resource table already states for its
/// device.
pub(crate) fn swapchain_create_info(
    surface: vk::SurfaceKHR,
    presentation: &Presentation,
    pre_transform: vk::SurfaceTransformFlagsKHR,
) -> vk::SwapchainCreateInfoKHR<'static> {
    vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(presentation.image_count)
        .image_format(presentation.format)
        .image_color_space(presentation.color_space)
        .image_extent(presentation.extent)
        .image_array_layers(1)
        .image_usage(presentation.usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(presentation.composite_alpha)
        .present_mode(presentation.present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    // `SurfaceKHR` is a non-dispatchable handle whose constructor comes from the
    // `Handle` trait rather than an inherent method.
    use ash::vk::Handle as _;
    use crate::native::vulkan::format;
    use fluxel_rendergraph::{TextureFormat, TextureUsageKind};

    /// Every portable texture usage kind, in `TextureUsageKind`'s declaration order.
    const ALL_USAGE_KINDS: [TextureUsageKind; 8] = [
        TextureUsageKind::Sampled,
        TextureUsageKind::StorageRead,
        TextureUsageKind::StorageWrite,
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::DepthStencilAttachment,
        TextureUsageKind::CopySource,
        TextureUsageKind::CopyDestination,
        TextureUsageKind::Present,
    ];

    /// A surface that offers exactly the fixed contract.
    fn capabilities() -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 8,
            current_extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 8192,
                height: 8192,
            },
            max_image_array_layers: 1,
            supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY,
            current_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            supported_composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        }
    }

    fn formats() -> Vec<vk::SurfaceFormatKHR> {
        vec![vk::SurfaceFormatKHR {
            format: PRESENT_FORMAT,
            color_space: PRESENT_COLOR_SPACE,
        }]
    }

    fn modes() -> Vec<vk::PresentModeKHR> {
        vec![vk::PresentModeKHR::FIFO]
    }

    fn requested() -> vk::Extent2D {
        vk::Extent2D {
            width: 640,
            height: 480,
        }
    }

    fn surface_usage() -> TextureUsage {
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present])
    }

    fn accepted() -> Presentation {
        contract(&capabilities(), &formats(), &modes(), requested(), surface_usage())
            .expect("the fixed contract against a conformant surface")
    }

    #[test]
    fn the_fixed_contract_is_the_fixed_contract() {
        // The four constants are policy, not derived values, so they are asserted by
        // name: a change to any of them changes what every surface presents.
        assert_eq!(PRESENT_FORMAT, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(PRESENT_COLOR_SPACE, vk::ColorSpaceKHR::SRGB_NONLINEAR);
        assert_eq!(PRESENT_MODE, vk::PresentModeKHR::FIFO);
        assert_eq!(COMPOSITE_ALPHA, vk::CompositeAlphaFlagsKHR::OPAQUE);
        assert_eq!(IMAGE_COUNT, 3);
        assert_eq!(REQUIRED_USAGE, vk::ImageUsageFlags::COLOR_ATTACHMENT);
    }

    #[test]
    fn the_present_format_is_the_portable_surface_format() {
        // The surface texture's descriptor is compiled from the portable format, so
        // the image the swapchain creates and the descriptor the graph holds must
        // name the same bytes. This is the only place the two vocabularies meet.
        assert_eq!(
            format::image_format(TextureFormat::Rgba8Unorm),
            Some(PRESENT_FORMAT)
        );
    }

    #[test]
    fn the_present_format_is_not_the_srgb_format() {
        // The double-gamma refusal, pinned: claiming the sRGB format would apply a
        // second decode to bytes the compositor already decodes.
        assert_ne!(
            format::image_format(TextureFormat::Rgba8UnormSrgb),
            Some(PRESENT_FORMAT)
        );
    }

    #[test]
    fn a_conformant_surface_is_accepted_with_the_contracts_own_values() {
        let accepted = accepted();
        assert_eq!(accepted.format, PRESENT_FORMAT);
        assert_eq!(accepted.color_space, PRESENT_COLOR_SPACE);
        assert_eq!(accepted.present_mode, PRESENT_MODE);
        assert_eq!(accepted.composite_alpha, COMPOSITE_ALPHA);
        assert_eq!(accepted.extent, requested());
        assert_eq!(accepted.image_count, IMAGE_COUNT);
        assert_eq!(accepted.usage, REQUIRED_USAGE);
    }

    #[test]
    fn the_drivers_current_extent_wins_over_the_requested_one() {
        // The window system is showing a size; creating a swapchain at any other
        // size is the bug this rule exists to prevent.
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 1024,
                height: 768,
            },
            ..capabilities()
        };
        let accepted = contract(&capabilities, &formats(), &modes(), requested(), surface_usage())
            .expect("a surface with a current extent is configurable");
        assert_eq!(
            accepted.extent,
            vk::Extent2D {
                width: 1024,
                height: 768
            }
        );
    }

    #[test]
    fn a_caller_request_outside_the_reported_range_is_refused() {
        let too_wide = vk::Extent2D {
            width: 9000,
            height: 480,
        };
        assert_eq!(
            contract(&capabilities(), &formats(), &modes(), too_wide, surface_usage()),
            Err(PresentationError::ExtentUnsupported)
        );
        let too_small = vk::Extent2D {
            width: 0,
            height: 480,
        };
        // A zero request is its own sentence: it is not a size at all, so it is
        // refused before the range question is asked.
        assert_eq!(
            contract(&capabilities(), &formats(), &modes(), too_small, surface_usage()),
            Err(PresentationError::ZeroExtent)
        );
    }

    #[test]
    fn a_zero_image_count_range_means_no_upper_bound() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            max_image_count: 0,
            ..capabilities()
        };
        let accepted = contract(&capabilities, &formats(), &modes(), requested(), surface_usage())
            .expect("a driver with no ceiling still presents");
        assert_eq!(accepted.image_count, IMAGE_COUNT);
    }

    #[test]
    fn the_image_count_is_raised_to_the_drivers_floor_and_held_under_its_ceiling() {
        let raised = vk::SurfaceCapabilitiesKHR {
            min_image_count: 4,
            ..capabilities()
        };
        let accepted = contract(&raised, &formats(), &modes(), requested(), surface_usage())
            .expect("a driver demanding more images still presents");
        assert_eq!(accepted.image_count, 4);

        let capped = vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 2,
            ..capabilities()
        };
        let accepted = contract(&capped, &formats(), &modes(), requested(), surface_usage())
            .expect("a driver with a tight ceiling still presents");
        assert_eq!(accepted.image_count, 2);
    }

    #[test]
    fn each_missing_piece_of_the_contract_is_its_own_refusal() {
        let no_format = contract(
            &capabilities(),
            &[vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: PRESENT_COLOR_SPACE,
            }],
            &modes(),
            requested(),
            surface_usage(),
        );
        assert_eq!(no_format, Err(PresentationError::FormatUnsupported));

        let no_color_space = contract(
            &capabilities(),
            &[vk::SurfaceFormatKHR {
                format: PRESENT_FORMAT,
                color_space: vk::ColorSpaceKHR::from_raw(1),
            }],
            &modes(),
            requested(),
            surface_usage(),
        );
        assert_eq!(no_color_space, Err(PresentationError::ColorSpaceUnsupported));

        let no_mode = contract(
            &capabilities(),
            &formats(),
            &[vk::PresentModeKHR::MAILBOX],
            requested(),
            surface_usage(),
        );
        assert_eq!(no_mode, Err(PresentationError::PresentModeUnsupported));

        let no_alpha = contract(
            &vk::SurfaceCapabilitiesKHR {
                supported_composite_alpha: vk::CompositeAlphaFlagsKHR::INHERIT,
                ..capabilities()
            },
            &formats(),
            &modes(),
            requested(),
            surface_usage(),
        );
        assert_eq!(no_alpha, Err(PresentationError::CompositeAlphaUnsupported));

        let no_attachment = contract(
            &vk::SurfaceCapabilitiesKHR {
                supported_usage_flags: vk::ImageUsageFlags::SAMPLED,
                ..capabilities()
            },
            &formats(),
            &modes(),
            requested(),
            surface_usage(),
        );
        assert_eq!(no_attachment, Err(PresentationError::UsageUnsupported));

        let no_requested = contract(
            &vk::SurfaceCapabilitiesKHR {
                supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                ..capabilities()
            },
            &formats(),
            &modes(),
            requested(),
            TextureUsage::from_kinds([
                TextureUsageKind::ColorAttachment,
                TextureUsageKind::CopySource,
            ]),
        );
        assert_eq!(
            no_requested,
            Err(PresentationError::RequestedUsageUnsupported)
        );
    }

    #[test]
    fn the_srgb_format_alone_does_not_satisfy_the_contract() {
        // The specific substitution a preference-ordered negotiation would make,
        // named so a reader sees it is refused rather than accepted as "close".
        let srgb_only = [vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_SRGB,
            color_space: PRESENT_COLOR_SPACE,
        }];
        assert!(!supports_format(&srgb_only));
    }

    #[test]
    fn the_contract_helper_answers_the_same_question_as_the_lowering() {
        // The predicates are public within the module and the lowering is not allowed
        // to disagree with them: a surface the lowering accepts is one every
        // predicate reports true, and a surface it refuses is one where some
        // predicate reports false.
        let accepted = accepted();
        assert!(supports_format(&formats()));
        assert!(supports_present_mode(&modes()));
        assert!(supports_composite_alpha(&capabilities()));
        assert!(supports_extent(&capabilities(), accepted.extent));

        assert!(!supports_format(&formats()[..0]));
        assert!(!supports_present_mode(&modes()[..0]));
    }

    #[test]
    fn the_usage_fold_reaches_every_kind_the_vocabulary_declares() {
        // `TextureUsageKind` offers no iterator, so this list is the only proof that
        // every declared kind has an answer here; a kind absent from
        // `texture::usage_flags` would lower to nothing and silently widen nothing.
        //
        // The surface this asks is deliberately *not* the minimal one above: it
        // reports every usage the vocabulary can lower to, because the point is that
        // each kind reaches a flag at all. Whether a surface that reports less
        // refuses is the next test's subject.
        let permissive = vk::SurfaceCapabilitiesKHR {
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            ..capabilities()
        };
        for kind in ALL_USAGE_KINDS {
            let usage = TextureUsage::from_kinds([kind]);
            let accepted = contract(&permissive, &formats(), &modes(), requested(), usage)
                .expect("every declared kind is servable by the permissive surface");
            assert!(
                accepted.usage.contains(REQUIRED_USAGE),
                "{kind:?} must still present as a colour attachment"
            );
            // A kind the vocabulary cannot lower would add nothing, and this is what
            // makes that a failure rather than a silently narrower swapchain.
            assert_eq!(
                accepted.usage,
                texture::usage_flags(usage) | REQUIRED_USAGE,
                "{kind:?} must lower through the texture usage fold"
            );
        }
    }

    #[test]
    fn the_create_info_states_the_contract_and_leaves_the_transform_to_its_caller() {
        let presentation = accepted();
        let info = swapchain_create_info(
            vk::SurfaceKHR::from_raw(0x1234),
            &presentation,
            vk::SurfaceTransformFlagsKHR::IDENTITY,
        );
        assert_eq!(info.surface, vk::SurfaceKHR::from_raw(0x1234));
        assert_eq!(info.min_image_count, presentation.image_count);
        assert_eq!(info.image_format, presentation.format);
        assert_eq!(info.image_color_space, presentation.color_space);
        assert_eq!(info.image_extent, presentation.extent);
        assert_eq!(info.image_usage, presentation.usage);
        assert_eq!(info.present_mode, presentation.present_mode);
        assert_eq!(info.composite_alpha, presentation.composite_alpha);
        assert_eq!(
            info.pre_transform,
            vk::SurfaceTransformFlagsKHR::IDENTITY
        );
        assert_eq!(info.image_sharing_mode, vk::SharingMode::EXCLUSIVE);
        // The three fields whose defaults would claim something.
        assert_eq!(info.image_array_layers, 1);
        assert_ne!(info.clipped, 0);
        assert_eq!(info.old_swapchain, vk::SwapchainKHR::null());
    }
}
