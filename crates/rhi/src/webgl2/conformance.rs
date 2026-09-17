//! Opening one real DesktopGl4 context and reporting what it turned out to be.
//!
//! This module exists because of a hole rather than a plan.  The GL-family
//! providers had no reachable entry point: [`WglContextSurface`] and its `open`
//! were named nowhere outside their own module, no test in this repository had
//! ever created a WGL context, and [`crate::Backend`] has no GL-family variant
//! -- so the matrix cell for desktop GL had no vehicle at all, and a release
//! gate that asks for real-hardware evidence could not collect any.  This is
//! the vehicle: the smallest thing that opens the context through the same
//! constructor a real adapter would use and reports what the driver answered.
//!
//! # What it is, and what it deliberately is not
//!
//! It is a **scoped** entry.  The context is opened, observed, and dropped
//! inside one call, and the borrowed provider stack that a frame would need
//! never escapes it -- which is the whole reason this can exist before the
//! question "should a caller be able to open a GL-family device?" is answered.
//! That question is a 0.16 ownership question ([`NativeGlProvider`] borrows its
//! context, so a `'static` public device needs an ownership rework), and nothing
//! here answers it: there is no public device, no `Backend` variant, and no
//! path from this module to one.
//!
//! It is not a second discovery implementation.  Everything reported comes from
//! the snapshot [`WglContextSurface::open`] already gathered and validated --
//! the same snapshot the executor reads -- so a report cannot disagree with the
//! context the adapter would go on to use.
//!
//! # The window is the caller's
//!
//! This module never creates a window, owns a message pump, or chooses a
//! drawable size.  It takes a raw handle from whatever produced one -- the
//! standard trait every window provider implements, including the ecosystem's
//! own host -- and the extent the caller says that drawable has.  That is the
//! narrow protocol the boundary is defined by, and it is why the hardware gate
//! lives outside this crate while the scenario that needs private identities
//! lives inside it.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::api::{
    ContextEpoch, ContextStamp, DeviceIdentity, GlCapability, GlDiscoverySnapshot, GlLimits,
    GlSurfaceFacts, OwnerThreadIdentity, WglContextSurface,
};

/// What one real desktop GL context answered when it was asked.
///
/// Every field is a projection of the snapshot the context opened with, and the
/// projection is deliberately lossless where the ledger asks for a fact and
/// deliberately string-shaped where a field exists to be read by an
/// out-of-workspace gate rather than by two Rust types sharing a contract.
///
/// The identity strings are reported raw and unparsed: a driver is entitled to
/// spell its own version, and normalizing it here would replace evidence with
/// this crate's opinion of it.  The `(name, value)` pairs are the same idea one
/// level down -- the names are the source table's own field names, taken from
/// the field rather than retyped, so a row cannot drift from the thing it
/// describes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesktopGl4ContextReport {
    /// The normalized profile the context was accepted as.
    pub profile: String,
    /// The driver's own version string.
    pub version: String,
    /// The driver's own shading-language version string.
    pub shading_language_version: String,
    /// The vendor string.
    pub vendor: String,
    /// The renderer string.
    pub renderer: String,
    /// Driver or browser identity, as the platform reported it.
    pub driver_or_browser: String,
    /// Whether the context was created with a debug flag.
    pub debug: bool,
    /// Whether the context was created forward-compatible.
    pub forward_compatible: bool,
    /// Whether the context advertises robust access.
    pub robust_access: bool,
    /// Whether the context was created with error reporting suppressed.
    pub no_error: bool,
    /// Flags the driver reported that this crate does not classify.
    pub other_flags: Vec<String>,
    /// How many runtime extension spellings the driver listed.
    pub reported_extension_count: usize,
    /// Those spellings, verbatim.
    pub reported_extensions: Vec<String>,
    /// Typed evidence for the extensions this crate models: registry spelling
    /// against how far the acquisition actually got.
    pub typed_extensions: Vec<(String, String)>,
    /// Each capability this crate resolves, against whether the evidence
    /// enabled it.
    pub capabilities: Vec<(String, bool)>,
    /// The numerical facts, as `(field name, value)`.
    pub limits: Vec<(String, String)>,
    /// What was observed about the drawable behind this context.
    pub surface_facts: String,
    /// The extent this context was opened for.
    pub drawable_extent: [u32; 2],
    /// The context owner's thread identity.
    pub owner_thread: String,
}

/// Opens a real desktop GL context over `host`'s drawable and reports it.
///
/// The context is a desktop OpenGL 4.3 core context made current on the calling
/// thread, and it is dropped before this returns.  `identity` is the nonzero
/// device identity the caller assigns this context; it is allocation metadata
/// rather than a driver fact, so the caller owns it, and a zero is refused
/// rather than replaced with a default.
///
/// # Errors
///
/// The `Err` is a rendered description of the platform's own failure.  It is a
/// `String` rather than the crate's error type on purpose: the typed errors
/// here are crate-private, and a caller outside the crate has no business
/// matching on them -- what a hardware gate needs from a failure is the exact
/// message, which is what it gets.
pub fn observe_desktop_gl4_context<H>(
    host: &H,
    extent: [u32; 2],
    identity: u64,
) -> Result<DesktopGl4ContextReport, String>
where
    H: HasWindowHandle + HasDisplayHandle,
{
    let device = DeviceIdentity::new(identity)
        .ok_or_else(|| "a device identity has to be nonzero".to_owned())?;
    let stamp = ContextStamp::new(device, ContextEpoch::INITIAL);
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
        .map_err(|error| format!("the context opened but discovery refused it: {error:?}"))?;

    Ok(report(snapshot, context.extent(), &context.owner_thread()))
}

/// Projects one validated snapshot into the report.
///
/// Reads only; nothing here can alter the context or its evidence.
pub(crate) fn report(
    snapshot: &GlDiscoverySnapshot,
    drawable_extent: [u32; 2],
    owner_thread: &OwnerThreadIdentity,
) -> DesktopGl4ContextReport {
    let context = snapshot.context();
    let flags = context.flags();
    let extensions = snapshot.extensions();

    DesktopGl4ContextReport {
        profile: format!("{:?}", context.profile()),
        version: context.version().to_owned(),
        shading_language_version: context.shading_language_version().to_owned(),
        vendor: context.vendor().to_owned(),
        renderer: context.renderer().to_owned(),
        driver_or_browser: context.driver_or_browser().to_owned(),
        debug: flags.debug,
        forward_compatible: flags.forward_compatible,
        robust_access: flags.robust_access,
        no_error: flags.no_error,
        other_flags: flags.other.iter().cloned().collect(),
        reported_extension_count: extensions.raw_reported_names().count(),
        reported_extensions: extensions.raw_reported_names().map(str::to_owned).collect(),
        typed_extensions: typed_extensions(snapshot),
        capabilities: capabilities(snapshot),
        limits: limits(snapshot.limits()),
        surface_facts: surface_facts(snapshot.surface_facts()),
        drawable_extent,
        owner_thread: format!("{owner_thread:?}"),
    }
}

/// The typed extension ledger, as registry spelling against evidence state.
///
/// The spelling comes from the extension's own registry entry rather than from
/// a literal here, so this list cannot name something the registry does not.
fn typed_extensions(snapshot: &GlDiscoverySnapshot) -> Vec<(String, String)> {
    use super::api::{ExtensionProvenance, GlKnownExtension};

    const TYPED: [GlKnownExtension; 21] = [
        GlKnownExtension::ArbComputeShader,
        GlKnownExtension::ArbShaderStorageBufferObject,
        GlKnownExtension::ArbShaderImageLoadStore,
        GlKnownExtension::ExtDisjointTimerQueryWebgl2,
        GlKnownExtension::ExtColorBufferFloat,
        GlKnownExtension::ExtFloatBlend,
        GlKnownExtension::OesTextureFloatLinear,
        GlKnownExtension::ExtTextureFilterAnisotropic,
        GlKnownExtension::WebglMultiDraw,
        GlKnownExtension::WebglMultiDrawInstancedBaseVertexBaseInstance,
        GlKnownExtension::OvrMultiview2,
        GlKnownExtension::WebglShaderPixelLocalStorage,
        GlKnownExtension::KhrParallelShaderCompile,
        GlKnownExtension::KhrRobustness,
        GlKnownExtension::KhrDebug,
        GlKnownExtension::CompressedTextureS3tc,
        GlKnownExtension::CompressedTextureS3tcSrgb,
        GlKnownExtension::CompressedTextureBptc,
        GlKnownExtension::CompressedTextureRgtc,
        GlKnownExtension::CompressedTextureAstc,
        GlKnownExtension::CompressedTextureEtc,
    ];

    let extensions = snapshot.extensions();
    TYPED
        .into_iter()
        .filter_map(|extension| {
            let state = match extensions.provenance(extension)? {
                ExtensionProvenance::Reported => "reported",
                ExtensionProvenance::Acquired => "acquired",
                ExtensionProvenance::Probed => "probed",
                ExtensionProvenance::Failed => "failed",
            };
            Some((extension.raw_name().to_owned(), state.to_owned()))
        })
        .collect()
}

/// Every capability this crate resolves, against whether it is enabled.
///
/// The names are the common vocabulary's own, which is also why they are
/// spelled here rather than derived: a capability is a Fluxel concept, not a
/// driver string, so there is nothing upstream to take the spelling from.
fn capabilities(snapshot: &GlDiscoverySnapshot) -> Vec<(String, bool)> {
    const ALL: [(GlCapability, &str); 9] = [
        (GlCapability::Compute, "compute"),
        (GlCapability::StorageBuffer, "storage-buffer"),
        (GlCapability::StorageImage, "storage-image"),
        (GlCapability::IndirectDraw, "indirect-draw"),
        (GlCapability::IndirectDispatch, "indirect-dispatch"),
        (GlCapability::MultiDrawIndirect, "multi-draw-indirect"),
        (GlCapability::MultiDraw, "multi-draw"),
        (GlCapability::Multiview, "multiview"),
        (GlCapability::TimerQuery, "timer-query"),
    ];

    let resolved = snapshot.capabilities();
    ALL.into_iter()
        .map(|(capability, name)| (name.to_owned(), resolved.supports(capability)))
        .collect()
}

/// The numerical facts, as `(field name, value)` rows.
///
/// The row names come from `stringify!` on the field selector itself, which is
/// the point: the alternative was retyping forty names beside forty fields, and
/// a hand-typed name is a second spelling of a fact that can drift from the
/// first without anything failing.  Values are rendered rather than typed
/// because four of these are arrays and two are optional -- one rendering for
/// all of them keeps the report a record instead of a second, contradictory
/// model of the same table.
///
/// `max_texture_anisotropy` is the single exception and is pushed after the
/// macro for that reason.  The crate keeps a queried anisotropy as exact
/// IEEE-754 bits, so the `{:?}` every other row uses prints the *pattern*: the
/// first real run of this entry reported `Some(GlFiniteF32(1098907648))` where
/// a gate needs `Some(16)`.  Retaining the bits is right for the type and
/// publishing them is wrong as evidence, so this one row is rendered as its
/// value and sits last to keep the exception visible rather than buried.
fn limits(limits: GlLimits) -> Vec<(String, String)> {
    macro_rules! rows {
        ($($field:ident),+ $(,)?) => {{
            let mut rows = Vec::new();
            $(
                rows.push((
                    stringify!($field).to_owned(),
                    format!("{:?}", limits.$field),
                ));
            )+
            rows
        }};
    }

    let mut rows = rows!(
        max_texture_size,
        max_3d_texture_size,
        max_array_texture_layers,
        max_cube_map_texture_size,
        max_renderbuffer_size,
        max_color_attachments,
        max_draw_buffers,
        max_vertex_attributes,
        max_viewport_dimensions,
        max_viewports,
        max_vertex_texture_image_units,
        max_fragment_texture_image_units,
        max_combined_texture_image_units,
        max_uniform_buffer_bindings,
        max_uniform_block_size,
        uniform_buffer_offset_alignment,
        max_vertex_uniform_blocks,
        max_fragment_uniform_blocks,
        max_compute_uniform_blocks,
        max_combined_uniform_blocks,
        max_storage_buffer_bindings,
        max_storage_block_size,
        storage_buffer_offset_alignment,
        max_vertex_storage_blocks,
        max_fragment_storage_blocks,
        max_compute_storage_blocks,
        max_combined_storage_blocks,
        max_image_units,
        max_combined_image_units,
        max_samples,
        max_color_texture_samples,
        max_depth_texture_samples,
        max_integer_samples,
        max_compute_work_group_count,
        max_compute_work_group_size,
        max_compute_work_group_invocations,
        max_multiview_view_count,
        max_multi_draw_indirect_count,
        query_counter_bits,
    );

    rows.push((
        "max_texture_anisotropy".to_owned(),
        match limits.max_texture_anisotropy {
            Some(value) => format!("Some({})", value.get()),
            None => "None".to_owned(),
        },
    ));
    rows
}

/// What was observed about the drawable, as one phrase.
///
/// Worth a row of its own because "the context is fine" and "the drawable was
/// never described" are different facts, and a gate that cannot see the second
/// one would read a context with no surface as a fully resolved profile.
fn surface_facts(facts: GlSurfaceFacts) -> String {
    match facts {
        GlSurfaceFacts::Unavailable => "unavailable".to_owned(),
        other => format!("{other:?}"),
    }
}
