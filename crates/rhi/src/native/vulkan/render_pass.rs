//! The raster pass the draw path begins: the portable attachment set lowered onto
//! the `Vulkan` render pass a draw records into.
//!
//! # Why the pass has to exist at all
//!
//! The instance this backend opens requests `VK_API_VERSION_1_0` (step 1) and enables
//! no dynamic-rendering extension, so `vkCmdBeginRenderPass` needs a `VkRenderPass`
//! and a `VkFramebuffer`. `pipeline::create_raster` already had to build the first of
//! those for creation (step 5), and this module is the description the *recording*
//! pass is built from. The two are compared by `Vulkan`'s render-pass compatibility
//! rules, which is why [`color_description`] is the one place a colour attachment's
//! description is written: the creation pass and the recording pass both call it, so
//! a fact they must agree on cannot drift between them.
//!
//! # The preserved semantic that lives here
//!
//! Plan section 4: **a pass admits exactly one colour attachment at index zero and
//! no depth-stencil attachment.** That is the GL family's `ONE_COLOUR_TARGET`
//! refusal (`webgl2/compat/device/pass.rs`), preserved here rather than relaxed,
//! because a fresh backend is tempted to be permissive and a pass whose attachment
//! set has no pipeline that could run in it is a caller mistake rather than a
//! feature. The pipeline vocabulary can still *express* a depth-stencil attachment
//! (step 5 lowers one), and that is deliberate: the refusal is visible at the pass,
//! where no retained recipe declares depth, rather than hidden by a vocabulary that
//! could not say it at all.
//!
//! # Pure, and nothing owned
//!
//! [`admit`], [`describe`] and [`framebuffer_extent`] create nothing and reach no
//! driver entry point, so every refusal is provable without a device. The owning half
//! is [`super::framebuffer`]: it creates the `VkRenderPass` from [`describe`]'s
//! attachment, builds the `VkFramebuffer` from the attachment's view and the extent
//! [`framebuffer_extent`] decides, and owns both. Nothing here owns a handle.
//!
//! # Why the create-info is not returned
//!
//! `VkRenderPassCreateInfo` borrows its attachment and subpass slices, and the
//! `VkSubpassDescription` it holds borrows the colour-reference slice in turn, so a
//! value that contained them could not outlive the locals holding those slices.
//! [`PassAttachment`] therefore returns the pieces -- a description, a reference and a
//! clear value -- and the owning half assembles the create-info in the one scope
//! where the borrows are valid. A self-referential type would be the same borrow with
//! an unsafe promise attached.

use ash::vk;
use fluxel_rendergraph::{
    AttachmentOps, LoadOp, RasterColorAttachment, RasterDepthStencilAttachment, StoreOp, TextureDesc,
    TextureDimension,
};

use crate::common::base::resource::TextureId;

/// Why a pass this backend cannot run is refused.
///
/// Two variants rather than one because the two mistakes need different fixes, and
/// the GL family keeps them apart for the same reason: an attachment set with no
/// colour target at index zero has no pipeline that could run in it, while a
/// depth-stencil attachment names a recipe no retained artifact declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PassError {
    /// The pass does not declare exactly one colour attachment at index zero.
    ColorTargets,
    /// The pass declares a depth-stencil attachment.
    DepthStencil,
}

/// The one colour attachment an admitted pass writes.
///
/// Returns the attachment rather than merely admitting the set, so the caller's next
/// step -- resolving the texture's image view and format -- has no second place to
/// decide what "one attachment at index zero" means and no index to trust. The
/// refusals are the three facts this backend's pass vocabulary has no case for: zero
/// colour attachments, more than one, one that is not at index zero, and any
/// depth-stencil attachment at all.
///
/// The subresource `range` is deliberately not decided here: it selects the image
/// view the owning half builds, not a field of the render pass, so this function
/// answers the pass question and nothing else.
pub(crate) fn admit<'a>(
    colors: &'a [RasterColorAttachment<'a, TextureId>],
    depth_stencil: Option<&RasterDepthStencilAttachment<'_, TextureId>>,
) -> Result<&'a RasterColorAttachment<'a, TextureId>, PassError> {
    let [attachment] = colors else {
        return Err(PassError::ColorTargets);
    };
    if attachment.index != 0 {
        return Err(PassError::ColorTargets);
    }
    if depth_stencil.is_some() {
        return Err(PassError::DepthStencil);
    }
    Ok(attachment)
}

/// One colour attachment's `Vulkan` description.
///
/// This is the shared half of the creation and recording render passes. The fields
/// `Vulkan` compares two passes by -- the format and the sample count -- are stated
/// once here, and the contents operations are parameters because the two passes state
/// different ones: the creation pass performs nothing and says `DONT_CARE`, while the
/// recording pass states the operations the graph compiled.
///
/// Both layouts are `COLOR_ATTACHMENT_OPTIMAL` because that is the layout
/// `barrier::image_state` gives `ColorAttachmentWrite` (step 7), which is the
/// transition a graph records around the pass. A pass whose initial layout differed
/// from where the barrier left the image would be rejected by the driver rather than
/// silently reordered, but stating the pair here means there is no second place to
/// disagree with the barrier.
pub(crate) fn color_description(
    format: vk::Format,
    samples: vk::SampleCountFlags,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription::default()
        .format(format)
        .samples(samples)
        .load_op(load)
        .store_op(store)
        // Ignored for a colour format, and written anyway so the field is never left
        // at whatever `ash` happens to default to.
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
}

/// The `Vulkan` facts one admitted colour attachment lowers to.
///
/// The clear payload is carried as the `[f32; 4]` the portable `LoadOp::Clear`
/// states rather than as a `vk::ClearValue`, because that union has no `Debug` and is
/// built where the begin-info needs it ([`Self::clear_value`]). The payload is read
/// by `Vulkan` only when the load operation is `CLEAR`; for the other two it is
/// written as zeroes, because `VkRenderPassBeginInfo` needs one entry per clear
/// operation and a value the driver ignores should not be an uninitialized one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PassAttachment {
    /// The `Vulkan` format of the attachment's image view.
    pub(crate) format: vk::Format,
    /// The sample count the image was created with.
    pub(crate) samples: vk::SampleCountFlags,
    /// The load operation the graph compiled.
    pub(crate) load: vk::AttachmentLoadOp,
    /// The store operation the graph compiled.
    pub(crate) store: vk::AttachmentStoreOp,
    /// The clear colour, meaningful only where `load` is `CLEAR`.
    pub(crate) clear: [f32; 4],
}

impl PassAttachment {
    /// The attachment description for slot zero.
    pub(crate) fn description(&self) -> vk::AttachmentDescription {
        color_description(self.format, self.samples, self.load, self.store)
    }

    /// The subpass colour reference naming slot zero.
    pub(crate) fn color_reference(&self) -> vk::AttachmentReference {
        vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    }

    /// The clear value `VkRenderPassBeginInfo` carries for slot zero.
    ///
    /// One entry is always supplied even when the load operation is not `CLEAR`, which
    /// the specification permits: the array is indexed by attachment and a value for
    /// an attachment that does not clear is ignored.
    pub(crate) fn clear_value(&self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: self.clear,
            },
        }
    }
}

/// Lowers one admitted colour attachment onto the facts the recording pass needs.
///
/// `format` and `samples` are the resolved facts of the attachment's texture, because
/// the portable descriptor names only a texture id: the resource table owns the
/// format and the sample count, exactly as it owns the image view the framebuffer is
/// built from.
///
/// Both matches are exhaustive over this crate's closed vocabulary, which is the
/// opposite shape from a mapping over a `#[non_exhaustive]` portable enum: a
/// `LoadOp` or `StoreOp` variant added later must be taught to this lowering rather
/// than silently taking a wildcard's value.
pub(crate) fn describe(
    attachment: &RasterColorAttachment<'_, TextureId>,
    format: vk::Format,
    samples: vk::SampleCountFlags,
) -> PassAttachment {
    let operations: AttachmentOps<[f32; 4]> = attachment.operations;
    let (load, clear) = match operations.load {
        LoadOp::Load => (vk::AttachmentLoadOp::LOAD, [0.0; 4]),
        LoadOp::Clear(value) => (vk::AttachmentLoadOp::CLEAR, value),
        LoadOp::DontCare => (vk::AttachmentLoadOp::DONT_CARE, [0.0; 4]),
    };
    let store = match operations.store {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
    };
    PassAttachment {
        format,
        samples,
        load,
        store,
        clear,
    }
}

/// The framebuffer size one admitted colour attachment's texture describes.
///
/// Returns `None` where the description is not one this backend builds a framebuffer
/// from. The refusals are `Vulkan`'s own rules for a framebuffer attachment rather
/// than a preference of this layer:
///
/// - a view of anything but a two-dimensional image at exactly one mip level and one
///   layer is not an attachment of a render pass this backend can begin. A
///   three-dimensional view is not a legal framebuffer attachment at all, and a
///   layered or multi-mip view names subresources a non-multiview pass neither covers
///   nor compares a pipeline against. That is exactly the target shape the borrowed
///   native raster path preserves (`D2`, one mip, one layer, one depth slice), stated
///   once here rather than repeated in the owning half;
/// - a zero extent is not an image `Vulkan` accepts.
///
/// The dimension is compared rather than matched because [`TextureDimension`] is a
/// foreign, `#[non_exhaustive]` enum: a variant added upstream cannot be lowered to a
/// guessed attachment shape, and this function answers `None` for it.
pub(crate) fn framebuffer_extent(desc: &TextureDesc) -> Option<vk::Extent2D> {
    if desc.dimension != TextureDimension::D2 {
        return None;
    }
    if desc.extent.width == 0 || desc.extent.height == 0 || desc.extent.depth != 1 {
        return None;
    }
    if desc.mip_levels.max(1) != 1 || desc.array_layers.max(1) != 1 {
        return None;
    }
    Some(vk::Extent2D {
        width: desc.extent.width,
        height: desc.extent.height,
    })
}

#[cfg(test)]
mod tests {
    use fluxel_rendergraph::{
        DeviceIdentity, Extent3d, PhysicalResourceIdentity, TextureFormat, TextureRange,
        WriteCoverage,
    };

    use super::*;
    use crate::common::base::stamp::DeviceStamp;

    fn texture() -> TextureId {
        TextureId::new(
            DeviceStamp::initial(DeviceIdentity::new(1)),
            PhysicalResourceIdentity::new(1),
        )
    }

    fn ops(load: LoadOp<[f32; 4]>, store: StoreOp) -> AttachmentOps<[f32; 4]> {
        AttachmentOps {
            load,
            store,
            write_coverage: WriteCoverage::Full,
        }
    }

    fn color<'a>(
        index: u32,
        operations: AttachmentOps<[f32; 4]>,
        texture: &'a TextureId,
    ) -> RasterColorAttachment<'a, TextureId> {
        RasterColorAttachment {
            index,
            texture,
            range: TextureRange::Whole,
            operations,
        }
    }

    fn depth<'a>(texture: &'a TextureId) -> RasterDepthStencilAttachment<'a, TextureId> {
        RasterDepthStencilAttachment {
            texture,
            range: TextureRange::Whole,
            depth: None,
            stencil: None,
        }
    }

    fn plain() -> LoadOp<[f32; 4]> {
        LoadOp::Load
    }

    #[test]
    fn the_retained_one_colour_pass_is_admitted() {
        let texture = texture();
        let colors = [color(0, ops(plain(), StoreOp::Store), &texture)];
        let admitted = admit(&colors, None).expect("one colour at index zero is the retained set");
        assert_eq!(admitted.index, 0);
    }

    #[test]
    fn no_colour_target_is_refused() {
        assert_eq!(admit(&[], None).err(), Some(PassError::ColorTargets));
    }

    #[test]
    fn a_second_colour_target_is_refused() {
        let texture = texture();
        let colors = [
            color(0, ops(plain(), StoreOp::Store), &texture),
            color(1, ops(plain(), StoreOp::Store), &texture),
        ];
        assert_eq!(admit(&colors, None).err(), Some(PassError::ColorTargets));
    }

    #[test]
    fn a_colour_target_not_at_index_zero_is_refused() {
        let texture = texture();
        let colors = [color(1, ops(plain(), StoreOp::Store), &texture)];
        assert_eq!(admit(&colors, None).err(), Some(PassError::ColorTargets));
    }

    #[test]
    fn a_depth_stencil_attachment_is_refused() {
        let texture = texture();
        let colors = [color(0, ops(plain(), StoreOp::Store), &texture)];
        let depth_stencil = depth(&texture);
        assert_eq!(
            admit(&colors, Some(&depth_stencil)).err(),
            Some(PassError::DepthStencil)
        );
    }

    #[test]
    fn the_two_refusals_are_distinct_sentences() {
        assert_ne!(PassError::ColorTargets, PassError::DepthStencil);
    }

    #[test]
    fn load_operations_lower_to_their_named_values() {
        let texture = texture();
        let load = color(0, ops(plain(), StoreOp::Store), &texture);
        assert_eq!(
            describe(&load, vk::Format::R8G8B8A8_UNORM, vk::SampleCountFlags::TYPE_1).load,
            vk::AttachmentLoadOp::LOAD
        );

        let clear_value = [0.25, 0.5, 0.75, 1.0];
        let clear = color(
            0,
            ops(LoadOp::Clear(clear_value), StoreOp::Store),
            &texture,
        );
        let lowered = describe(&clear, vk::Format::R8G8B8A8_UNORM, vk::SampleCountFlags::TYPE_1);
        assert_eq!(lowered.load, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(lowered.clear, clear_value);

        let dont_care = color(0, ops(LoadOp::DontCare, StoreOp::Store), &texture);
        assert_eq!(
            describe(
                &dont_care,
                vk::Format::R8G8B8A8_UNORM,
                vk::SampleCountFlags::TYPE_1
            )
            .load,
            vk::AttachmentLoadOp::DONT_CARE
        );
    }

    #[test]
    fn store_operations_lower_to_their_named_values() {
        let texture = texture();
        let store = color(0, ops(plain(), StoreOp::Store), &texture);
        assert_eq!(
            describe(&store, vk::Format::R8G8B8A8_UNORM, vk::SampleCountFlags::TYPE_1).store,
            vk::AttachmentStoreOp::STORE
        );

        let discard = color(0, ops(plain(), StoreOp::Discard), &texture);
        assert_eq!(
            describe(
                &discard,
                vk::Format::R8G8B8A8_UNORM,
                vk::SampleCountFlags::TYPE_1
            )
            .store,
            vk::AttachmentStoreOp::DONT_CARE
        );
    }

    #[test]
    fn a_load_that_is_not_a_clear_carries_no_clear_payload() {
        let texture = texture();
        for load in [LoadOp::Load, LoadOp::DontCare] {
            let attachment = color(0, ops(load, StoreOp::Store), &texture);
            let lowered = describe(
                &attachment,
                vk::Format::R8G8B8A8_UNORM,
                vk::SampleCountFlags::TYPE_1,
            );
            assert_eq!(lowered.clear, [0.0; 4], "{load:?} clears nothing");
        }
    }

    #[test]
    fn the_description_pins_every_field_the_graph_does_not_state() {
        let texture = texture();
        let attachment = color(0, ops(plain(), StoreOp::Store), &texture);
        let lowered = describe(
            &attachment,
            vk::Format::B8G8R8A8_UNORM,
            vk::SampleCountFlags::TYPE_4,
        );
        let described = lowered.description();
        assert_eq!(described.format, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(described.samples, vk::SampleCountFlags::TYPE_4);
        assert_eq!(described.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(described.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(described.stencil_load_op, vk::AttachmentLoadOp::DONT_CARE);
        assert_eq!(
            described.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(
            described.initial_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            described.final_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn the_colour_reference_names_slot_zero() {
        let texture = texture();
        let attachment = color(0, ops(plain(), StoreOp::Store), &texture);
        let lowered = describe(
            &attachment,
            vk::Format::R8G8B8A8_UNORM,
            vk::SampleCountFlags::TYPE_1,
        );
        let reference = lowered.color_reference();
        assert_eq!(reference.attachment, 0);
        assert_eq!(reference.layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn the_clear_value_carries_the_lowered_colour() {
        let texture = texture();
        let attachment = color(
            0,
            ops(LoadOp::Clear([0.1, 0.2, 0.3, 0.4]), StoreOp::Store),
            &texture,
        );
        let lowered = describe(
            &attachment,
            vk::Format::R8G8B8A8_UNORM,
            vk::SampleCountFlags::TYPE_1,
        );
        let value = lowered.clear_value();
        // SAFETY: `clear_value` writes the `color` member, so reading the same member
        // is the initialized variant of the union.
        let floats = unsafe { value.color.float32 };
        assert_eq!(floats, [0.1, 0.2, 0.3, 0.4]);
    }

    fn texture_desc() -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        }
    }

    #[test]
    fn the_retained_attachment_shape_has_a_framebuffer_extent() {
        assert_eq!(
            framebuffer_extent(&texture_desc()),
            Some(vk::Extent2D {
                width: 16,
                height: 8,
            })
        );
    }

    #[test]
    fn the_shapes_a_framebuffer_cannot_attach_are_refused() {
        // Every one of these is a view `Vulkan` either forbids as a framebuffer
        // attachment or that a render pass without multiview cannot cover, and each
        // is the shape the borrowed native raster path refuses as well.
        let layered = TextureDesc {
            array_layers: 4,
            ..texture_desc()
        };
        assert_eq!(framebuffer_extent(&layered), None, "a layered view");

        let volumetric = TextureDesc {
            dimension: TextureDimension::D3,
            ..texture_desc()
        };
        assert_eq!(
            framebuffer_extent(&volumetric),
            None,
            "a three-dimensional view is not a framebuffer attachment"
        );

        let one_dimensional = TextureDesc {
            dimension: TextureDimension::D1,
            ..texture_desc()
        };
        assert_eq!(framebuffer_extent(&one_dimensional), None, "a 1D view");

        let mipmapped = TextureDesc {
            mip_levels: 3,
            ..texture_desc()
        };
        assert_eq!(framebuffer_extent(&mipmapped), None, "a multi-mip view");

        let deep = TextureDesc {
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 2,
            },
            ..texture_desc()
        };
        assert_eq!(framebuffer_extent(&deep), None, "a depth axis that is not one");

        let empty = TextureDesc {
            extent: Extent3d {
                width: 0,
                height: 8,
                depth: 1,
            },
            ..texture_desc()
        };
        assert_eq!(framebuffer_extent(&empty), None, "a zero-sized image");
    }
}
