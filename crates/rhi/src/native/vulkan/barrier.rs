//! Step 7's pure half: RenderGraph's access states lowered onto the Vulkan
//! synchronization facts one `vkCmdPipelineBarrier` needs.
//!
//! # What is lowered, and what is not
//!
//! The graph states an access in **semantic** terms -- "the fragment stage samples
//! this texture", "this buffer is a copy destination" -- and Vulkan wants four
//! different spellings of the same fact: a source stage mask, a source access mask,
//! a destination stage mask, a destination access mask, and, for an image, the
//! layout each side needs. This module is the one place that translation happens,
//! so the recorder in [`super::command`] never spells a mask itself.
//!
//! # Why every mapping returns `Option`
//!
//! [`ResourceAccessState`] is `#[non_exhaustive]`, and it is one enum shared by
//! buffers and textures. Two consequences follow, and both are answered with a
//! value rather than a wildcard:
//!
//! - a state this backend has not been taught is refused, not lowered to a guessed
//!   stage or layout, because a guessed barrier is an ordering bug that no compiler
//!   and no driver reports;
//! - a state that names the other resource kind is refused: `VertexRead` is not a
//!   texture access and `ColorAttachmentWrite` is not a buffer access, so asking
//!   for one on the wrong kind is a caller mistake. `None` is that refusal.
//!
//! # The one preserved semantic that lives here
//!
//! **`before == after` is still a barrier.** Plan section 4 and the
//! [`ExecutionBackend`] contract both say a same-state transition is a memory
//! dependency rather than a discarded no-op: a read after a write of the *same*
//! state still needs the write made visible. Nothing in this module compares
//! `before` and `after`; both sides are lowered and a barrier is built, so the
//! equality shortcut cannot be added without deleting a function. A test pins the
//! equal-state shape directly.
//!
//! [`ExecutionBackend`]: fluxel_rendergraph::ExecutionBackend

use ash::vk;
use fluxel_rendergraph::{BufferRange, ResourceAccessState, TextureAspect, TextureRange};

/// The stages one access state's work executes in, and the accesses that must be
/// ordered around it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AccessScope {
    /// Pipeline stages the access executes in.
    pub stage: vk::PipelineStageFlags,
    /// Accesses that must be made available or visible.
    pub access: vk::AccessFlags,
}

/// What one texture access state requires: its scope and the layout it needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ImageState {
    /// The stage/access scope of the texture access.
    pub scope: AccessScope,
    /// The image layout that access requires.
    pub layout: vk::ImageLayout,
}

/// The stages a shader access can occur in.
///
/// All three rather than the one that happens to sample in today's recipes: the
/// semantic state does not say which stage asked, so narrowing it here would be a
/// second, unwritten fact. The borrowed Vulkan backend being replaced maps its
/// `RESOURCE` use to the same three stages, which is why this is not a new claim.
fn shader_stages() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::COMPUTE_SHADER
}

/// The two fragment-test stages a depth-stencil attachment access occurs in.
fn fragment_test_stages() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
}

/// Lowers one access state onto the scope a buffer barrier needs.
///
/// `None` means the state is either one this backend has not been taught or one
/// that names a texture access. Both are refused by the caller before the driver
/// is reached.
pub(crate) fn buffer_scope(state: ResourceAccessState) -> Option<AccessScope> {
    let scope = match state {
        // No prior access is known or required, so nothing must be made available.
        // `TOP_OF_PIPE` rather than an empty mask: Vulkan requires a non-zero stage
        // mask, and the empty access mask is what "nothing" actually is.
        ResourceAccessState::Undefined => AccessScope {
            stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            access: vk::AccessFlags::empty(),
        },
        ResourceAccessState::UniformRead => AccessScope {
            stage: shader_stages(),
            access: vk::AccessFlags::UNIFORM_READ,
        },
        ResourceAccessState::ShaderStorageRead => AccessScope {
            stage: shader_stages(),
            access: vk::AccessFlags::SHADER_READ,
        },
        ResourceAccessState::ShaderStorageWrite => AccessScope {
            stage: shader_stages(),
            access: vk::AccessFlags::SHADER_WRITE,
        },
        ResourceAccessState::ShaderStorageReadWrite => AccessScope {
            stage: shader_stages(),
            access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        },
        ResourceAccessState::VertexRead => AccessScope {
            stage: vk::PipelineStageFlags::VERTEX_INPUT,
            access: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        },
        ResourceAccessState::IndexRead => AccessScope {
            stage: vk::PipelineStageFlags::VERTEX_INPUT,
            access: vk::AccessFlags::INDEX_READ,
        },
        ResourceAccessState::IndirectRead => AccessScope {
            stage: vk::PipelineStageFlags::DRAW_INDIRECT,
            access: vk::AccessFlags::INDIRECT_COMMAND_READ,
        },
        ResourceAccessState::CopySource => AccessScope {
            stage: vk::PipelineStageFlags::TRANSFER,
            access: vk::AccessFlags::TRANSFER_READ,
        },
        ResourceAccessState::CopyDestination => AccessScope {
            stage: vk::PipelineStageFlags::TRANSFER,
            access: vk::AccessFlags::TRANSFER_WRITE,
        },
        // Attachment, sampled and present states name a texture, and a state this
        // backend has not been taught is not lowered to a guess.
        _ => return None,
    };
    Some(scope)
}

/// Lowers one access state onto the scope and layout a texture barrier needs.
///
/// `is_depth` is asked of the *mapped* `Vulkan` format
/// ([`super::format::is_depth`]), so the sampled-read layout -- the one case whose
/// answer depends on the format -- keeps a single source of truth.
pub(crate) fn image_state(state: ResourceAccessState, is_depth: bool) -> Option<ImageState> {
    let (scope, layout) = match state {
        ResourceAccessState::Undefined => (
            AccessScope {
                stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                access: vk::AccessFlags::empty(),
            },
            vk::ImageLayout::UNDEFINED,
        ),
        ResourceAccessState::ColorAttachmentRead => (
            AccessScope {
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags::COLOR_ATTACHMENT_READ,
            },
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccessState::ColorAttachmentWrite => (
            AccessScope {
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            },
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccessState::ColorAttachmentReadWrite => (
            AccessScope {
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            },
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccessState::DepthStencilRead => (
            AccessScope {
                stage: fragment_test_stages(),
                access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            },
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        ResourceAccessState::DepthStencilWrite => (
            AccessScope {
                stage: fragment_test_stages(),
                access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            },
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccessState::DepthStencilReadWrite => (
            AccessScope {
                stage: fragment_test_stages(),
                access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            },
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        ResourceAccessState::ShaderSampledRead => (
            AccessScope {
                stage: shader_stages(),
                access: vk::AccessFlags::SHADER_READ,
            },
            // A depth texture cannot be sampled through the colour read-only layout;
            // the borrowed path makes the same split, so this is not a new claim.
            if is_depth {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            },
        ),
        // Storage access has no optimal layout: read and write forms share `GENERAL`
        // because the graph may hand the same subresource either one next.
        ResourceAccessState::ShaderStorageRead => (
            AccessScope {
                stage: shader_stages(),
                access: vk::AccessFlags::SHADER_READ,
            },
            vk::ImageLayout::GENERAL,
        ),
        ResourceAccessState::ShaderStorageWrite => (
            AccessScope {
                stage: shader_stages(),
                access: vk::AccessFlags::SHADER_WRITE,
            },
            vk::ImageLayout::GENERAL,
        ),
        ResourceAccessState::ShaderStorageReadWrite => (
            AccessScope {
                stage: shader_stages(),
                access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            },
            vk::ImageLayout::GENERAL,
        ),
        ResourceAccessState::CopySource => (
            AccessScope {
                stage: vk::PipelineStageFlags::TRANSFER,
                access: vk::AccessFlags::TRANSFER_READ,
            },
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        ResourceAccessState::CopyDestination => (
            AccessScope {
                stage: vk::PipelineStageFlags::TRANSFER,
                access: vk::AccessFlags::TRANSFER_WRITE,
            },
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        // The portable present state is the swapchain's own layout, and the
        // borrowed path orders it with no stage and no access on either side. The
        // acquire that produces it belongs to step 10.
        ResourceAccessState::Present => (
            AccessScope {
                stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                access: vk::AccessFlags::empty(),
            },
            vk::ImageLayout::PRESENT_SRC_KHR,
        ),
        // Buffer-only states and states added upstream: refused, not guessed.
        _ => return None,
    };
    Some(ImageState { scope, layout })
}

/// Lowers one portable texture range onto the subresource range a barrier names.
///
/// `is_depth` resolves the one range that does not name an aspect itself:
/// [`TextureRange::Whole`] and [`TextureAspect::All`] both mean "every aspect this
/// format has", and Vulkan spells colour and depth aspects differently. A range
/// with a zero mip or layer count is refused: Vulkan requires both to be non-zero,
/// and the portable `Subresources` variant documents them as non-empty, so a zero
/// arriving here is a value the caller got wrong rather than something to send on.
pub(crate) fn image_subresource_range(
    range: TextureRange,
    is_depth: bool,
) -> Option<vk::ImageSubresourceRange> {
    let whole_aspect = if is_depth {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let subresource = match range {
        TextureRange::Whole => vk::ImageSubresourceRange {
            aspect_mask: whole_aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        },
        TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        } => {
            if mip_level_count == 0 || array_layer_count == 0 {
                return None;
            }
            let aspect_mask = match aspect {
                TextureAspect::Color => vk::ImageAspectFlags::COLOR,
                TextureAspect::Depth => vk::ImageAspectFlags::DEPTH,
                TextureAspect::Stencil => vk::ImageAspectFlags::STENCIL,
                TextureAspect::All => whole_aspect,
                // An aspect added upstream is refused rather than defaulted to one
                // of the aspects this backend knows.
                _ => return None,
            };
            vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level,
                level_count: mip_level_count,
                base_array_layer,
                layer_count: array_layer_count,
            }
        }
    };
    Some(subresource)
}

/// Builds the buffer barrier one transition records.
///
/// Both queue family indices are `QUEUE_FAMILY_IGNORED`, which is the correct
/// spelling for a barrier inside one queue family: this backend records and submits
/// on logical queue 0 only, so there is no ownership transfer to express. Stating
/// the ignored value rather than a zero is what makes that a fact about one queue
/// rather than about family zero.
///
/// The barrier carries the declared range; a whole-buffer range is `WHOLE_SIZE`
/// rather than a byte count the caller would have to compute.
pub(crate) fn buffer_barrier(
    buffer: vk::Buffer,
    range: BufferRange,
    before: AccessScope,
    after: AccessScope,
) -> vk::BufferMemoryBarrier<'static> {
    let (offset, size) = match range {
        BufferRange::Whole => (0, vk::WHOLE_SIZE),
        BufferRange::Bytes { offset, size } => (offset, size),
    };
    vk::BufferMemoryBarrier::default()
        .src_access_mask(before.access)
        .dst_access_mask(after.access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)
}

/// Builds the image barrier one transition records.
///
/// `old_layout` and `new_layout` are separate inputs, so an equal-state transition
/// produces a barrier with the layout unchanged on both sides. That is still a
/// memory dependency, and it is deliberately not a shape this function can turn
/// into "no barrier".
pub(crate) fn image_barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    before: ImageState,
    after: ImageState,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(before.scope.access)
        .dst_access_mask(after.scope.access)
        .old_layout(before.layout)
        .new_layout(after.layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every state that names a buffer access.
    const BUFFER_STATES: [ResourceAccessState; 10] = [
        ResourceAccessState::Undefined,
        ResourceAccessState::UniformRead,
        ResourceAccessState::ShaderStorageRead,
        ResourceAccessState::ShaderStorageWrite,
        ResourceAccessState::ShaderStorageReadWrite,
        ResourceAccessState::VertexRead,
        ResourceAccessState::IndexRead,
        ResourceAccessState::IndirectRead,
        ResourceAccessState::CopySource,
        ResourceAccessState::CopyDestination,
    ];

    /// Every state that names a texture access.
    const TEXTURE_STATES: [ResourceAccessState; 14] = [
        ResourceAccessState::Undefined,
        ResourceAccessState::ColorAttachmentRead,
        ResourceAccessState::ColorAttachmentWrite,
        ResourceAccessState::ColorAttachmentReadWrite,
        ResourceAccessState::DepthStencilRead,
        ResourceAccessState::DepthStencilWrite,
        ResourceAccessState::DepthStencilReadWrite,
        ResourceAccessState::ShaderSampledRead,
        ResourceAccessState::ShaderStorageRead,
        ResourceAccessState::ShaderStorageWrite,
        ResourceAccessState::ShaderStorageReadWrite,
        ResourceAccessState::CopySource,
        ResourceAccessState::CopyDestination,
        ResourceAccessState::Present,
    ];

    #[test]
    fn every_named_buffer_state_lowers_to_a_non_empty_stage_mask() {
        // An empty stage mask is not a legal `vkCmdPipelineBarrier` input, so this
        // is the property a guessed mapping would break first.
        for state in BUFFER_STATES {
            let scope = buffer_scope(state).unwrap_or_else(|| panic!("{state:?} is a buffer state"));
            assert!(
                !scope.stage.is_empty(),
                "{state:?} lowered to an empty stage mask"
            );
        }
    }

    #[test]
    fn every_named_texture_state_lowers_to_a_non_empty_stage_mask() {
        for state in TEXTURE_STATES {
            let image = image_state(state, false)
                .unwrap_or_else(|| panic!("{state:?} is a texture state"));
            assert!(
                !image.scope.stage.is_empty(),
                "{state:?} lowered to an empty stage mask"
            );
        }
    }

    #[test]
    fn a_texture_only_state_is_refused_for_a_buffer() {
        // The resource kind is part of the state's meaning: an attachment state
        // cannot be ordered as a buffer access, and saying so is the refusal.
        assert_eq!(
            buffer_scope(ResourceAccessState::ColorAttachmentWrite),
            None
        );
        assert_eq!(buffer_scope(ResourceAccessState::DepthStencilWrite), None);
        assert_eq!(buffer_scope(ResourceAccessState::ShaderSampledRead), None);
        assert_eq!(buffer_scope(ResourceAccessState::Present), None);
    }

    #[test]
    fn a_buffer_only_state_is_refused_for_an_image() {
        assert_eq!(image_state(ResourceAccessState::UniformRead, false), None);
        assert_eq!(image_state(ResourceAccessState::VertexRead, false), None);
        assert_eq!(image_state(ResourceAccessState::IndexRead, false), None);
        assert_eq!(image_state(ResourceAccessState::IndirectRead, false), None);
    }

    #[test]
    fn the_same_state_transition_is_still_a_barrier() {
        // The preserved semantic, expressed as the shape of what is built: a
        // transition from a state to itself yields a barrier whose two sides are
        // equal and whose layout does not change. If a later edit adds a
        // `before == after` early return, this test cannot be written any more --
        // which is exactly the signal that the semantic was lost.
        let state = ResourceAccessState::ColorAttachmentWrite;
        let lowered = image_state(state, false).expect("an attachment state");
        let barrier = image_barrier(
            vk::Image::null(),
            vk::ImageSubresourceRange::default(),
            lowered,
            lowered,
        );
        assert_eq!(barrier.old_layout, barrier.new_layout);
        assert_eq!(barrier.src_access_mask, barrier.dst_access_mask);
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        assert!(
            !barrier.dst_access_mask.is_empty(),
            "a same-state write barrier still carries its access"
        );

        let buffer = ResourceAccessState::ShaderStorageWrite;
        let scope = buffer_scope(buffer).expect("a storage state");
        let barrier = buffer_barrier(
            vk::Buffer::null(),
            BufferRange::Whole,
            scope,
            scope,
        );
        assert_eq!(barrier.src_access_mask, barrier.dst_access_mask);
        assert_eq!(barrier.dst_access_mask, vk::AccessFlags::SHADER_WRITE);
    }

    #[test]
    fn the_sampled_read_layout_follows_the_format() {
        // The one lowering whose answer depends on the format: a depth texture
        // sampled through the colour read-only layout is an invalid barrier.
        let color = image_state(ResourceAccessState::ShaderSampledRead, false)
            .expect("a sampled state");
        let depth = image_state(ResourceAccessState::ShaderSampledRead, true)
            .expect("a sampled state");
        assert_eq!(color.layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            depth.layout,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        );
        assert_eq!(color.scope, depth.scope, "only the layout depends on format");
    }

    #[test]
    fn a_sampled_read_orders_against_every_shader_stage() {
        // The state does not name a stage, so narrowing it to fragment would be a
        // second fact no caller wrote down.
        let sampled = image_state(ResourceAccessState::ShaderSampledRead, false)
            .expect("a sampled state");
        for stage in [
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ] {
            assert!(sampled.scope.stage.contains(stage), "{stage:?}");
        }
        assert_eq!(sampled.scope.access, vk::AccessFlags::SHADER_READ);
    }

    #[test]
    fn each_copy_direction_gets_its_own_stage_access_and_layout() {
        let source = image_state(ResourceAccessState::CopySource, false).expect("a copy source");
        let destination =
            image_state(ResourceAccessState::CopyDestination, false).expect("a copy destination");
        assert_eq!(source.scope.stage, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(destination.scope.stage, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(source.scope.access, vk::AccessFlags::TRANSFER_READ);
        assert_eq!(destination.scope.access, vk::AccessFlags::TRANSFER_WRITE);
        assert_eq!(source.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(destination.layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    }

    #[test]
    fn a_whole_range_covers_every_subresource_of_the_format() {
        let color = image_subresource_range(TextureRange::Whole, false).expect("a whole range");
        assert_eq!(color.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!(color.base_mip_level, 0);
        assert_eq!(color.level_count, vk::REMAINING_MIP_LEVELS);
        assert_eq!(color.base_array_layer, 0);
        assert_eq!(color.layer_count, vk::REMAINING_ARRAY_LAYERS);

        let depth = image_subresource_range(TextureRange::Whole, true).expect("a whole range");
        assert_eq!(depth.aspect_mask, vk::ImageAspectFlags::DEPTH);
    }

    #[test]
    fn an_explicit_range_keeps_its_aspect_and_levels() {
        let range = image_subresource_range(
            TextureRange::Subresources {
                base_mip_level: 2,
                mip_level_count: 3,
                base_array_layer: 1,
                array_layer_count: 2,
                aspect: TextureAspect::Stencil,
            },
            false,
        )
        .expect("a non-empty subresource range");
        assert_eq!(range.aspect_mask, vk::ImageAspectFlags::STENCIL);
        assert_eq!(range.base_mip_level, 2);
        assert_eq!(range.level_count, 3);
        assert_eq!(range.base_array_layer, 1);
        assert_eq!(range.layer_count, 2);
    }

    #[test]
    fn a_zero_count_subresource_range_is_refused() {
        // Vulkan requires both counts to be non-zero, and the portable variant
        // documents them as non-empty, so a zero here is a caller's mistake rather
        // than a description to send on.
        assert!(
            image_subresource_range(
                TextureRange::Subresources {
                    base_mip_level: 0,
                    mip_level_count: 0,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: TextureAspect::Color,
                },
                false,
            )
            .is_none()
        );
        assert!(
            image_subresource_range(
                TextureRange::Subresources {
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 0,
                    aspect: TextureAspect::Color,
                },
                false,
            )
            .is_none()
        );
    }

    #[test]
    fn a_same_queue_barrier_ignores_both_queue_family_indices() {
        // One queue family records and submits everything, so there is no
        // ownership transfer to express -- and an ignored value says that, while a
        // zero would claim the barrier is about family zero.
        let scope = buffer_scope(ResourceAccessState::VertexRead).expect("a vertex state");
        let buffer = buffer_barrier(
            vk::Buffer::null(),
            BufferRange::Bytes {
                offset: 16,
                size: 64,
            },
            scope,
            scope,
        );
        assert_eq!(buffer.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(buffer.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(buffer.offset, 16);
        assert_eq!(buffer.size, 64);

        let whole = buffer_barrier(vk::Buffer::null(), BufferRange::Whole, scope, scope);
        assert_eq!(whole.offset, 0);
        assert_eq!(whole.size, vk::WHOLE_SIZE);

        let image = image_barrier(
            vk::Image::null(),
            vk::ImageSubresourceRange::default(),
            image_state(ResourceAccessState::ShaderStorageWrite, false).expect("a storage state"),
            image_state(ResourceAccessState::ShaderStorageWrite, false).expect("a storage state"),
        );
        assert_eq!(image.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(image.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }
}
