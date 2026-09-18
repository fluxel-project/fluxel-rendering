//! W2's storage-role families, pure half: the facts one storage binding is admitted
//! against.
//!
//! Both families the `common` layer calls *resource roles* live here, because both
//! answer the same shape of question -- is this resource one a storage binding may
//! name, and for what? -- without recording a command:
//!
//! - [`StorageBufferBinding`] is the buffer half: a validated range that fits the
//!   buffer's own declared size.
//! - [`StorageTextureBinding`] is the texture half: a texture whose declared usage
//!   names a storage direction and whose `(format, sample count)` pair the device's
//!   format table proved for every direction it declared.
//!
//! # `Vulkan` has no storage-buffer object, and no storage-texture one either
//!
//! A storage buffer is a `VkDescriptorBufferInfo` written into a descriptor set and a
//! storage texture is a `VkDescriptorImageInfo` naming the view the table already
//! owns, so what this backend can build before a set exists is the *fact* the write
//! needs. The range's end is computed in checked arithmetic, so a `u64::MAX` offset is
//! a refusal rather than a wrapped range that passes the bound.
//!
//! [`super::bind_group::buffer_info`] obeys exactly that rule where a binding number
//! and a driver handle exist -- the boundary repetition step 8 records for the copy
//! ranges -- and it calls [`check_range`] rather than keeping a second spelling of
//! the arithmetic. There is one implementation and the driver path inherits it.
//!
//! # Why the bindings are values rather than handles
//!
//! `StorageBuffer` and `StorageTexture` are *resource roles*: they gate how a binding
//! is built and record nothing, so the handles that build them own no encoder. The two
//! values carry base resource identity and nothing else, and their `at` methods are how
//! they reach a [`BindGroupEntry`]: the binding number is the layout's business, and a
//! value that carried one before a layout named it would be inventing a fact.
//!
//! # What is deliberately not here
//!
//! No alignment rule. `Vulkan` places no offset alignment on a storage buffer beyond
//! the minimum it reports for the descriptor type, and that minimum is a device limit
//! the bind-group step reads where it can actually enforce it; a second, invented
//! alignment here would refuse ranges the driver accepts.
//!
//! No view dimension rule either. The view the table owns is the one the texture's
//! description built, and `Vulkan` accepts a storage image descriptor over it; a
//! second classification here would be a second answer about a fact the description
//! already carries.

use fluxel_rendergraph::{TextureDesc, TextureFormat, TextureUsage, TextureUsageKind};

use crate::common::base::resource::{BufferId, TextureId};
use crate::common::binding::{BindGroupEntry, BindingResource};
use crate::common::formats::FormatCapabilities;

/// Why a storage-buffer range cannot be bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RangeError {
    /// The range is zero bytes, so it binds nothing.
    Zero,
    /// The range does not fit inside the buffer, or its end would wrap.
    OutOfBounds,
}

/// Checks the one rule every storage-buffer range obeys.
///
/// The size is checked before the end, so an empty range is its own sentence rather
/// than an arithmetic result, and the end is computed with `checked_add`, so
/// `offset = u64::MAX` is an overrun instead of a wrapped range that satisfies
/// `end <= buffer_size`.
pub(crate) fn check_range(offset: u64, size: u64, buffer_size: u64) -> Result<(), RangeError> {
    if size == 0 {
        return Err(RangeError::Zero);
    }
    let fits = offset
        .checked_add(size)
        .is_some_and(|end| end <= buffer_size);
    if !fits {
        return Err(RangeError::OutOfBounds);
    }
    Ok(())
}

/// A storage-buffer range this device's table accepted.
///
/// It is produced only by the storage-buffer family's verb, which is what makes "a
/// storage binding names a buffer this device owns, for a range that fits it" a fact
/// of the value rather than a check the caller has to remember.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageBufferBinding {
    /// The buffer the binding reads or writes.
    buffer: BufferId,
    /// The byte offset the bound range begins at.
    offset: u64,
    /// The size of the bound range in bytes.
    size: u64,
}

impl StorageBufferBinding {
    /// Wraps a range already checked against the buffer's own declared size.
    pub(crate) const fn new(buffer: BufferId, offset: u64, size: u64) -> Self {
        Self {
            buffer,
            offset,
            size,
        }
    }

    /// Returns the buffer this binding names.
    pub(crate) const fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// Returns the byte offset the bound range begins at.
    pub(crate) const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the bound range's size in bytes.
    pub(crate) const fn size(&self) -> u64 {
        self.size
    }

    /// Places this binding at `binding` of a bind group.
    ///
    /// The number is supplied here rather than stored, because the layout is what
    /// declares which numbers exist and the family verb that built this value never
    /// saw one.
    pub(crate) const fn at(&self, binding: u32) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Buffer {
                buffer: self.buffer,
                offset: self.offset,
                size: self.size,
            },
        }
    }
}

/// The two directions a storage-texture binding can be declared for.
///
/// It exists so a refusal can name which direction the format failed to prove, which
/// is the fact a caller needs to fix: a read-only row and a write-only row describe
/// different domains, and `FormatCapabilities` deliberately keeps them split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageDirection {
    /// The shader reads texels through the binding.
    Read,
    /// The shader writes texels through the binding.
    Write,
}

/// Why one live texture may not be named by a storage-texture binding.
///
/// Every variant is a value returned before the driver is reached, and each names a
/// different fact, because the fixes differ: the description named more levels than
/// this vocabulary can address, the texture was created for something other than
/// storage, no discovery examined its format, or the format was examined and does not
/// serve the direction the usage declared.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageTextureRuleError {
    /// The description declares more than one mip level.
    ///
    /// The view the resource table owns spans every declared level, so binding such a
    /// texture would name a view covering more than the one level this vocabulary
    /// describes. A per-level view is the subresource question
    /// [`crate::common::api::families::StorageTextureApi`] says it does not carry.
    MultiLevel {
        /// The level count the description declared.
        levels: u32,
    },
    /// The declared usage names neither storage direction.
    UsageNotDeclared,
    /// Nothing examined the texture's `(format, sample count)` pair.
    FormatUnproved {
        /// The format the texture was created with.
        format: TextureFormat,
        /// The sample count the texture was created with.
        sample_count: u32,
    },
    /// The pair was examined and this declared direction is not supported.
    DirectionUnproved {
        /// The direction the declared usage named.
        direction: StorageDirection,
        /// The format that failed to prove it.
        format: TextureFormat,
    },
}

/// Decides whether one live texture may be named by a storage-texture binding.
///
/// The description is checked first, then the declared usage, then the device's own
/// per-format facts, so the cheapest and most specific sentence is the one a caller
/// reads. The sample count is the description's, not a constant: this backend creates
/// single-sample images only ([`super::texture::image_create_info`]), and asking the
/// table for the count the texture actually has means a multisampled row arriving
/// later is answered by the table rather than by an assumption here.
///
/// Every declared direction must be proved. The buffer half deliberately asks only for
/// "at least one storage direction", because the layout -- which the family verb never
/// sees -- decides whether that binding is read-only or read-write. A texture's usage
/// names its directions itself, so the stronger question is also the honest one: a
/// texture declared for writes and bound through a read-write layout would otherwise
/// be admitted on a proof only half of its declared use has.
pub(crate) fn check_storage_texture(
    desc: &TextureDesc,
    usage: TextureUsage,
    facts: Option<FormatCapabilities>,
) -> Result<(), StorageTextureRuleError> {
    if desc.mip_levels > 1 {
        return Err(StorageTextureRuleError::MultiLevel {
            levels: desc.mip_levels,
        });
    }
    let read = usage.contains(TextureUsageKind::StorageRead);
    let write = usage.contains(TextureUsageKind::StorageWrite);
    if !read && !write {
        return Err(StorageTextureRuleError::UsageNotDeclared);
    }
    let facts = facts.ok_or(StorageTextureRuleError::FormatUnproved {
        format: desc.format,
        sample_count: desc.sample_count,
    })?;
    // Read is asked first, in the vocabulary's own declaration order, so a row that
    // proves neither direction answers with the direction the usage named first.
    for (declared, proved, direction) in [
        (read, facts.storage_read, StorageDirection::Read),
        (write, facts.storage_write, StorageDirection::Write),
    ] {
        if declared && !proved {
            return Err(StorageTextureRuleError::DirectionUnproved {
                direction,
                format: desc.format,
            });
        }
    }
    Ok(())
}

/// A texture this device's table admitted as a storage binding.
///
/// It is produced only by the storage-texture family's verb, which is what makes "a
/// storage binding names a texture this device owns, for a direction its format
/// proved" a fact of the value rather than a check the caller has to remember. It
/// carries base identity rather than the view, so the driver handle stays inside the
/// table exactly as [`StorageBufferBinding`] keeps the buffer handle there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StorageTextureBinding {
    /// The texture the binding reads or writes.
    texture: TextureId,
}

impl StorageTextureBinding {
    /// Wraps a texture whose usage and format facts were already checked.
    pub(crate) const fn new(texture: TextureId) -> Self {
        Self { texture }
    }

    /// Returns the texture this binding names.
    pub(crate) const fn texture(&self) -> TextureId {
        self.texture
    }

    /// Places this binding at `binding` of a bind group.
    ///
    /// The number is supplied here rather than stored, for the reason
    /// [`StorageBufferBinding::at`] states: the layout declares which numbers exist.
    pub(crate) const fn at(&self, binding: u32) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Texture(self.texture),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::base::stamp::DeviceStamp;
    use crate::common::formats::FormatEvidence;
    use fluxel_rendergraph::{
        DeviceIdentity, Extent3d, PhysicalResourceIdentity, TextureDimension,
    };

    fn buffer() -> BufferId {
        BufferId::new(
            DeviceStamp::initial(DeviceIdentity::new(1)),
            PhysicalResourceIdentity::new(7),
        )
    }

    #[test]
    fn a_range_that_fits_the_buffer_is_accepted() {
        assert_eq!(check_range(0, 256, 256), Ok(()));
        assert_eq!(check_range(256, 256, 512), Ok(()));
        // The end exactly at the buffer's own end is inside it.
        assert_eq!(check_range(128, 128, 256), Ok(()));
    }

    #[test]
    fn a_zero_sized_range_is_refused_by_name() {
        // `Vulkan` would accept the descriptor, so the refusal has to happen here:
        // a binding that names no bytes is a description mistake, not a valid range.
        assert_eq!(check_range(0, 0, 256), Err(RangeError::Zero));
        assert_eq!(check_range(256, 0, 256), Err(RangeError::Zero));
    }

    #[test]
    fn a_range_past_the_buffer_is_refused() {
        assert_eq!(check_range(128, 256, 256), Err(RangeError::OutOfBounds));
        assert_eq!(check_range(257, 1, 256), Err(RangeError::OutOfBounds));
    }

    #[test]
    fn a_wrapping_end_is_an_overrun_rather_than_a_range_that_passes() {
        // `u64::MAX - 3` is the largest size whose end still wraps, and `u64::MAX`
        // itself is the offset the borrowed path's unchecked addition would wrap.
        assert_eq!(
            check_range(u64::MAX - 3, 8, u64::MAX),
            Err(RangeError::OutOfBounds)
        );
        assert_eq!(
            check_range(u64::MAX, 1, u64::MAX),
            Err(RangeError::OutOfBounds)
        );
    }

    #[test]
    fn a_binding_carries_its_buffer_and_its_range() {
        let binding = StorageBufferBinding::new(buffer(), 64, 128);
        assert_eq!(binding.buffer(), buffer());
        assert_eq!(binding.offset(), 64);
        assert_eq!(binding.size(), 128);
    }

    #[test]
    fn a_binding_reaches_a_bind_group_entry_at_the_number_it_is_given() {
        // The layout declares the number, so the same value can fill any number
        // without the family having guessed one.
        let binding = StorageBufferBinding::new(buffer(), 16, 80);
        let entry = binding.at(3);
        assert_eq!(entry.binding, 3);
        assert_eq!(
            entry.resource,
            BindingResource::Buffer {
                buffer: buffer(),
                offset: 16,
                size: 80,
            }
        );
        assert_ne!(binding.at(0), binding.at(1));
    }

    #[test]
    fn two_bindings_differ_when_any_of_their_three_facts_do() {
        let base = StorageBufferBinding::new(buffer(), 0, 256);
        assert_eq!(base, StorageBufferBinding::new(buffer(), 0, 256));
        assert_ne!(base, StorageBufferBinding::new(buffer(), 4, 256));
        assert_ne!(base, StorageBufferBinding::new(buffer(), 0, 128));
        assert_ne!(
            base,
            StorageBufferBinding::new(
                BufferId::new(buffer().stamp(), PhysicalResourceIdentity::new(8)),
                0,
                256,
            )
        );
    }

    fn described(format: TextureFormat, mip_levels: u32) -> TextureDesc {
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 16,
                height: 8,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            sample_count: 1,
            format,
        }
    }

    fn declared(kinds: &[TextureUsageKind]) -> TextureUsage {
        TextureUsage::from_kinds(kinds.iter().copied())
    }

    /// One examined row whose storage facts are exactly the two arguments.
    fn examined(format: TextureFormat, read: bool, write: bool) -> FormatCapabilities {
        FormatCapabilities {
            format,
            sample_count: 1,
            evidence: FormatEvidence::OperationProbed,
            sampled: true,
            filterable: true,
            renderable: true,
            blendable: true,
            storage_read: read,
            storage_write: write,
            copy_source: true,
            copy_destination: true,
        }
    }

    #[test]
    fn a_texture_whose_format_proves_every_declared_direction_is_admitted() {
        let usage = declared(&[TextureUsageKind::StorageRead, TextureUsageKind::StorageWrite]);
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 1),
                usage,
                Some(examined(TextureFormat::Rgba8Unorm, true, true)),
            ),
            Ok(())
        );
        // One declared direction needs only that direction proved: the format row is
        // the domain, and the usage names the half of it this texture is for.
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 1),
                declared(&[TextureUsageKind::StorageRead]),
                Some(examined(TextureFormat::Rgba8Unorm, true, false)),
            ),
            Ok(())
        );
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 1),
                declared(&[TextureUsageKind::StorageWrite]),
                Some(examined(TextureFormat::Rgba8Unorm, false, true)),
            ),
            Ok(())
        );
    }

    #[test]
    fn a_texture_created_for_something_else_is_refused_by_name() {
        // The texture step's rule is that the mapping never widens, so a sampled
        // texture is not a storage one however the driver's format facts read.
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 1),
                declared(&[TextureUsageKind::Sampled]),
                Some(examined(TextureFormat::Rgba8Unorm, true, true)),
            ),
            Err(StorageTextureRuleError::UsageNotDeclared)
        );
    }

    #[test]
    fn an_unexamined_format_is_refused_as_unproved_rather_than_unsupported() {
        // The two are different sentences, exactly as the format table keeps them:
        // nothing asked about this pair, which is not the same as a proved refusal.
        let desc = described(TextureFormat::Rgba8Unorm, 1);
        assert_eq!(
            check_storage_texture(&desc, declared(&[TextureUsageKind::StorageWrite]), None),
            Err(StorageTextureRuleError::FormatUnproved {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            })
        );
    }

    #[test]
    fn an_examined_format_that_misses_a_declared_direction_names_it() {
        let desc = described(TextureFormat::Rgba8Unorm, 1);
        assert_eq!(
            check_storage_texture(
                &desc,
                declared(&[TextureUsageKind::StorageRead, TextureUsageKind::StorageWrite]),
                Some(examined(TextureFormat::Rgba8Unorm, true, false)),
            ),
            Err(StorageTextureRuleError::DirectionUnproved {
                direction: StorageDirection::Write,
                format: TextureFormat::Rgba8Unorm,
            })
        );
        // Read is asked first, so a row that proves neither names read.
        assert_eq!(
            check_storage_texture(
                &desc,
                declared(&[TextureUsageKind::StorageRead, TextureUsageKind::StorageWrite]),
                Some(examined(TextureFormat::Rgba8Unorm, false, false)),
            ),
            Err(StorageTextureRuleError::DirectionUnproved {
                direction: StorageDirection::Read,
                format: TextureFormat::Rgba8Unorm,
            })
        );
    }

    #[test]
    fn a_multi_level_texture_is_refused_before_the_format_is_asked() {
        // The table's view spans every declared level, so this vocabulary -- which
        // names one -- refuses the description rather than binding a wider view.
        let usage = declared(&[TextureUsageKind::StorageRead]);
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 2),
                usage,
                Some(examined(TextureFormat::Rgba8Unorm, true, true)),
            ),
            Err(StorageTextureRuleError::MultiLevel { levels: 2 })
        );
        // Zero is the description's way of saying one, which is what the image and
        // view lowerings floor it to, so it is admitted.
        assert_eq!(
            check_storage_texture(
                &described(TextureFormat::Rgba8Unorm, 0),
                usage,
                Some(examined(TextureFormat::Rgba8Unorm, true, true)),
            ),
            Ok(())
        );
    }

    #[test]
    fn a_storage_texture_binding_carries_its_texture_at_the_number_it_is_given() {
        let texture = TextureId::new(buffer().stamp(), PhysicalResourceIdentity::new(9));
        let binding = StorageTextureBinding::new(texture);
        assert_eq!(binding.texture(), texture);
        let entry = binding.at(2);
        assert_eq!(entry.binding, 2);
        assert_eq!(entry.resource, BindingResource::Texture(texture));
        assert_ne!(binding.at(0), binding.at(1));
        assert_ne!(
            binding,
            StorageTextureBinding::new(TextureId::new(
                buffer().stamp(),
                PhysicalResourceIdentity::new(10),
            ))
        );
    }
}
