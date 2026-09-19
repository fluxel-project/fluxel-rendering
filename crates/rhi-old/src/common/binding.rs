//! The bind-group layout vocabulary a backend creates descriptor sets from.
//!
//! # Why this vocabulary is closed, and what "closed" means here
//!
//! A bind-group layout entry states what a shader may read or write at one binding
//! number: which stages see it, and which kind of resource sits there. That is a
//! backend fact -- a graph may not depend on a binding number, a descriptor type or
//! a view dimension -- so it belongs in this layer rather than in
//! `fluxel-rendergraph` (plan section 9.1).
//!
//! The set of kinds is the closed one the retained artifacts declare, exactly as
//! [`super::vertex`] is closed: a caller cannot describe a binding no retained
//! artifact uses, and a new kind arrives only with the artifact that needs it. What
//! is deliberately absent is descriptor *arrays*: every retained binding is a
//! single descriptor, every backend lowers it to a count of one, and a `count`
//! field nobody sets would be vocabulary for a recipe that does not exist.
//!
//! # The one rule that is checked here, and where the rest is checked
//!
//! Two entries naming the same binding number cannot describe one set, and an
//! entry no shader stage can see can never be read. Both are properties of the
//! layout itself and both are refused by [`BindGroupLayout::validate`] before a
//! backend sees the description, because a driver validation error is a worse
//! answer than a reason a caller can act on.
//!
//! What is *not* checked here is anything needing a device: whether a format can be
//! a storage texture, whether a sample count is supported, or whether a filterable
//! float sample is legal for the format. Those are capability questions and belong
//! to the ledger, not to a value type.

use fluxel_rendergraph::TextureFormat;

use crate::common::base::resource::{BufferId, SamplerId, TextureId};

/// Which shader stages may read one binding.
///
/// A bitset rather than an enum because the retained raster recipe's frame uniform
/// is visible to two stages at once, and because visibility composes: the native
/// lowering folds the bits it holds into the backend's own stage flags. The
/// representation is private and only the named constants and [`Self::union`]
/// produce a value, so a bit that no stage corresponds to cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShaderVisibility(u8);

impl ShaderVisibility {
    /// Visible to the vertex stage only.
    pub(crate) const VERTEX: Self = Self(0b0000_0001);
    /// Visible to the fragment stage only.
    pub(crate) const FRAGMENT: Self = Self(0b0000_0010);
    /// Visible to the compute stage only.
    pub(crate) const COMPUTE: Self = Self(0b0000_0100);
    /// Visible to both the vertex and the fragment stage.
    ///
    /// The retained raster recipe's frame uniform has exactly this visibility, and
    /// naming the pair rather than writing `VERTEX.union(FRAGMENT)` at each call
    /// site is what keeps the two backends from spelling it differently.
    pub(crate) const VERTEX_FRAGMENT: Self = Self(Self::VERTEX.0 | Self::FRAGMENT.0);

    /// Every stage in both values is visible.
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every stage in `other` is also visible here.
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no shader stage can see the binding.
    ///
    /// An entry in this state is refused by [`BindGroupLayout::validate`]: a
    /// descriptor no stage can read is a description mistake, not a valid layout.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// How a buffer binding may be used by the shader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BufferBindingType {
    /// A uniform buffer: read-only, and the backend may require a minimum size.
    Uniform,
    /// A storage buffer the shader reads, or reads and writes.
    Storage {
        /// Whether the shader only reads the buffer.
        read_only: bool,
    },
}

/// The element type a sampled texture binding exposes to the shader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TextureSampleType {
    /// A floating-point texture, filterable or not.
    ///
    /// `filterable` is carried rather than derived because the retained linear-clamp
    /// recipe's legality depends on it and the capability ledger is what proved it.
    Float {
        /// Whether a filtering sampler may be used with this binding.
        filterable: bool,
    },
    /// A signed-integer texture.
    Sint,
    /// An unsigned-integer texture.
    Uint,
    /// A depth texture.
    Depth,
}

/// How a sampler binding is used by the shader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum SamplerBindingType {
    /// A sampler that interpolates; requires a filterable texture.
    Filtering,
    /// A sampler that may not interpolate.
    NonFiltering,
    /// A sampler that performs a per-sample comparison.
    Comparison,
}

/// What a shader may do to a storage texture binding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StorageTextureAccess {
    /// The shader writes texels but does not read them.
    WriteOnly,
    /// The shader reads texels but does not write them.
    ReadOnly,
    /// The shader reads and writes texels.
    ReadWrite,
}

/// The dimensionality of the view a binding reads.
///
/// Distinct from [`fluxel_rendergraph::TextureDimension`]: that states how the
/// storage is allocated, this states how one binding views it. A layered
/// two-dimensional texture is `D2` storage and a `D2Array` view, and conflating the
/// two is how a binding silently selects layer zero.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ViewDimension {
    /// A one-dimensional view.
    D1,
    /// A single two-dimensional layer.
    D2,
    /// Every layer of a layered two-dimensional texture.
    D2Array,
    /// A cube view.
    Cube,
    /// A three-dimensional view.
    D3,
}

/// What one binding number holds and which stages see it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BindingKind {
    /// A buffer binding.
    Buffer {
        /// How the buffer may be used.
        ty: BufferBindingType,
        /// Whether the bound offset is supplied at bind time rather than fixed.
        has_dynamic_offset: bool,
        /// The smallest range a bind group may supply, or `None` for no minimum.
        ///
        /// `None` is the storage-buffer case, where the recipe binds the whole
        /// allocation. `Some(0)` is not a meaningful minimum and is a caller error
        /// the bind-group check refuses; it is not rejected here because this layer
        /// only states the fact and no driver sees it at layout creation.
        min_binding_size: Option<u64>,
    },
    /// A sampled texture binding.
    Texture {
        /// The element type the binding exposes.
        sample_type: TextureSampleType,
        /// How the binding views the texture.
        view_dimension: ViewDimension,
        /// Whether the binding samples a multisampled texture.
        multisampled: bool,
    },
    /// A sampler binding.
    Sampler(SamplerBindingType),
    /// A texture the shader reads or writes directly.
    StorageTexture {
        /// What the shader may do to it.
        access: StorageTextureAccess,
        /// The format the view must have.
        format: TextureFormat,
        /// How the binding views the texture.
        view_dimension: ViewDimension,
    },
}

/// One entry of a bind-group layout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BindGroupLayoutEntry {
    /// The binding number the shader declares.
    pub binding: u32,
    /// The stages that may read this binding.
    pub visibility: ShaderVisibility,
    /// What the binding holds.
    pub kind: BindingKind,
}

/// A complete bind-group layout: the entries one descriptor set is created from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindGroupLayout {
    /// The entries, in the order the artifact declares them.
    pub entries: Vec<BindGroupLayoutEntry>,
}

/// Why a bind-group layout cannot describe a descriptor set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BindGroupLayoutError {
    /// Two entries name the same binding number.
    DuplicateBinding {
        /// The repeated binding number.
        binding: u32,
    },
    /// An entry is visible to no shader stage.
    EmptyVisibility {
        /// The binding number with no visibility.
        binding: u32,
    },
}

/// One resource a bind group binds at one binding number.
///
/// This is the *value* half of the bind-group vocabulary: the layout states what a
/// binding number accepts, and an entry states what is actually placed there. It
/// names a base resource id rather than a backend handle, so a backend lowers this
/// and never the other way round -- a raw handle cannot be written here at all.
///
/// A texture entry is one variant for both the sampled and the storage case: which
/// one it is, and therefore which descriptor type and image layout it lowers to, is
/// the *layout's* answer, and a second spelling here could disagree with it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BindingResource {
    /// A buffer range: the resource, the base offset and the bound size.
    Buffer {
        /// The buffer the binding reads.
        buffer: BufferId,
        /// The byte offset the bound range begins at.
        offset: u64,
        /// The size of the bound range in bytes.
        size: u64,
    },
    /// A texture view the binding reads or writes.
    Texture(TextureId),
    /// A sampler the binding reads through.
    Sampler(SamplerId),
}

/// One entry of a bind group: what one binding number is bound to.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BindGroupEntry {
    /// The binding number, which the layout must declare.
    pub binding: u32,
    /// The resource placed at that binding.
    pub resource: BindingResource,
}

impl BindGroupLayout {
    /// Returns the entry this layout declares at `binding`, or `None`.
    ///
    /// A backend lowering a bind group asks this rather than iterating the entries
    /// itself, so "which binding numbers exist" has one answer.
    pub(crate) fn entry(&self, binding: u32) -> Option<&BindGroupLayoutEntry> {
        self.entries.iter().find(|entry| entry.binding == binding)
    }

    /// Rejects a layout that cannot describe one descriptor set.
    ///
    /// The whole check is local: it compares the entries against each other and
    /// never against a device, because a limit or capability question belongs to the
    /// ledger. An empty layout is legal and is what the retained compute artifact
    /// that declares no bindings asks for.
    pub(crate) fn validate(&self) -> Result<(), BindGroupLayoutError> {
        let mut seen = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if seen.contains(&entry.binding) {
                return Err(BindGroupLayoutError::DuplicateBinding {
                    binding: entry.binding,
                });
            }
            seen.push(entry.binding);
            if entry.visibility.is_empty() {
                return Err(BindGroupLayoutError::EmptyVisibility {
                    binding: entry.binding,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::base::stamp::DeviceStamp;
    use fluxel_rendergraph::{DeviceIdentity, PhysicalResourceIdentity};

    fn entry(binding: u32, visibility: ShaderVisibility, kind: BindingKind) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            kind,
        }
    }

    fn uniform(binding: u32) -> BindGroupLayoutEntry {
        entry(
            binding,
            ShaderVisibility::VERTEX_FRAGMENT,
            BindingKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(80),
            },
        )
    }

    fn texture(binding: u32, filterable: bool) -> BindGroupLayoutEntry {
        entry(
            binding,
            ShaderVisibility::FRAGMENT,
            BindingKind::Texture {
                sample_type: TextureSampleType::Float { filterable },
                view_dimension: ViewDimension::D2,
                multisampled: false,
            },
        )
    }

    fn sampler(binding: u32) -> BindGroupLayoutEntry {
        entry(
            binding,
            ShaderVisibility::FRAGMENT,
            BindingKind::Sampler(SamplerBindingType::Filtering),
        )
    }

    #[test]
    fn the_retained_textured_raster_layout_is_accepted() {
        // The exact layout the linear-clamp artifact declares: one frame uniform,
        // one filterable texture, one filtering sampler.
        let layout = BindGroupLayout {
            entries: vec![uniform(0), texture(1, true), sampler(2)],
        };
        assert_eq!(layout.validate(), Ok(()));
    }

    #[test]
    fn an_empty_layout_is_accepted() {
        // The retained compute artifact that declares no bindings asks for this.
        let layout = BindGroupLayout { entries: Vec::new() };
        assert_eq!(layout.validate(), Ok(()));
    }

    #[test]
    fn two_entries_may_not_share_a_binding_number() {
        let layout = BindGroupLayout {
            entries: vec![uniform(0), texture(0, false)],
        };
        assert_eq!(
            layout.validate(),
            Err(BindGroupLayoutError::DuplicateBinding { binding: 0 })
        );
    }

    #[test]
    fn an_entry_no_stage_can_see_is_rejected() {
        // The constants never produce this state; the raw representation is
        // reachable only from inside this module, which is where the refusal is
        // tested so that no other module can assemble the input to it.
        let invisible = ShaderVisibility(0);
        assert!(invisible.is_empty());
        let layout = BindGroupLayout {
            entries: vec![entry(
                3,
                invisible,
                BindingKind::Sampler(SamplerBindingType::NonFiltering),
            )],
        };
        assert_eq!(
            layout.validate(),
            Err(BindGroupLayoutError::EmptyVisibility { binding: 3 })
        );
    }

    #[test]
    fn visibility_composes_without_gaining_a_stage() {
        let both = ShaderVisibility::VERTEX.union(ShaderVisibility::FRAGMENT);
        assert!(both.contains(ShaderVisibility::VERTEX));
        assert!(both.contains(ShaderVisibility::FRAGMENT));
        // The pair was named rather than derived, so the constant and the union
        // cannot disagree about which stages are visible.
        assert_eq!(both, ShaderVisibility::VERTEX_FRAGMENT);
        assert!(!both.contains(ShaderVisibility::COMPUTE));
        assert!(!ShaderVisibility::COMPUTE.contains(ShaderVisibility::VERTEX));
    }

    #[test]
    fn the_binding_kinds_are_distinct_values() {        // Every kind a retained artifact declares, kept distinct so a copy-paste in
        // a lowering match shows up as two kinds comparing equal.
        let kinds = [
            BindingKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(80),
            },
            BindingKind::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingKind::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: ViewDimension::D2,
                multisampled: false,
            },
            BindingKind::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: ViewDimension::D2,
                multisampled: false,
            },
            BindingKind::Sampler(SamplerBindingType::Filtering),
            BindingKind::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: ViewDimension::D2,
            },
        ];
        for (index, kind) in kinds.iter().enumerate() {
            for (other_index, other) in kinds.iter().enumerate() {
                if index != other_index {
                    assert_ne!(kind, other, "kinds {index} and {other_index} compare equal");
                }
            }
        }
    }

    fn stamp() -> DeviceStamp {
        DeviceStamp::initial(DeviceIdentity::new(1))
    }

    #[test]
    fn a_layout_answers_for_the_bindings_it_declares() {
        // One answer to "which binding numbers exist", so a backend does not invent a
        // second scan that could disagree with the layout.
        let layout = BindGroupLayout {
            entries: vec![uniform(0), texture(1, true), sampler(2)],
        };
        assert_eq!(layout.entry(1).map(|entry| entry.binding), Some(1));
        assert_eq!(layout.entry(0).map(|entry| entry.visibility), Some(ShaderVisibility::VERTEX_FRAGMENT));
        assert!(layout.entry(3).is_none());
    }

    #[test]
    fn a_bind_group_entry_carries_its_binding_and_its_resource() {
        // The value half of the vocabulary: a buffer range keeps both numbers, and
        // the three resource kinds are distinct values.
        let buffer = BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: BufferId::new(stamp(), PhysicalResourceIdentity::new(1)),
                offset: 16,
                size: 80,
            },
        };
        let texture = BindGroupEntry {
            binding: 1,
            resource: BindingResource::Texture(TextureId::new(
                stamp(),
                PhysicalResourceIdentity::new(2),
            )),
        };
        let sampler = BindGroupEntry {
            binding: 2,
            resource: BindingResource::Sampler(SamplerId::new(
                stamp(),
                PhysicalResourceIdentity::new(3),
            )),
        };
        assert_eq!(buffer.binding, 0);
        match buffer.resource {
            BindingResource::Buffer { offset, size, .. } => {
                assert_eq!(offset, 16);
                assert_eq!(size, 80);
            }
            _ => panic!("the entry is a buffer"),
        }
        assert_ne!(buffer, texture);
        assert_ne!(texture, sampler);
        assert_ne!(buffer, sampler);
    }
}
