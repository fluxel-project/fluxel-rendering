//! The mapping from a recorded command to the actual uses it produces
//! (specification sections 37.2, 37.3, and 34.6).
//!
//! This module owns one thing: *given a bound resource and the reason a command
//! touched it, which* [`ResourceUse`] *describes that touch*. It is the single
//! place where the access masks of section 37.2's table are written down, so that
//! the raster path and the compute path cannot disagree about what a
//! `StorageBuffer { access: ReadWrite }` costs, and the single place where a
//! copy's direction becomes an access, so that a copy source cannot be read as a
//! write in one verb and a read in another.
//!
//! It also holds the one validation rule the two scopes state identically: section
//! 22.4's dynamic-offset rule, [`require_valid_dynamic_offsets`]. It lives here for
//! the same reason the access masks do — "dynamic offsets valid" is one sentence
//! that section 32.3 and section 33 each quote, so it is written once rather than
//! once per scope.
//!
//! # What this module does not own
//!
//! - When a use happens. Section 32.4 generates actual use at draw and dispatch
//!   and not at the state-setting verbs; that "when" is
//!   [`crate::api::command::raster`]'s, [`crate::api::command::compute`]'s, and
//!   [`crate::api::command::CommandRecorder`]'s.
//! - Upper-layer scheduling contracts. This module records only the portable
//!   commands that actually happened.
//! - Sampler participation. Section 37.2 says a sampler generates no memory
//!   hazard but still enters command semantics; the only `ResourceUse` variants
//!   are buffer, texture, and frame, so a sampler produces no record here. It is
//!   not lost: the [`crate::api::binding::BindGroup`] that holds it is cloned
//!   into the recorded command, so the sampler is still part of what the command
//!   says.
//!
//! # The invariant this module enforces
//!
//! **Every element of a `Fixed(n)` binding is used.** Section 37.2 permits a
//! smaller element set only when certified metadata proves it, and P0 has no
//! such metadata, so a conservatively-whole use is the only true answer here. A
//! use list that covered one element of a four-element array would let a hazard
//! through, and no later stage could tell that it had.

use crate::api::binding::{
    BindGroup, BindGroupIndex, BindingKind, BindingResource, BufferBindingAccess, StorageAccess,
};
use crate::api::command::copy::BufferTextureCopy;
use crate::api::command::record::{BoundGroup, CopyRecord};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::pipeline::PipelineInterface;
use crate::api::presentation::FrameAttachment;
use crate::api::resource::buffer::{Buffer, BufferRange};
use crate::api::resource::subresource::{
    TextureSubresourceLayers, TextureSubresourceRange, aspect_bits,
};
use crate::api::resource::texture::{Texture, TextureDimension};
use crate::api::resource::view::TextureView;
use crate::api::shader::ShaderStages;

/// Which pipeline domains one actual resource use is visible to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineScope(u32);

impl PipelineScope {
    /// Vertex-stage access.
    pub const VERTEX: Self = Self(1 << 0);
    /// Fragment-stage access.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// Compute-stage access.
    pub const COMPUTE: Self = Self(1 << 2);
    /// Copy, upload, or readback access.
    pub const COPY: Self = Self(1 << 3);

    /// Returns whether this scope contains every bit in `other`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the union of two pipeline scopes.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::fmt::Display for PipelineScope {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_bit_names(
            formatter,
            self.0,
            &[
                (Self::VERTEX.0, "VERTEX"),
                (Self::FRAGMENT.0, "FRAGMENT"),
                (Self::COMPUTE.0, "COMPUTE"),
                (Self::COPY.0, "COPY"),
            ],
        )
    }
}

/// How an actual resource use accesses memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccessMask(u32);

impl AccessMask {
    /// Vertex-buffer read access.
    pub const VERTEX_READ: Self = Self(1 << 0);
    /// Index-buffer read access.
    pub const INDEX_READ: Self = Self(1 << 1);
    /// Uniform-buffer read access.
    pub const UNIFORM_READ: Self = Self(1 << 2);
    /// Shader resource read access.
    pub const SHADER_READ: Self = Self(1 << 3);
    /// Shader storage write access.
    pub const SHADER_WRITE: Self = Self(1 << 4);
    /// Color-attachment read access.
    pub const COLOR_READ: Self = Self(1 << 5);
    /// Color-attachment write access.
    pub const COLOR_WRITE: Self = Self(1 << 6);
    /// Depth-attachment read access.
    pub const DEPTH_READ: Self = Self(1 << 7);
    /// Depth-attachment write access.
    pub const DEPTH_WRITE: Self = Self(1 << 8);
    /// Stencil-attachment read access.
    pub const STENCIL_READ: Self = Self(1 << 9);
    /// Stencil-attachment write access.
    pub const STENCIL_WRITE: Self = Self(1 << 10);
    /// Copy-source read access.
    pub const COPY_READ: Self = Self(1 << 11);
    /// Copy-destination write access.
    pub const COPY_WRITE: Self = Self(1 << 12);

    /// Returns whether this mask contains every bit in `other`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the union of two access masks.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::fmt::Display for AccessMask {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_bit_names(
            formatter,
            self.0,
            &[
                (Self::VERTEX_READ.0, "VERTEX_READ"),
                (Self::INDEX_READ.0, "INDEX_READ"),
                (Self::UNIFORM_READ.0, "UNIFORM_READ"),
                (Self::SHADER_READ.0, "SHADER_READ"),
                (Self::SHADER_WRITE.0, "SHADER_WRITE"),
                (Self::COLOR_READ.0, "COLOR_READ"),
                (Self::COLOR_WRITE.0, "COLOR_WRITE"),
                (Self::DEPTH_READ.0, "DEPTH_READ"),
                (Self::DEPTH_WRITE.0, "DEPTH_WRITE"),
                (Self::STENCIL_READ.0, "STENCIL_READ"),
                (Self::STENCIL_WRITE.0, "STENCIL_WRITE"),
                (Self::COPY_READ.0, "COPY_READ"),
                (Self::COPY_WRITE.0, "COPY_WRITE"),
            ],
        )
    }
}

/// The portable role a texture played in an actual command.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUseIntent {
    /// Shader sampled or read-only access.
    ShaderRead,
    /// Shader read/write storage access.
    ShaderReadWrite,
    /// Color-attachment access.
    ColorAttachment,
    /// Read-only depth/stencil attachment access.
    DepthStencilRead,
    /// Writable depth/stencil attachment access.
    DepthStencilWrite,
    /// Copy-source access.
    CopySrc,
    /// Copy-destination access.
    CopyDst,
    /// Multisample resolve source.
    ResolveSrc,
    /// Multisample resolve destination.
    ResolveDst,
}

/// One buffer range used by recorded work.
#[derive(Clone)]
pub struct BufferUse {
    /// The buffer being used.
    pub buffer: Buffer,
    /// The byte range being used.
    pub range: BufferRange,
    /// Pipeline domains issuing the access.
    pub stages: PipelineScope,
    /// Memory access performed by the command.
    pub access: AccessMask,
}

/// One texture subresource range used by recorded work.
#[derive(Clone)]
pub struct TextureUse {
    /// The texture being used.
    pub texture: Texture,
    /// The subresources being used.
    pub subresources: TextureSubresourceRange,
    /// Pipeline domains issuing the access.
    pub stages: PipelineScope,
    /// Memory access performed by the command.
    pub access: AccessMask,
    /// Portable role of the texture in the command.
    pub intent: TextureUseIntent,
}

/// An acquired frame used by recorded work.
#[derive(Clone, Copy, Debug)]
pub struct FrameAttachmentUse {
    /// Identity of the acquired frame.
    pub frame: crate::api::presentation::AcquiredFrameId,
    /// Pipeline domains issuing the access.
    pub stages: PipelineScope,
    /// Memory access performed by the command.
    pub access: AccessMask,
}

/// One resource actually touched by recorded work.
#[non_exhaustive]
#[derive(Clone)]
pub enum ResourceUse {
    /// A buffer-range access.
    Buffer(BufferUse),
    /// A texture-subresource access.
    Texture(TextureUse),
    /// An acquired-frame attachment access.
    Frame(FrameAttachmentUse),
}

fn write_bit_names(
    formatter: &mut core::fmt::Formatter<'_>,
    bits: u32,
    names: &[(u32, &str)],
) -> core::fmt::Result {
    let mut separator = "";
    let mut any = false;
    for &(bit, name) in names {
        if bits & bit != 0 {
            formatter.write_str(separator)?;
            formatter.write_str(name)?;
            separator = "|";
            any = true;
        }
    }
    if any {
        Ok(())
    } else {
        formatter.write_str("<none>")
    }
}

/// A buffer-range use.
pub(crate) fn buffer_use(
    buffer: &Buffer,
    range: BufferRange,
    stages: PipelineScope,
    access: AccessMask,
) -> ResourceUse {
    ResourceUse::Buffer(BufferUse {
        buffer: buffer.clone(),
        range,
        stages,
        access,
    })
}

/// A texture-subresource use.
pub(crate) fn texture_use(
    texture: &Texture,
    subresources: TextureSubresourceRange,
    stages: PipelineScope,
    access: AccessMask,
    intent: TextureUseIntent,
) -> ResourceUse {
    ResourceUse::Texture(TextureUse {
        texture: texture.clone(),
        subresources,
        stages,
        access,
        intent,
    })
}

/// A texture use covering exactly what a view sees.
pub(crate) fn texture_use_of_view(
    view: &TextureView,
    stages: PipelineScope,
    access: AccessMask,
    intent: TextureUseIntent,
) -> ResourceUse {
    let descriptor = view.descriptor();
    texture_use(
        view.texture(),
        TextureSubresourceRange {
            aspects: descriptor.aspects,
            base_mip: descriptor.base_mip,
            mip_count: descriptor.mip_count,
            base_layer: descriptor.base_layer,
            layer_count: descriptor.layer_count,
        },
        stages,
        access,
        intent,
    )
}

/// A frame use.
///
/// A frame is not a texture, so it records through its own variant and its own
/// [`crate::api::presentation::AcquiredFrameId`]; there is no subresource range,
/// because a caller does not choose which layer of an acquired image it renders
/// into.
pub(crate) fn frame_use(
    frame: &FrameAttachment,
    stages: PipelineScope,
    access: AccessMask,
) -> ResourceUse {
    ResourceUse::Frame(FrameAttachmentUse {
        frame: frame.frame_id(),
        stages,
        access,
    })
}

/// The pipeline scope a binding's stage visibility maps onto.
///
/// [`ShaderStages`] and [`PipelineScope`] are not the same set: the first names
/// shader stages, and the second adds `COPY`, which has no shader stage but does
/// have a scope (section 37 explains why the two are separate types). The three
/// shader stages map one to one; `COPY` cannot be named by a binding, so it is
/// not reachable from here.
///
/// `None` means the visibility set is empty, which section 21 refuses when a
/// layout is created. It is reported rather than defaulted, because guessing a
/// stage for a slot no stage can see would put a use in the record that no shader
/// performed.
pub(crate) fn pipeline_scope_of(stages: ShaderStages) -> Option<PipelineScope> {
    let mut scope: Option<PipelineScope> = None;
    for (bit, mask) in [
        (PipelineScope::VERTEX, ShaderStages::VERTEX),
        (PipelineScope::FRAGMENT, ShaderStages::FRAGMENT),
        (PipelineScope::COMPUTE, ShaderStages::COMPUTE),
    ] {
        if stages.contains(mask) {
            scope = Some(match scope {
                Some(existing) => existing.union(bit),
                None => bit,
            });
        }
    }
    scope
}

/// The access a binding kind implies at a draw or dispatch.
///
/// Section 37.2's table, in the same order: a uniform buffer is read, a storage
/// buffer is read or read-written, a sampled texture is read, and a storage
/// texture follows its own access. The function returns `None` for a sampler,
/// which generates no memory hazard.
fn access_of_kind(kind: &BindingKind) -> Option<AccessMask> {
    match kind {
        BindingKind::UniformBuffer { .. } => Some(AccessMask::UNIFORM_READ),
        BindingKind::StorageBuffer { access, .. } => Some(match access {
            BufferBindingAccess::ReadOnly => AccessMask::SHADER_READ,
            BufferBindingAccess::ReadWrite => {
                AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE)
            }
        }),
        BindingKind::SampledTexture { .. } => Some(AccessMask::SHADER_READ),
        BindingKind::StorageTexture { access, .. } => Some(match access {
            StorageAccess::ReadOnly => AccessMask::SHADER_READ,
            StorageAccess::WriteOnly => AccessMask::SHADER_WRITE,
            StorageAccess::ReadWrite => AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE),
        }),
        BindingKind::Sampler { .. } => None,
    }
}

/// The texture role a binding kind implies.
///
/// The intent vocabulary has no write-only variant, so a write-only storage
/// texture reports [`TextureUseIntent::ShaderReadWrite`]: the role of a storage
/// image is read-write as far as attachment and copy compatibility are concerned,
/// and the exact access is in the mask beside it.
///
/// Only texture kinds reach this: `access_of_kind` answers `None` for a sampler,
/// and a buffer kind filled with a texture resource is refused as a mismatch
/// before this is read. The buffer and sampler arms therefore exist to make the
/// function total, and they return the intent a texture read would have.
fn intent_of_kind(kind: &BindingKind) -> TextureUseIntent {
    match kind {
        BindingKind::SampledTexture { .. } => TextureUseIntent::ShaderRead,
        BindingKind::StorageTexture { .. } => TextureUseIntent::ShaderReadWrite,
        BindingKind::UniformBuffer { .. }
        | BindingKind::StorageBuffer { .. }
        | BindingKind::Sampler { .. } => TextureUseIntent::ShaderRead,
    }
}

/// Whether a binding kind is a buffer kind or a texture kind.
///
/// Used to refuse a mismatch between a slot's declared kind and the resource
/// that filled it. Creation already refuses one
/// ([`crate::api::binding::group::validate_bind_group_descriptor`]), so a mismatch here
/// would mean the group was built by a path that skipped that check — which is
/// exactly the case worth reporting rather than guessing through.
fn expects_buffer(kind: &BindingKind) -> bool {
    match kind {
        BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. } => true,
        BindingKind::SampledTexture { .. }
        | BindingKind::StorageTexture { .. }
        | BindingKind::Sampler { .. } => false,
    }
}

/// The uses one bound bind group produces at a draw or dispatch.
///
/// Section 37.2's whole rule: the source is the pipeline's interface, the bound
/// group, and the dynamic offsets — **not** every resource in the group. The
/// group's own layout is the interface after creation validated them against each
/// other, so walking the group's slots walks exactly the slots the shader can
/// see, and a slot whose kind is [`BindingKind::Sampler`] produces no record.
///
/// Every element of a `Fixed(n)` array binding is reported, conservatively and on
/// purpose; see the module documentation.
pub(crate) fn bound_group_uses(group: &BindGroup) -> RhiResult<Vec<ResourceUse>> {
    let layout = group.layout();
    let entries = &group.descriptor().entries;
    let mut uses = Vec::new();

    for slot in &layout.descriptor().entries {
        let Some(entry) = entries.iter().find(|entry| entry.slot == slot.slot) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group slot {} is declared but no resource was bound to it",
                    slot.slot.get()
                ),
            ));
        };

        let Some(access) = access_of_kind(&slot.kind) else {
            // A sampler. Section 37.2: no memory hazard, so no use record.
            continue;
        };
        let Some(stages) = pipeline_scope_of(slot.visibility) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group slot {} declares no stage visibility, so no stage can use it",
                    slot.slot.get()
                ),
            ));
        };
        let intent = intent_of_kind(&slot.kind);
        let wants_buffer = expects_buffer(&slot.kind);

        match &entry.resource {
            BindingResource::Buffer(binding) => {
                if !wants_buffer {
                    return Err(mismatched(slot.slot.get()));
                }
                uses.push(buffer_use(&binding.buffer, binding.range, stages, access));
            }
            BindingResource::BufferArray(bindings) => {
                if !wants_buffer {
                    return Err(mismatched(slot.slot.get()));
                }
                for binding in bindings {
                    uses.push(buffer_use(&binding.buffer, binding.range, stages, access));
                }
            }
            BindingResource::Texture(view) => {
                if wants_buffer {
                    return Err(mismatched(slot.slot.get()));
                }
                uses.push(texture_use_of_view(view, stages, access, intent));
            }
            BindingResource::TextureArray(views) => {
                if wants_buffer {
                    return Err(mismatched(slot.slot.get()));
                }
                for view in views {
                    uses.push(texture_use_of_view(view, stages, access, intent));
                }
            }
            BindingResource::Sampler(_) | BindingResource::SamplerArray(_) => {
                return Err(mismatched(slot.slot.get()));
            }
        }
    }

    Ok(uses)
}

/// Refuses a slot filled with the wrong kind of resource.
fn mismatched(slot: u32) -> RhiError {
    RhiError::new(
        RhiErrorKind::InvalidUsage,
        format!(
            "bind group slot {} was filled with a resource of a different kind than the slot \
             declares",
            slot
        ),
    )
}

/// Refuses dynamic offsets that section 22.4 does not allow.
///
/// Section 32.3 and section 33 both list "dynamic offsets valid" as a draw and
/// dispatch check, and the same rule is re-checked when a group is bound so that
/// the caller is told at the verb that supplied the offsets. One function, because
/// the two paths must not disagree about what "valid" means — and that is every
/// rule of section 22.4 this side can decide, not just the count:
///
/// ```text
/// dynamic_offsets.len() == layout.dynamic_offset_count()   the count check below
/// each offset is u32                                       the parameter's own type
/// each offset satisfies its buffer-kind alignment          not decidable here; see below
/// effective_offset + range.size <= buffer.size             the range check below
/// ```
///
/// **"Each offset is `u32`" is settled by the signature rather than by a check.**
/// The parameter is `&[u32]`, so an offset that is not representable as a `u32` is
/// not a value a caller can state; there is nothing left for a runtime check to
/// refuse, and a wider parameter would have to be narrowed here instead.
///
/// The offsets are paired with their bindings in section 21.3's consumption order
/// — ascending [`BindingSlotId`](crate::api::binding::BindingSlotId), then
/// ascending element index within one `Fixed(n)` binding — which is the same order
/// [`BindGroupLayout::dynamic_offset_count`](crate::api::binding::BindGroupLayout::dynamic_offset_count)
/// counts. Walking both in that order is what stops an offset from being measured
/// against a neighbouring binding.
///
/// # The alignment rule is not decided here, and that is a known gap
///
/// "validated against corresponding buffer-kind alignment" compares an offset
/// against `MinUniformBufferOffsetAlignment` or
/// `MinStorageBufferOffsetAlignment` — device limits. The recording path holds a
/// [`DeviceIdentity`](crate::api::identity::DeviceIdentity), not a device, and
/// [`crate::api::binding::group::BindGroupLimits`], the shape that check needs, is
/// filled only by the device façade (its own `expect` reason records that), so
/// there is no number here to compare against. A portable default would refuse
/// offsets a real device allows, so the rule is left undone rather than guessed:
/// an unaligned offset is caught by the backend or the driver, not by this layer.
/// Closing it means giving the recording path the two numbers.
pub(crate) fn require_valid_dynamic_offsets(
    index: BindGroupIndex,
    group: &BindGroup,
    dynamic_offsets: &[u32],
) -> RhiResult<()> {
    let layout = group.layout();
    let expected = layout.dynamic_offset_count() as usize;
    if dynamic_offsets.len() != expected {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the bind group at {} has {} dynamic-offset bindings but {} offsets were supplied",
                index.get(),
                expected,
                dynamic_offsets.len()
            ),
        ));
    }

    // Section 21.3's consumption order, read off the layout's own canonical
    // entries: the offsets carry no slot of their own, so the layout is the only
    // thing that says which binding each one moves.
    let mut moved = Vec::with_capacity(expected);
    for slot in &layout.descriptor().entries {
        if !slot.dynamic_offset {
            continue;
        }
        let Some(entry) = group
            .descriptor()
            .entries
            .iter()
            .find(|entry| entry.slot == slot.slot)
        else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "bind group slot {} is declared but no resource was bound to it",
                    slot.slot.get()
                ),
            ));
        };
        match &entry.resource {
            BindingResource::Buffer(binding) => moved.push(binding),
            BindingResource::BufferArray(bindings) => moved.extend(bindings.iter()),
            // A layout refuses a dynamic offset on a non-buffer kind (section
            // 20.5), so this can only be reached by a group built through a path
            // that skipped the packet validator, which is exactly the case worth
            // reporting rather than walking past.
            _ => return Err(mismatched(slot.slot.get())),
        }
    }

    // The packet validator refuses a `Fixed(n)` binding filled with a
    // different number of elements, so this can only disagree for a group that
    // skipped it. Refused rather than zipped, because a short list would leave
    // the offsets past its end unmeasured — a fail open in the one function whose
    // job is to fail closed.
    if moved.len() != dynamic_offsets.len() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "the bind group at {} declares {} dynamic-offset elements but fills {} of them",
                index.get(),
                dynamic_offsets.len(),
                moved.len()
            ),
        ));
    }

    for (binding, offset) in moved.iter().zip(dynamic_offsets) {
        // Section 22.4's effective offset, with section 12.4's no-overflow rule
        // folded in: the range arithmetic is the buffer module's own
        // [`BufferRange::end`], so the effective range is measured by the code
        // that measured the static one at group creation rather than by a second
        // copy of it.
        let effective_end = binding
            .range
            .offset
            .checked_add(u64::from(*offset))
            .and_then(|effective| BufferRange::new(effective, binding.range.size).end());
        let buffer_size = binding.buffer.descriptor().size;
        match effective_end {
            Some(end) if end <= buffer_size => {}
            _ => {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "a binding of {} bytes based at {} with a dynamic offset of {} does not \
                         fit inside its buffer of {buffer_size} bytes at bind group {}",
                        binding.range.size,
                        binding.range.offset,
                        offset,
                        index.get()
                    ),
                ));
            }
        }
    }

    Ok(())
}

/// Checks the bound groups a pipeline interface says are used.
///
/// Section 32.3's and section 33's shared requirement: "every group actually used
/// by the shader is bound and layout-compatible, with valid dynamic offsets". The
/// two scopes differ only in what they can *do* with a group, not in what makes one
/// usable, so the check lives once.
///
/// "Actually used" is read off the pipeline's interned [`PipelineInterface`], whose
/// group list is the contract pipeline creation validated the shader against. A
/// group whose layout has no entries sees no resource and therefore needs no
/// binding, which is why an empty layout is skipped rather than required.
pub(crate) fn validate_bound_groups(
    interface: &PipelineInterface,
    groups: &[BoundGroup],
) -> RhiResult<()> {
    for (position, layout) in interface.descriptor().groups.iter().enumerate() {
        if layout.descriptor().entries.is_empty() {
            continue;
        }
        let index = BindGroupIndex::new(position as u32);
        let Some(bound) = groups.iter().find(|group| group.index == index) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "the pipeline uses bind group {} and this scope has none bound there",
                    index.get()
                ),
            ));
        };
        if bound.group.layout().compatibility_id() != layout.compatibility_id() {
            return Err(RhiError::new(
                RhiErrorKind::IncompatibleInterface,
                format!(
                    "the bind group bound at {} is not layout-compatible with the pipeline's",
                    index.get()
                ),
            ));
        }
        require_valid_dynamic_offsets(index, &bound.group, &bound.dynamic_offsets)?;
    }
    Ok(())
}

/// The uses one copy-family command produces.
///
/// Section 34.6: "a Copy-family command is itself an actual resource-use point."
/// The direction of each pair is what the verb means — a source is read and a
/// destination is written — and the scope is always
/// [`PipelineScope::COPY`], which
/// is the one scope no shader stage can name and therefore the one that makes a
/// copy distinguishable from a shader access in the merged summary.
///
/// The intent is the copy role rather than an attachment role
/// ([`TextureUseIntent::CopySrc`] / [`TextureUseIntent::CopyDst`]), because
/// section 37 keeps attachment compatibility and copy compatibility separate: a
/// texture that may be a copy destination is not thereby an attachment, and the
/// intent is what a later check reads to tell them apart.
pub(crate) fn copy_uses(copy: &CopyRecord) -> Vec<ResourceUse> {
    match copy {
        CopyRecord::Buffer(copy) => vec![
            buffer_use(
                &copy.src,
                BufferRange::new(copy.src_offset, copy.size),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
            ),
            buffer_use(
                &copy.dst,
                BufferRange::new(copy.dst_offset, copy.size),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
            ),
        ],

        CopyRecord::BufferToTexture(copy) => vec![
            buffer_use(
                &copy.buffer,
                BufferRange::new(copy.buffer_offset, texel_copy_span(copy)),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
            ),
            texture_use(
                &copy.texture,
                subresource_of(&copy.texture_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
                TextureUseIntent::CopyDst,
            ),
        ],

        CopyRecord::TextureToBuffer(copy) => vec![
            texture_use(
                &copy.texture,
                subresource_of(&copy.texture_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
                TextureUseIntent::CopySrc,
            ),
            buffer_use(
                &copy.buffer,
                BufferRange::new(copy.buffer_offset, texel_copy_span(copy)),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
            ),
        ],

        CopyRecord::Texture(copy) => vec![
            texture_use(
                &copy.src,
                subresource_of(&copy.src_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
                TextureUseIntent::CopySrc,
            ),
            texture_use(
                &copy.dst,
                subresource_of(&copy.dst_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
                TextureUseIntent::CopyDst,
            ),
        ],

        CopyRecord::Resolve(resolve) => vec![
            texture_use(
                &resolve.src,
                subresource_of(&resolve.src_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
                TextureUseIntent::ResolveSrc,
            ),
            texture_use(
                &resolve.dst,
                subresource_of(&resolve.dst_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
                TextureUseIntent::ResolveDst,
            ),
        ],

        CopyRecord::Blit(blit) => vec![
            texture_use(
                &blit.src,
                subresource_of(&blit.src_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_READ,
                TextureUseIntent::CopySrc,
            ),
            texture_use(
                &blit.dst,
                subresource_of(&blit.dst_subresource),
                PipelineScope::COPY,
                AccessMask::COPY_WRITE,
                TextureUseIntent::CopyDst,
            ),
        ],
    }
}

/// The tracking range one layer-and-mip selection names.
///
/// A copy names exactly one mip level and a run of array layers, so the range's
/// `mip_count` is always 1: a copy that spanned two mips would be two copies, and
/// section 34.3 does not define a multiplane one. The aspect becomes a one-bit
/// [`TextureAspects`](crate::api::resource::subresource::TextureAspects) so that a
/// depth-only copy does not claim to touch stencil.
fn subresource_of(layers: &TextureSubresourceLayers) -> TextureSubresourceRange {
    TextureSubresourceRange {
        aspects: aspect_bits(layers.aspect),
        base_mip: layers.mip_level,
        mip_count: 1,
        base_layer: layers.base_layer,
        layer_count: layers.layer_count,
    }
}

/// How many bytes of the buffer side a buffer-image copy addresses.
///
/// Section 34.2's footprint: one image is `bytes_per_row` by `rows_per_image`,
/// and a copy carries as many images as it has array layers — or, for a 3D
/// texture, as many as its Z extent, because section 34.2 addresses 3D slices
/// through `origin`/`extent` rather than as array subresources. The distinction is
/// why the dimension is read rather than assumed.
fn texel_copy_span(copy: &BufferTextureCopy) -> u64 {
    let images = match copy.texture.descriptor().dimension {
        TextureDimension::D3 => u64::from(copy.extent.depth),
        TextureDimension::D1 | TextureDimension::D2 => {
            u64::from(copy.texture_subresource.layer_count)
        }
    };
    u64::from(copy.bytes_per_row) * u64::from(copy.rows_per_image) * images
}
