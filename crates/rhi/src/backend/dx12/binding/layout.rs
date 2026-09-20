//! The shape of one layout's descriptor table, computed once and used by both
//! halves of the mapping.
//!
//! # Why this is one function and not two
//!
//! A root signature declares, per group, the ranges a descriptor table is made
//! of: `BaseShaderRegister`, `RegisterSpace`, `RangeType` and
//! `OffsetInDescriptorsFromTableStart`. A bind group then writes views into the
//! slots those ranges name. If the two sides derived that layout independently,
//! a disagreement would not be caught by Direct3D 12: when a shader carries no
//! embedded root signature, the driver does **not** check the pipeline's root
//! signature against the tables bound at dispatch, so a mismatch surfaces as a
//! wrong read rather than as an error. Computing the shape once, in the file that
//! both [`super::group`] and [`crate::backend::dx12::pipeline`] call, is what
//! makes "the two agree" a property of the code rather than of a review.
//!
//! # The two tables
//!
//! Direct3D 12 keeps sampler descriptors in a heap type that no CBV/SRV/UAV copy
//! can mix with, so a layout's slots are split into two tables — the view table
//! and the sampler table — and each group therefore has **two** root parameters,
//! at indices `2 * group` and `2 * group + 1`. The scheme is stated here because
//! it is a contract between the root signature and the dispatch lowering, and
//! neither of them owns it.
//!
//! # What this module does not own
//!
//! Any native call. A plan is a description; [`super::group`] writes the
//! descriptors it describes, and [`crate::backend::dx12::pipeline::interface`]
//! declares the ranges.

use crate::api::binding::{BindGroupLayoutDescriptor, BindingKind, BindingSlot, BindingSlotId};
use crate::backend::dx12::failure::Dx12Failure;

use super::vocabulary::{RegisterClass, class_of};

/// The root-parameter index a group's view table is set at.
///
/// Half of the two-table scheme the module doc describes. Both callers use this
/// function rather than the arithmetic, so the scheme has one spelling.
pub(crate) fn view_parameter(group: u32) -> u32 {
    group * 2
}

/// The root-parameter index a group's sampler table is set at.
pub(crate) fn sampler_parameter(group: u32) -> u32 {
    group * 2 + 1
}

/// How one layout's slots map onto the two descriptor tables.
///
/// Held by [`super::group`] for the length of its writes and rebuilt by
/// [`crate::backend::dx12::pipeline::interface`] for the root signature. It is
/// cheap to rebuild — a layout has a handful of slots — which is why it is
/// derived rather than stored on the portable layout, where it would be a native
/// concept in a portable type.
pub(crate) struct TablePlan {
    /// The CBV/SRV/UAV ranges, in table order.
    views: Vec<RangePlan>,
    /// The sampler ranges, in table order.
    samplers: Vec<RangePlan>,
    /// How many descriptors the view table needs.
    view_descriptors: u32,
    /// How many descriptors the sampler table needs.
    sampler_descriptors: u32,
}

/// One slot's range in one of the two tables.
#[derive(Clone)]
pub(crate) struct RangePlan {
    /// The logical slot this range serves, which is the register number the
    /// shader reads it at.
    pub(crate) slot: BindingSlotId,
    /// The kind, kept so a writer knows which view to build without looking the
    /// slot up again.
    pub(crate) kind: BindingKind,
    /// The register class, which decides `Create*View` and the range type.
    pub(crate) class: RegisterClass,
    /// Where this range starts, counted from the beginning of *its own* table.
    pub(crate) first: u32,
    /// How many descriptors it covers: one per array element.
    pub(crate) count: u32,
}

impl TablePlan {
    /// Reads a layout into the two tables it becomes.
    ///
    /// The entries arrive canonicalized — ascending by slot id, which section
    /// 22.1 requires and `BindGroupLayoutDescriptor::canonicalized` has already
    /// enforced by the time a layout exists — so the table order is the slot
    /// order and no sort happens here. Depending on that rather than re-sorting
    /// is deliberate: a second sort would be a second authority for what order a
    /// layout is in, and if the two ever disagreed the descriptor offsets would
    /// disagree with the root signature silently.
    ///
    /// # Errors
    ///
    /// [`Dx12Failure::Unsupported`] for the two slot shapes this ABI cannot
    /// express: a dynamic offset, and a sampler. Both are refusals about a
    /// *lowering that is not written* rather than about the request being
    /// illegal, which is why they are `Unsupported` and not `InvalidUsage`.
    pub(crate) fn of(layout: &BindGroupLayoutDescriptor) -> Result<Self, Dx12Failure> {
        let mut plan = Self {
            views: Vec::with_capacity(layout.entries.len()),
            samplers: Vec::new(),
            view_descriptors: 0,
            sampler_descriptors: 0,
        };
        for entry in &layout.entries {
            plan.push(entry)?;
        }
        Ok(plan)
    }

    /// Places one slot in whichever table its register class belongs to.
    fn push(&mut self, entry: &BindingSlot) -> Result<(), Dx12Failure> {
        // Checked before the class, because a dynamic offset is a property of the
        // *slot* and would have to change the root parameter's type rather than
        // its contents: a dynamic offset is byte-granular and a descriptor table
        // is not, so the lowering for one is a root descriptor and not a table
        // range. Refusing here means the pipeline is never created, so no legal
        // packet can reach a dispatch that would read the wrong bytes.
        if entry.dynamic_offset {
            return Err(Dx12Failure::Unsupported {
                what: "a binding slot with a dynamic offset",
                why: "this backend builds descriptor tables only, and a table is \
                      addressed in whole descriptors: a byte-granular offset needs \
                      a root CBV/SRV/UAV parameter, whose lowering is not written",
            });
        }
        let class = class_of(&entry.kind);
        let count = entry.count.elements();
        let (ranges, descriptors) = match class {
            // The sampler heap and its per-entry write are the lowering this
            // backend has not built. The refusal names both halves, because a
            // reader who reaches it needs to know that no caller can reach it
            // either: `Device::create_sampler` stops with `unimplemented!()`, so
            // no portable `Sampler` value exists to put in such a slot.
            RegisterClass::Sampler => {
                return Err(Dx12Failure::Unsupported {
                    what: "a sampler binding slot",
                    why: "the sampler heap and the per-entry sampler write are not \
                          written, and no caller can reach them anyway: \
                          Device::create_sampler stops with unimplemented!(), so no \
                          portable Sampler value exists to bind",
                });
            }
            RegisterClass::ConstantBuffer
            | RegisterClass::ShaderResource
            | RegisterClass::UnorderedAccess => (&mut self.views, &mut self.view_descriptors),
        };
        let range = RangePlan {
            slot: entry.slot,
            kind: entry.kind.clone(),
            class,
            first: *descriptors,
            count,
        };
        *descriptors += count;
        ranges.push(range);
        Ok(())
    }

    /// The CBV/SRV/UAV ranges, in table order.
    pub(crate) fn views(&self) -> &[RangePlan] {
        &self.views
    }

    /// The sampler ranges, in table order.
    pub(crate) fn samplers(&self) -> &[RangePlan] {
        &self.samplers
    }

    /// How many descriptors the view table needs.
    pub(crate) fn view_descriptors(&self) -> u32 {
        self.view_descriptors
    }

    /// How many descriptors the sampler table needs.
    pub(crate) fn sampler_descriptors(&self) -> u32 {
        self.sampler_descriptors
    }

    /// The range serving one slot, if this layout has it.
    pub(crate) fn range_for(&self, slot: BindingSlotId) -> Option<&RangePlan> {
        self.views
            .iter()
            .chain(self.samplers.iter())
            .find(|range| range.slot == slot)
    }
}
