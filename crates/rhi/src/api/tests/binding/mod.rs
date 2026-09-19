//! Bind group and bind group layout contract tests (specification sections 20
//! through 22).
//!
//! Every rule this chapter states is a rule about *values*: a slot's visibility,
//! a binding kind, a declared count, a buffer range, a view's dimension. None of
//! them needs hardware, which is why the two validators take the device's binding
//! answers as parameters and why the tests below can drive every accept and
//! reject path with stand-in answers.
//!
//! The convention from `tests/resource.rs` carries over unchanged: a test asserts
//! the exact [`RhiErrorKind`], because the kind is the part a caller may branch
//! on, and objects are assembled through the crate-private constructors that the
//! device verbs will call.
//!
//! One rule of the chapter is not tested here because the type system already
//! holds it: section 20.5 requires a slot's visibility to be non-empty, and
//! [`ShaderStages`] has no empty constructor — the three stage constants and
//! `union` are its whole surface, so an empty set is unrepresentable. The
//! validator still checks it, because a layout may arrive from a backend or a
//! future stage mapping rather than from this crate's own constants.
//!
//! # Files
//!
//! One section of the chapter per file, mirroring the `api/binding` split, so
//! that each file states its own rules and no file states two sections' rules at
//! once:
//!
//! ```text
//! mod.rs         the fixtures every section drives, and nothing else
//! vocabulary.rs  section 20.1-20.2, indices and counts
//! layout.rs      sections 20.5 and 21.2, layout validation and canonicalization
//! group.rs       section 22, the packet
//! ```
//!
//! Keeping the fixtures in `mod.rs` is what lets each section file say only what
//! the section says: `use super::*` brings them in, exactly as `tests/pipeline`
//! does.

use crate::api::binding::group::{BindGroupLimits, validate_bind_group_descriptor};
use crate::api::binding::layout::validate_bind_group_layout_descriptor;
use crate::api::binding::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayout,
    BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor, BindingCount, BindingKind,
    BindingResource, BindingSlot, BindingSlotId, BindingSupport, BindingSupportQuery,
    BufferBindingAccess, LayoutFingerprint, SamplerKind, StorageAccess, TextureSampleType,
};
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::resource::buffer::{
    Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUsage,
};
use crate::api::resource::sampler::{CompareFunction, Sampler, SamplerDescriptor};
use crate::api::resource::texture::{Texture, TextureDescriptor, TextureUsage};
use crate::api::resource::view::{TextureView, TextureViewDescriptor, TextureViewDimension};
use crate::api::shader::ShaderStages;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn identity(instance: u64, generation: u64) -> DeviceIdentity {
    DeviceIdentity::new(
        DeviceInstanceId::new(instance),
        DeviceGeneration::new(generation),
    )
}

fn device() -> DeviceIdentity {
    identity(1, 1)
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

fn assert_kind(result: RhiResult<()>, expected: RhiErrorKind) {
    match result {
        Ok(()) => panic!("expected {expected}, but the operation was accepted"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

fn slot(value: u32) -> BindingSlotId {
    BindingSlotId::new(value)
}

/// The device answer that expresses every binding, so that only the rule under
/// test can refuse.
fn permissive(_: &BindingSupportQuery) -> BindingSupport {
    BindingSupport::Supported
}

/// Every format samples as `Float` and every storage access is available, which
/// keeps the tests that are about something else about something else.
fn sample_type_float(_: TextureFormat) -> Option<TextureSampleType> {
    Some(TextureSampleType::Float)
}

fn storage_access_available(_: TextureFormat, _: StorageAccess) -> bool {
    true
}

fn uniform_slot(value: u32, min_size: u64) -> BindingSlot {
    BindingSlot::new(
        slot(value),
        ShaderStages::VERTEX,
        BindingKind::UniformBuffer { min_size },
    )
}

fn layout_from(entries: Vec<BindingSlot>) -> BindGroupLayout {
    BindGroupLayout::new(
        object(10),
        device(),
        BindGroupLayoutDescriptor::new(entries).canonicalized(),
        BindGroupLayoutCompatibilityId::new(1),
        LayoutFingerprint([0; 32]),
    )
}

fn generous_limits() -> BindGroupLimits {
    BindGroupLimits::new(1 << 16, 1 << 20, 0, 0)
}

fn buffer(id: u64, size: u64, usage: BufferUsage) -> Buffer {
    Buffer::new(object(id), device(), BufferDescriptor::new(size, usage))
}

fn range_of(id: u64, offset: u64, size: u64, usage: BufferUsage) -> BindingResource {
    BindingResource::Buffer(BufferBinding::new(
        buffer(id, size * 4, usage),
        BufferRange { offset, size },
    ))
}

/// The one-slot packet shape almost every test needs: one entry at slot 0.
fn group_with(layout: BindGroupLayout, resource: BindingResource) -> BindGroupDescriptor {
    BindGroupDescriptor::new(layout).with_entry(BindGroupEntry::new(slot(0), resource))
}

fn sampled_view(id: u64, usage: TextureUsage, format: TextureFormat) -> TextureView {
    let texture = Texture::new(
        object(id),
        device(),
        TextureDescriptor::new_2d(4, 4, format, usage),
    );
    let descriptor = TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)
        .expect("a whole 2D view of a 2D texture");
    TextureView::new(object(id + 1000), device(), texture, descriptor)
}

fn sampler(id: u64, compare: Option<CompareFunction>) -> Sampler {
    let mut descriptor = SamplerDescriptor::new();
    if let Some(compare) = compare {
        descriptor = descriptor.with_compare(compare);
    }
    Sampler::new(object(id), device(), descriptor)
}

fn texture_slot(dimension: TextureViewDimension, sample_type: TextureSampleType) -> BindingSlot {
    BindingSlot::new(
        slot(0),
        ShaderStages::FRAGMENT,
        BindingKind::SampledTexture {
            dimension,
            sample_type,
            multisampled: false,
        },
    )
}

/// Drives the packet validator with the two permissive format answers.
fn check(desc: &BindGroupDescriptor) -> RhiResult<()> {
    validate_bind_group_descriptor(
        desc,
        generous_limits(),
        sample_type_float,
        storage_access_available,
    )
}

/// A layout with one uniform buffer slot, and the packet that fills it.
fn one_uniform_slot() -> (BindGroupLayout, BindGroupDescriptor) {
    let layout = layout_from(vec![uniform_slot(0, 64)]);
    let group = group_with(layout.clone(), range_of(1, 0, 64, BufferUsage::UNIFORM));
    (layout, group)
}

mod group;
mod layout;
mod vocabulary;
