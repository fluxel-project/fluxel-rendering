//! RenderGraph bridge contract tests (specification 04 §37, 04 §38.3, 06 §50,
//! and 06 §51).
//!
//! These are the review instrument for the bridge, not a conformance suite. The
//! bridge is the one chapter whose whole subject is *what crosses a boundary*, so
//! the question every test here asks is whether a caller on one side can say what
//! it means without learning the other side's vocabulary: a recorder reporting
//! what it touched, a graph compiler declaring what it intended, and a graph
//! asking what a transient resource needs.
//!
//! Three kinds of test are mixed here, and they are labelled where they appear:
//!
//! * **Behavioural tests** exercise the parts that are already real: the two
//!   bitset vocabularies, their `Display`, and `check_same_device`'s O(1)
//!   cross-device refusal. They run.
//! * **Shape tests** are ordinary functions compiled but never called, written as
//!   realistic call sites for the parts whose bodies are still `unimplemented!()`.
//!   They take the types that module 04 and module 05 own as *parameters* rather
//!   than constructing them, so that this file checks the interface it is about
//!   rather than a constructor signature of someone else's.
//! * **Seam tests** stand in for the other side of the boundary. A graph compiler
//!   lives outside this crate, so the tests that need a declaration or a packing
//!   plan build one through the crate-private constructor the port will use; the
//!   gap that leaves is recorded on each type rather than papered over here.
//!
//! Nothing in this file is GPU evidence. There is no device behind any of it.

use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::graph_bridge::{
    AccessMask, AllocationCompatibilityClass, AllocationRequirements, DeclaredContentContract,
    DeclaredWorkContract, FrameAttachmentUse, PipelineScope, ResourceUse, TextureUse,
    TextureUseIntent, TransientAllocationPlan, TransientAllocationService, TransientRealization,
    TransientResourceDesc, validate_recorded_work,
};
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, Label, ObjectId};
use crate::api::resource::buffer::{Buffer, BufferDescriptor, BufferRange, BufferUsage};
use crate::api::resource::subresource::{TextureAspects, TextureSubresourceRange};
use crate::api::resource::texture::{Texture, TextureDescriptor, TextureUsage};

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

fn buffer(size: u64) -> Buffer {
    Buffer::new(
        object(1),
        device(),
        BufferDescriptor::new(size, BufferUsage::UNIFORM),
    )
}

fn texture() -> Texture {
    Texture::new(
        object(2),
        device(),
        TextureDescriptor::new_2d(
            64,
            64,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        ),
    )
}

fn full_texture_range() -> TextureSubresourceRange {
    TextureSubresourceRange {
        aspects: TextureAspects::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    }
}

// ---------------------------------------------------------------------------
// Section 37 — the use vocabulary.
// ---------------------------------------------------------------------------

/// A use is visible to a *set* of stages, which is why this is a bitset and not
/// an enum: a uniform buffer read by both a vertex and a fragment shader is one
/// use with two stages, and section 37 has no combined variant for that pair.
#[test]
fn a_use_is_visible_to_a_set_of_stages() {
    let both = PipelineScope::VERTEX.union(PipelineScope::FRAGMENT);

    assert!(both.contains(PipelineScope::VERTEX));
    assert!(both.contains(PipelineScope::FRAGMENT));
    assert!(!both.contains(PipelineScope::COMPUTE));
    assert!(!both.contains(PipelineScope::COPY));

    // A stage contains itself, and a stage does not contain a different one.
    assert!(PipelineScope::COMPUTE.contains(PipelineScope::COMPUTE));
    assert!(!PipelineScope::COPY.contains(PipelineScope::COMPUTE));
}

/// `COPY` has no shader stage and does have a pipeline scope, which is how
/// section 37.2 accounts for recorder upload and readback: the use is a real
/// access by a real command, and no shader was involved in it.
#[test]
fn the_copy_domain_is_a_scope_without_a_shader_stage() {
    assert!(!PipelineScope::COPY.contains(PipelineScope::VERTEX));
    assert!(!PipelineScope::VERTEX.contains(PipelineScope::COPY));

    let upload = PipelineScope::COPY.union(PipelineScope::COPY);
    assert!(upload.contains(PipelineScope::COPY));
}

/// Access is a set for the same reason scope is: one use can be a read *and* a
/// write, and hazard lowering needs both bits at once. A color attachment read
/// and written by one draw is exactly that case.
#[test]
fn one_use_can_be_a_read_and_a_write_at_once() {
    let color_attachment = AccessMask::COLOR_READ.union(AccessMask::COLOR_WRITE);

    assert!(color_attachment.contains(AccessMask::COLOR_READ));
    assert!(color_attachment.contains(AccessMask::COLOR_WRITE));
    assert!(!color_attachment.contains(AccessMask::DEPTH_WRITE));

    // A read-write storage image is a different pair with the same shape, which
    // is why the *role* is `TextureUseIntent` and not this mask.
    let storage_image = AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE);
    assert!(storage_image.contains(AccessMask::SHADER_WRITE));
    assert!(!storage_image.contains(AccessMask::COLOR_WRITE));

    assert!(!color_attachment.contains(storage_image));
}

/// The mask carries the three bits the v1 closure corrections make uncomfortable,
/// and this test exists to keep them visible rather than to endorse them.
///
/// Section 37 writes `HOST_READ`, `HOST_WRITE`, and `PRESENT` at
/// `04-recording-resource-uses.md` L886–L888, and the same section's prose states
/// what they mean: the two host bits are reserved for graph and tooling
/// observation, and a recorder upload reports `COPY` with `COPY_WRITE` instead.
/// `PRESENT` is the recorded open question 04-05 U5 — a frame travels as
/// [`ResourceUse::Frame`], and [`AccessMask::PRESENT`] is transcribed because
/// section 37 writes it, not because anything emits it.
///
/// The assertion is about bit *distinctness*, which is the one property that
/// would be silently wrong: sixteen constants over a `u32` with a repeated bit
/// would make two unequal accesses compare equal.
#[test]
fn the_reserved_and_present_bits_are_distinct_from_every_access_bit() {
    let all = [
        AccessMask::VERTEX_READ,
        AccessMask::INDEX_READ,
        AccessMask::UNIFORM_READ,
        AccessMask::SHADER_READ,
        AccessMask::SHADER_WRITE,
        AccessMask::COLOR_READ,
        AccessMask::COLOR_WRITE,
        AccessMask::DEPTH_READ,
        AccessMask::DEPTH_WRITE,
        AccessMask::STENCIL_READ,
        AccessMask::STENCIL_WRITE,
        AccessMask::COPY_READ,
        AccessMask::COPY_WRITE,
        AccessMask::HOST_READ,
        AccessMask::HOST_WRITE,
        AccessMask::PRESENT,
    ];

    for (index, bit) in all.iter().enumerate() {
        assert!(
            all.iter()
                .enumerate()
                .all(|(other, candidate)| index == other || bit.union(*candidate) != *candidate),
            "bit {index} is not distinct from an earlier constant"
        );
    }
}

/// The recorder upload route section 37.2 requires, written the way the prose
/// says it must be written: `COPY` scope with a copy access, never a host bit.
#[test]
fn a_recorder_upload_uses_the_copy_scope_and_not_a_host_bit() {
    let upload = PipelineScope::COPY;
    let upload_access = AccessMask::COPY_WRITE;

    assert!(upload.contains(PipelineScope::COPY));
    assert!(!upload_access.contains(AccessMask::HOST_WRITE));

    let readback_access = AccessMask::COPY_READ;
    assert!(!readback_access.contains(AccessMask::HOST_READ));
    assert!(!readback_access.contains(AccessMask::COPY_WRITE));
}

/// A refused or uncovered use has to name *which* stage set and *which* access
/// were involved, and a raw `u32` cannot say it in a message a caller can act on.
///
/// Section 37 does not declare either `Display`; both are added as ergonomic
/// completion of the interface, on the precedent of
/// [`crate::api::resource::buffer::BufferUsage`]. This test is what keeps the
/// rendering honest.
///
/// The `<none>` branch is not testable from here and that is deliberate: neither
/// bitset has an empty constructor, because section 37 writes only the four stage
/// constants and the sixteen access constants plus `contains` and `union`, and a
/// use with no stage is not a use. The branch stays for the same reason the empty
/// case stays in `BufferUsage`'s `Display` — defensiveness costs one comparison
/// and a `Display` that panicked on an empty set would be a worse failure.
#[test]
fn a_use_names_its_stages_and_accesses_rather_than_its_bits() {
    assert_eq!(
        PipelineScope::VERTEX
            .union(PipelineScope::FRAGMENT)
            .to_string(),
        "VERTEX|FRAGMENT"
    );
    assert_eq!(PipelineScope::COMPUTE.to_string(), "COMPUTE");
    assert_eq!(PipelineScope::COPY.to_string(), "COPY");

    assert_eq!(
        AccessMask::SHADER_READ
            .union(AccessMask::SHADER_WRITE)
            .to_string(),
        "SHADER_READ|SHADER_WRITE"
    );
    assert_eq!(
        AccessMask::COLOR_READ
            .union(AccessMask::COLOR_WRITE)
            .to_string(),
        "COLOR_READ|COLOR_WRITE"
    );
    assert_eq!(AccessMask::COPY_WRITE.to_string(), "COPY_WRITE");
}

/// Coverage is a comparison of two portable descriptions, so a use record names
/// the range and the subresources it touched rather than only the resource:
/// section 38.3 checks buffer byte ranges and texture mip/layer/aspect ranges,
/// and a record that named only the object could not express a partial write.
#[test]
fn a_use_record_names_the_range_it_touched_not_just_the_resource() {
    let range = BufferRange::new(256, 512);
    assert_eq!(range.end(), Some(768));

    let use_ = crate::api::graph_bridge::BufferUse {
        buffer: buffer(4096),
        range,
        stages: PipelineScope::VERTEX,
        access: AccessMask::VERTEX_READ,
    };

    // Held by clone, so the record survives the caller dropping its own handle
    // (section 38.2). The clone here is that promise, one line long.
    let held = use_.clone();
    drop(use_);
    assert_eq!(held.range.end(), Some(768));
    assert_eq!(held.buffer.device_identity(), device());

    let texture_use = TextureUse {
        texture: texture(),
        subresources: full_texture_range(),
        stages: PipelineScope::FRAGMENT,
        access: AccessMask::COLOR_WRITE,
        intent: TextureUseIntent::ColorAttachment,
    };

    assert_eq!(texture_use.subresources.mip_count, 1);
    assert_eq!(texture_use.intent, TextureUseIntent::ColorAttachment);
    assert_ne!(
        texture_use.intent,
        TextureUseIntent::DepthStencilWrite,
        "the role, not the access mask, is what attachment compatibility is checked against"
    );
}

/// The same access mask means different roles, which is why the intent is a
/// separate field and why it is `#[non_exhaustive]`.
///
/// `COLOR_READ | COLOR_WRITE` is a color attachment and `SHADER_READ |
/// SHADER_WRITE` is a read-write storage image: identical in shape, different in
/// what attachment and copy-intent compatibility may be checked against.
#[test]
fn the_same_access_mask_serves_two_roles_that_are_not_interchangeable() {
    let attachment = TextureUseIntent::ColorAttachment;
    let storage = TextureUseIntent::ShaderReadWrite;

    assert_ne!(attachment, storage);

    // The nine roles section 37 lists, each distinct from the others.
    let roles = [
        TextureUseIntent::ShaderRead,
        TextureUseIntent::ShaderReadWrite,
        TextureUseIntent::ColorAttachment,
        TextureUseIntent::DepthStencilRead,
        TextureUseIntent::DepthStencilWrite,
        TextureUseIntent::CopySrc,
        TextureUseIntent::CopyDst,
        TextureUseIntent::ResolveSrc,
        TextureUseIntent::ResolveDst,
        TextureUseIntent::Present,
    ];

    for (index, role) in roles.iter().enumerate() {
        assert!(
            roles
                .iter()
                .enumerate()
                .all(|(other, candidate)| { index == other || role != candidate }),
            "role {index} repeats an earlier variant"
        );
    }

    // Copy source and destination are different intents, not a flag: a resolve is
    // its own pair on the same principle.
    assert_ne!(TextureUseIntent::CopySrc, TextureUseIntent::ResolveSrc);
    assert_ne!(TextureUseIntent::ResolveDst, TextureUseIntent::CopyDst);
}

/// A presentation frame is not an ordinary texture — the root specification makes
/// `FrameAttachment` neither a `Texture` nor a `TextureView` — so it enters the
/// use model as its own variant.
///
/// Compiled, never called: [`crate::api::presentation::AcquiredFrameId`] is minted
/// by the acquisition path module 05 owns, so this takes one as a parameter rather
/// than depending on a constructor signature this chapter does not fix. Taking it
/// as a parameter is also the more honest test: what section 37's `Frame` variant
/// promises is that a frame *is accepted here*, not that this file can make one.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_frame_enters_the_use_model_as_its_own_record(
    frame: crate::api::presentation::AcquiredFrameId,
) {
    let use_ = ResourceUse::Frame(FrameAttachmentUse {
        frame,
        stages: PipelineScope::FRAGMENT,
        access: AccessMask::COLOR_WRITE,
    });

    // A frame has no subresource range to name, because a caller does not choose
    // which layer of an acquired image it renders into — which is the structural
    // reason it cannot be a `TextureUse` with an empty range.
    match use_ {
        ResourceUse::Frame(frame_use) => {
            let _: crate::api::presentation::AcquiredFrameId = frame_use.frame;
        }
        ResourceUse::Buffer(_) | ResourceUse::Texture(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Section 38.3 — declared versus actual.
// ---------------------------------------------------------------------------

/// A declaration may cover more than the recording used and may not cover less.
/// This is the seam test for that rule: it builds the conservative declaration
/// section 38.3 permits and checks that the shape a caller has to write is the
/// shape the rule needs.
///
/// The comparison itself is not run — [`validate_recorded_work`] panics until
/// module 04's recording sequence exists — so what this test reviews is whether a
/// graph compiler can *express* a conservative declaration at all, which it can:
/// `uses` is a plain `Vec<ResourceUse>` and a declaration adds a range without
/// having to mark it as unused.
#[test]
fn a_declaration_may_cover_more_than_the_recording_used() {
    let declared = DeclaredWorkContract {
        pass_label: Label(Some("shadow pass".to_owned())),
        uses: vec![ResourceUse::Buffer(crate::api::graph_bridge::BufferUse {
            buffer: buffer(4096),
            // The whole buffer, where the recording will touch 512 bytes of it.
            range: BufferRange::new(0, 4096),
            stages: PipelineScope::VERTEX.union(PipelineScope::FRAGMENT),
            access: AccessMask::UNIFORM_READ,
        })],
        content: DeclaredContentContract::new(),
    };

    assert_eq!(declared.pass_label.as_deref(), Some("shadow pass"));
    assert_eq!(declared.uses.len(), 1);

    // The declaration's range is strictly larger than what the recording touched,
    // and nothing in the shape refuses that — which is the rule, not an omission.
    match &declared.uses[0] {
        ResourceUse::Buffer(use_) => {
            assert_eq!(use_.range, BufferRange::new(0, 4096));
            assert!(use_.stages.contains(PipelineScope::FRAGMENT));
            assert!(use_.stages.contains(PipelineScope::VERTEX));
        }
        ResourceUse::Texture(_) | ResourceUse::Frame(_) => {
            panic!("a buffer use was declared as something else")
        }
    }
}

/// The declared-vs-actual entry point, written as a graph compiler calls it.
///
/// Compiled, never called: [`crate::api::command::RecordedWork`] is module 04's
/// type and does not exist yet, so this file takes it as a parameter instead of
/// constructing one. It is the call site the interface has to serve, and it is
/// the reason `validate_recorded_work` is a free function rather than a method:
/// neither the recording nor the declaration owns the check — the boundary does.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_declared_versus_actual(
    work: &crate::api::command::RecordedWork,
    declared: &DeclaredWorkContract,
) -> RhiResult<()> {
    validate_recorded_work(work, declared)
}

/// Section 38.3 fails a cross-device comparison immediately, and that half is O(1)
/// and needs no backend, so it is real here while the comparison it belongs to is
/// not.
///
/// This is the one part of the declared-vs-actual check that is behaviourally
/// testable today, and it is tested for the exact kind rather than for "an
/// error": a caller branches on [`RhiErrorKind::WrongDevice`] to decide to
/// re-record, and on nothing else.
#[test]
fn a_declaration_and_a_recording_from_different_devices_are_refused_at_once() {
    let recorded = identity(1, 1);
    let declared = identity(2, 1);

    let result = crate::api::graph_bridge::check_same_device(recorded, declared);
    match result {
        Ok(()) => panic!("expected WrongDevice, but two devices compared equal"),
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::WrongDevice),
    }

    // A generation bump is a different device by section 3.3's rule that identity
    // is instance *and* generation, which is what makes device loss recoverable
    // without stale handles resolving.
    let stale = identity(1, 1);
    let current = identity(1, 2);
    assert!(crate::api::graph_bridge::check_same_device(stale, current).is_err());

    let same = identity(1, 1);
    assert!(crate::api::graph_bridge::check_same_device(same, recorded).is_ok());
}

// ---------------------------------------------------------------------------
// Section 51 — transient allocation.
// ---------------------------------------------------------------------------

/// A transient is described exactly as it would be created, because the question
/// is asked before anything exists.
///
/// The enum is not `#[non_exhaustive]` and carries no label and no device: the
/// graph is expected to match both arms exhaustively, and a transient is
/// described by what it would be created as and nothing else.
#[test]
fn a_transient_is_described_exactly_as_it_would_be_created() {
    let as_buffer =
        TransientResourceDesc::Buffer(BufferDescriptor::new(1 << 16, BufferUsage::STORAGE));
    let as_texture = TransientResourceDesc::Texture(TextureDescriptor::new_2d(
        2048,
        2048,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    match as_buffer {
        TransientResourceDesc::Buffer(desc) => assert_eq!(desc.size, 1 << 16),
        TransientResourceDesc::Texture(_) => panic!("a buffer described itself as a texture"),
    }

    match as_texture {
        TransientResourceDesc::Texture(desc) => {
            assert_eq!(desc.extent.width, 2048);
            assert_eq!(desc.mip_levels, 1);
        }
        TransientResourceDesc::Buffer(_) => panic!("a texture described itself as a buffer"),
    }
}

/// The requirements answer is a set of physical placement facts, and every one of
/// them is target-specific: no portable formula derives alignment or a
/// compatibility class from a descriptor, which is exactly why section 51 makes
/// the graph ask instead of compute.
#[test]
fn allocation_requirements_are_target_facts_and_a_preference_is_not_a_rule() {
    let requirements = AllocationRequirements {
        size: 1 << 20,
        alignment: 256,
        compatibility_class: AllocationCompatibilityClass::new(3),
        prefers_dedicated: true,
    };

    // Equality is the whole interface of the class: two resources whose classes
    // are equal are placeable in the same memory, and the graph never needs to
    // know why.
    assert_eq!(
        requirements.compatibility_class,
        AllocationCompatibilityClass::new(3)
    );
    assert_ne!(
        requirements.compatibility_class,
        AllocationCompatibilityClass::new(4)
    );

    // `prefers_dedicated` is a preference: a graph that ignores it still gets a
    // legal realization, which is why it is a `bool` and not a fourth class.
    let ignored = AllocationRequirements {
        prefers_dedicated: false,
        ..requirements
    };
    assert_eq!(ignored.size, requirements.size);
}

/// The service seam, implemented in-crate.
///
/// This test is a stand-in for the graph compiler, which lives outside this crate
/// and therefore could not be written here at all: `AllocationCompatibilityClass`
/// and `TransientRealization` have no public constructor, on the grounds that a
/// caller which could mint a class could pack resources the target cannot share.
/// The in-crate implementation below is exactly the set of constructors a real
/// service implementation needs, which is the gap the type documentation records.
///
/// What it *does* review is the shape of the trait: whether `realize` taking
/// `&mut self` is usable behind a `dyn` reference, whether a service can be held
/// across calls, and whether the two calls compose into the exchange section 51
/// describes — ask, decide, realize.
#[test]
fn the_service_seam_asks_then_realizes() {
    struct RecordingService {
        asked: Vec<u64>,
        realized: usize,
    }

    impl TransientAllocationService for RecordingService {
        fn requirements(&self, desc: &TransientResourceDesc) -> RhiResult<AllocationRequirements> {
            let size = match desc {
                TransientResourceDesc::Buffer(buffer) => buffer.size,
                TransientResourceDesc::Texture(texture) => {
                    u64::from(texture.extent.width) * u64::from(texture.extent.height)
                }
            };
            Ok(AllocationRequirements {
                size,
                alignment: 256,
                compatibility_class: AllocationCompatibilityClass::new(1),
                prefers_dedicated: false,
            })
        }

        fn realize(&mut self, _plan: &TransientAllocationPlan) -> RhiResult<TransientRealization> {
            self.realized += 1;
            Ok(TransientRealization::new())
        }
    }

    let mut service = RecordingService {
        asked: Vec::new(),
        realized: 0,
    };

    let desc = TransientResourceDesc::Buffer(BufferDescriptor::new(4096, BufferUsage::STORAGE));
    let requirements = service
        .requirements(&desc)
        .expect("a buffer is describable");
    assert_eq!(requirements.size, 4096);
    service.asked.push(requirements.size);

    // Behind a trait object, which is how the graph will hold a service it was
    // handed rather than one it named — the reason `realize` takes `&mut self`
    // and the reason this call goes through a reborrow.
    let dynamic: &mut dyn TransientAllocationService = &mut service;
    dynamic
        .realize(&TransientAllocationPlan::new())
        .expect("a service realizes the packing it was given");

    assert_eq!(service.realized, 1);
    assert_eq!(service.asked, vec![4096]);
}

/// Both ends of the seam print their own name and an ellipsis, and nothing else.
///
/// Adjudication A16: an opaque handle gets a hand-written `Debug` that prints
/// portable identity only. A plan holds the graph's packing representation and a
/// realization holds native allocations; neither is portable state, and neither
/// type carries a portable identity to show instead, so the whole output is the
/// type name and an ellipsis — the ellipsis being the load-bearing half, since a
/// bare name would assert there is nothing behind it.
#[test]
fn an_opaque_packing_and_an_opaque_realization_print_only_their_names() {
    let plan = format!("{:?}", TransientAllocationPlan::new());
    let realization = format!("{:?}", TransientRealization::new());

    // Pinned whole rather than by prefix and suffix: the ellipsis lives inside
    // the braces, so `ends_with("..")` would be false for every shape this
    // output could take, including `Name { field: value, .. }`. Comparing the
    // entire string is both stricter and the thing the impl actually promises.
    assert_eq!(plan, "TransientAllocationPlan { .. }");
    assert_eq!(realization, "TransientRealization { .. }");

    // Neither private field name reaches the output.
    assert!(!plan.contains("TransientPlanDomain"), "{plan}");
    assert!(
        !realization.contains("TransientRealizationDomain"),
        "{realization}"
    );
}
