//! Tests for the buffer binding domain: the shared fixture, the trace readers,
//! and the three contracts the domain is held to.
//!
//! Every test in this collection runs the domain against the mock provider
//! rather than a hand-written double, because what is being tested is the
//! *trace* -- which calls were emitted, how many, and in what order -- and a
//! double written for these tests would answer with whatever trace the test
//! expected.  The mock is the same provider the Layer 1 contract suites run
//! against, so a change in Layer 1's own call surface shows up here as a failing
//! trace rather than as a mirror that quietly agrees with itself.
//!
//! The collection is split by the contract each file pins, not by size, because
//! the three are independently reviewable: which requests are the same request
//! ([`mirror`]), what an invalidation must forget ([`invalidation`]), and what a
//! refused call carries out ([`failure`]).  Everything they share -- the
//! recorders, the buffer identities and the trace readers -- lives here, so no
//! file restates a fixture another file could drift from.

use super::super::counters::DomainCounters;
use super::*;
use crate::webgl2::api::tests::{compute_storage_snapshot, snapshot};
use crate::webgl2::api::{
    GlBufferDesc, GlBufferUsage, GlFamilyProfile, GlResourceApi, GlStorageBufferUsage, MockCall,
    MockComputeStorageApi, MockGlFamilyApi,
};

mod failure;
mod invalidation;
mod mirror;

/// A buffer descriptor that both roles can bind.
///
/// One buffer legitimately serves several consumption roles at once, so nothing
/// here is a shortcut: it is what lets a single deletion be watched reaching both
/// mirrors.
fn both_roles() -> GlBufferDesc {
    GlBufferDesc {
        size: 512,
        usage: GlBufferUsage::UNIFORM | GlBufferUsage::STORAGE,
    }
}

/// One storage range of `buffer`, read-only, aligned and non-empty.
fn range(buffer: BufferId) -> GlStorageBufferRange {
    GlStorageBufferRange {
        buffer,
        offset: 0,
        size: 256,
        usage: GlStorageBufferUsage::ReadOnly,
    }
}

/// The two recorders the two roles need, and two identities live in both.
///
/// The domain keeps one mirror per role, but the storage role's verb lives on a
/// trait the command-backend bound does not include, so no single mock
/// implements both bounds: the uniform role runs against a command recorder and
/// the storage role against the optional-domain wrapper.  Both recorders are
/// built from the same snapshot, so the *n*th allocation in each carries the
/// same identity -- which is what lets one domain object, the thing the machine
/// really holds, be reconciled against both bounds with one `BufferId`, and is
/// the only way to watch a single deletion reach both roles.  The coincidence is
/// asserted at construction rather than assumed.
struct Fixture {
    uniform: MockGlFamilyApi,
    storage: MockComputeStorageApi,
    first: BufferId,
    second: BufferId,
}

fn two_role_backend() -> Fixture {
    let mut uniform = MockGlFamilyApi::from_discovery(compute_storage_snapshot(false));
    let first = uniform
        .create_buffer_resource(both_roles())
        .expect("a buffer both roles can bind");
    let second = uniform
        .create_buffer_resource(both_roles())
        .expect("a second buffer both roles can bind");
    let mut inner = MockGlFamilyApi::from_discovery(compute_storage_snapshot(false));
    let stored_first = inner
        .create_buffer_resource(both_roles())
        .expect("a buffer both roles can bind");
    let stored_second = inner
        .create_buffer_resource(both_roles())
        .expect("a second buffer both roles can bind");
    assert_eq!(
        first, stored_first,
        "the two recorders share an identity space"
    );
    assert_eq!(
        second, stored_second,
        "the two recorders share an identity space"
    );
    let storage = MockComputeStorageApi::new(inner).expect("the snapshot proved storage buffers");
    Fixture {
        uniform,
        storage,
        first,
        second,
    }
}

/// A command recorder holding one buffer the uniform role can bind.
///
/// A uniform-only recorder is used wherever the storage role is not under test,
/// because the storage role's index space is narrower and a test that never
/// enters it should not depend on the optional capability being proved.
fn uniform_backend() -> (MockGlFamilyApi, BufferId) {
    let mut backend = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let buffer = backend
        .create_buffer_resource(GlBufferDesc {
            size: 512,
            usage: GlBufferUsage::UNIFORM,
        })
        .expect("a uniform buffer");
    (backend, buffer)
}

/// One uniform binding word, for an exact trace comparison.
fn uniform(index: u32, buffer: Option<BufferId>, offset: u32, size: u32) -> MockCall {
    MockCall::BindUniformBuffer {
        index,
        buffer,
        offset,
        size,
    }
}

/// One storage binding word, for an exact trace comparison.
fn storage(binding: u32, buffer: BufferId, offset: u64, size: u64) -> MockCall {
    MockCall::BindStorageBuffer {
        binding,
        buffer,
        offset,
        size,
    }
}

/// Every uniform binding word the recorder has, in order.
fn uniform_calls(backend: &MockGlFamilyApi) -> Vec<MockCall> {
    backend
        .calls()
        .iter()
        .filter(|call| matches!(call, MockCall::BindUniformBuffer { .. }))
        .cloned()
        .collect()
}

/// Where the storage trace currently ends.
///
/// The optional-domain wrapper has no `clear_calls` of its own, so a test that
/// needs "what this step emitted" marks the trace and reads from the mark.
fn storage_mark(api: &MockComputeStorageApi) -> usize {
    api.calls().len()
}

/// Every storage binding word recorded since `mark`, in order.
fn storage_calls_since(api: &MockComputeStorageApi, mark: usize) -> Vec<MockCall> {
    api.calls()[mark..]
        .iter()
        .filter(|call| matches!(call, MockCall::BindStorageBuffer { .. }))
        .cloned()
        .collect()
}

/// This domain's tallies, as one comparable value.
fn counters_for(counters: &StateCounters) -> DomainCounters {
    *counters.domain_counts(StateDomain::Buffers)
}
