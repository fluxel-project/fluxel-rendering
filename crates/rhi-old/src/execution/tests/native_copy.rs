//! Native copy witnesses.

use super::native_common::add_copy;
use super::native_compute::run_buffer_case;
use crate::*;
use fluxel_rendergraph::*;
pub(super) fn run_c01(backend: crate::Backend) {
    let size = 64;
    let mut graph = RenderGraph::new();
    let source_slot = graph.import_buffer_slot(
        "c01-source",
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let destination_slot = graph.import_buffer_slot(
        "c01-destination",
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let output = add_copy(
        &mut graph,
        "c01-copy",
        &source_slot.version,
        destination_slot.version,
        8,
        16,
        24,
    );
    let export = graph.export_buffer(
        output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph
        .compile(&CopyBackend::portable_capabilities())
        .unwrap()
        .graph;
    let source: Vec<u8> = (0..size as u8).collect();
    let mut expected = vec![0xCD; size as usize];
    expected[16..40].copy_from_slice(&source[8..32]);
    run_buffer_case(
        "C01",
        backend,
        &compiled,
        export,
        &[
            (source_slot.slot, source),
            (destination_slot.slot, vec![0xCD; size as usize]),
        ],
        &expected,
    );
}

pub(super) fn run_c03(backend: crate::Backend) {
    let size = 64;
    let mut graph = RenderGraph::new();
    let source_a = graph.import_buffer_slot(
        "c03-source-a",
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let source_b = graph.import_buffer_slot(
        "c03-source-b",
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let destination = graph.import_buffer_slot(
        "c03-destination",
        ImportBufferContract {
            descriptor: BufferDesc { size },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let after_a = add_copy(
        &mut graph,
        "c03-copy-a",
        &source_a.version,
        destination.version,
        0,
        0,
        32,
    );
    let after_b = add_copy(
        &mut graph,
        "c03-copy-b",
        &source_b.version,
        after_a,
        8,
        16,
        24,
    );
    let export = graph.export_buffer(
        after_b,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph
        .compile(&CopyBackend::portable_capabilities())
        .unwrap()
        .graph;
    let source_a_bytes = vec![0xA1; size as usize];
    let source_b_bytes = vec![0xB2; size as usize];
    let mut expected = vec![0xCD; size as usize];
    expected[0..32].copy_from_slice(&source_a_bytes[0..32]);
    expected[16..40].copy_from_slice(&source_b_bytes[8..32]);
    run_buffer_case(
        "C03",
        backend,
        &compiled,
        export,
        &[
            (source_a.slot, source_a_bytes),
            (source_b.slot, source_b_bytes),
            (destination.slot, vec![0xCD; size as usize]),
        ],
        &expected,
    );
}
