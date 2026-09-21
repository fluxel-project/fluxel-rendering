//! Portable direct storage-buffer compute conformance workload.
//!
//! A fixture supplies the code-form-specific shader, pipeline interface and
//! descriptor-backed bind group.  The command sequence itself is deliberately
//! shared: bind, direct dispatch, readback, then async mapping-lease assertion.
//! That keeps a result difference evidence about lowering rather than two
//! backend-local test programs that merely look alike.

use crate::api::binding::{BindGroup, BindGroupIndex};
use crate::api::command::{ComputeScopeDescriptor, RecordedWork, RecorderDescriptor};
use crate::api::identity::Label;
use crate::api::pipeline::ComputePipeline;
use crate::api::platform::Device;
use crate::api::resource::buffer::{Buffer, BufferRange};
use crate::api::resource::transfer::{ReadbackRequest, ReadbackTicket, ReadbackViewData};

/// One direct compute recording, ready to move exactly once into a submission
/// batch. The ticket remains available after that move for the portable
/// async-readback assertion.
pub(crate) struct DirectComputeRecording {
    work: Option<RecordedWork>,
    pub(crate) ticket: ReadbackTicket,
}

impl DirectComputeRecording {
    /// Moves the single-use recording into a plan batch.
    pub(crate) fn take_work(&mut self) -> RecordedWork {
        self.work
            .take()
            .expect("direct compute recording was submitted more than once")
    }
}

/// Records `set_pipeline -> set_bind_group(0) -> dispatch(1, 1, 1) ->
/// buffer readback` using only portable RHI calls.
///
/// `output` is moved into the readback record. Thus after this returns a
/// backend fixture may drop all caller-side pipeline, bind-group, and resource
/// handles; retained `RecordedWork` ownership is what must keep native objects
/// alive until completion.
pub(crate) fn record_single_storage_buffer_compute(
    device: &Device,
    pipeline: &ComputePipeline,
    group: &BindGroup,
    output: Buffer,
    output_size: u64,
    label: &'static str,
) -> DirectComputeRecording {
    let mut recorder = device
        .create_recorder(&RecorderDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: recorder creation failed: {error}"));
    {
        let mut compute = recorder
            .begin_compute(&ComputeScopeDescriptor::new())
            .unwrap_or_else(|error| panic!("{label}: compute scope creation failed: {error}"));
        compute
            .set_pipeline(pipeline)
            .unwrap_or_else(|error| panic!("{label}: pipeline binding failed: {error}"));
        compute
            .set_bind_group(BindGroupIndex::new(0), group, &[])
            .unwrap_or_else(|error| panic!("{label}: bind-group binding failed: {error}"));
        compute
            .dispatch(1, 1, 1)
            .unwrap_or_else(|error| panic!("{label}: direct dispatch recording failed: {error}"));
        compute
            .end()
            .unwrap_or_else(|error| panic!("{label}: compute scope close failed: {error}"));
    }
    let ticket = recorder
        .encode_readback(ReadbackRequest::Buffer {
            label: Label(Some(format!("{label} output"))),
            src: output,
            range: BufferRange::new(0, output_size),
        })
        .unwrap_or_else(|error| panic!("{label}: output readback recording failed: {error}"));
    let work = recorder
        .finish()
        .unwrap_or_else(|error| panic!("{label}: recording completion failed: {error}"));
    DirectComputeRecording {
        work: Some(work),
        ticket,
    }
}

/// Maps a completed output ticket through the public RAII mapping lease and
/// compares its little-endian words with the fixture's shader-specific oracle.
pub(crate) async fn assert_direct_compute_output(
    ticket: &ReadbackTicket,
    expected: &[u32],
    label: &str,
) {
    let view = ticket
        .read()
        .await
        .unwrap_or_else(|error| panic!("{label}: compute readback failed: {error}"));
    let ReadbackViewData::Buffer { bytes } = view.data() else {
        panic!("{label}: compute buffer readback returned texture data");
    };
    assert_eq!(
        bytes.len() % 4,
        0,
        "{label}: compute output is not whole u32 words"
    );
    let actual = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("one u32")))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "{label}: direct compute output differs");
}
