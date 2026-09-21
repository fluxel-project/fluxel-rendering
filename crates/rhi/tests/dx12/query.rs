//! DX12 fixture for the portable empty-occlusion-query conformance case.
//!
//! Provider construction and the DX12 test executor remain private. The
//! resource creation, command recording, resolve, readback, and CPU assertion
//! are deliberately shared by every backend in `tests/common`.

use super::{block_on, copy_lane, portable_device};
use crate::api::submission::SubmissionPlanBuilder;

#[test]
fn empty_occlusion_query_fixture_runs_the_portable_case() {
    let device = portable_device();
    let mut common = crate::backend::conformance::record_empty_occlusion_query(
        &device,
        "DX12 empty occlusion query",
    );
    let mut plan = SubmissionPlanBuilder::new(&device);
    let point = plan
        .add_batch(copy_lane(&device), vec![common.take_work()])
        .expect("DX12 direct lane accepts raster, query, resolve, and readback work");
    let receipt = block_on(device.submit(plan.build().expect("plan"))).expect("submit");
    block_on(crate::backend::test_harness::require_complete(
        &device,
        receipt.completion_for(point).expect("batch completion"),
        "DX12 empty occlusion query",
    ));
    block_on(crate::backend::conformance::assert_empty_occlusion_result(
        &common.ticket,
        "DX12 empty occlusion query",
    ));
}
