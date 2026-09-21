//! Portable logical-creation and empty-submission conformance workload.
//!
//! Provider loading, adapter selection, and queue-family/native-device setup
//! remain fixture responsibilities. Once a public `Device` exists, this module
//! checks the same base object path on every backend without naming one.

use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::platform::Device;
use crate::api::resource::buffer::{BufferDescriptor, BufferSupportQuery, BufferUsage};
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::texture::{TextureDescriptor, TextureUsage};
use crate::api::resource::view::{TextureViewDescriptor, TextureViewDimension};
use crate::api::submission::{SubmissionPlan, SubmissionPlanBuilder};
use crate::backend::test_harness::CaseOutcome;

/// Creates ordinary logical resources after verifying the exact advertised
/// routes. A missing route is an `Unsupported` outcome; creation failure after
/// a positive fact is a conformance failure and therefore panics with context.
pub(crate) fn logical_creation(device: &Device, label: &str) -> CaseOutcome {
    let buffer_usage = BufferUsage::COPY_SRC
        .union(BufferUsage::COPY_DST)
        .union(BufferUsage::VERTEX);
    let buffer_support = device
        .capabilities()
        .buffer_support(&BufferSupportQuery::new(buffer_usage));
    if !buffer_support.is_supported() {
        return CaseOutcome::Unsupported("COPY_SRC | COPY_DST | VERTEX buffer route".into());
    }
    let buffer = device
        .create_buffer(&BufferDescriptor::new(4096, buffer_usage))
        .unwrap_or_else(|error| panic!("{label}: advertised core buffer creation failed: {error}"));
    assert_eq!(
        buffer.descriptor().size,
        4096,
        "{label}: buffer descriptor drift"
    );

    device
        .create_sampler(&SamplerDescriptor::new())
        .unwrap_or_else(|error| panic!("{label}: core sampler creation failed: {error}"));

    let texture_usage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);
    let texture_support = device
        .capabilities()
        .texture_support(&TextureSupportQuery::new(
            crate::api::resource::texture::TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            texture_usage,
            1,
        ));
    if !texture_support.is_supported() {
        return CaseOutcome::Unsupported("RGBA8 2D copy-src/copy-dst texture route".into());
    }
    let texture = device
        .create_texture(&TextureDescriptor::new_2d(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            texture_usage,
        ))
        .unwrap_or_else(|error| {
            panic!("{label}: advertised RGBA8 texture creation failed: {error}")
        });
    let view = TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)
        .unwrap_or_else(|error| panic!("{label}: whole RGBA8 view descriptor failed: {error}"));
    device
        .create_texture_view(&texture, &view)
        .unwrap_or_else(|error| panic!("{label}: advertised RGBA8 view creation failed: {error}"));
    CaseOutcome::Pass
}

/// Builds the portable empty plan. Fixtures submit it through their normal
/// async executor; no native queue/no-op object is encoded by this workload.
pub(crate) fn empty_plan(device: &Device, label: &str) -> SubmissionPlan {
    SubmissionPlanBuilder::new(device)
        .build()
        .unwrap_or_else(|error| panic!("{label}: portable empty plan construction failed: {error}"))
}
