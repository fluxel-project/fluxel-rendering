//! Structured Forward preparation failures.

use super::*;
use std::fmt;

/// A structured refusal while preparing one built-in Forward draw.
#[derive(Debug)]
pub enum ForwardPreparationError {
    /// A feature submitted the same local draw identity more than once.
    DuplicateFeatureLocalId {
        /// Feature whose request list was invalid.
        feature: FeatureHandle,
        /// Duplicate identity within that feature's request list.
        local_id: u64,
    },
    /// An exact asset generation referenced by a scene record was unavailable.
    MissingAsset {
        /// Asset category.
        asset: &'static str,
        /// Draw requesting the asset.
        source: ForwardDrawSource,
    },
    /// The supplied asset list contained one exact identity more than once.
    DuplicateAsset {
        /// Asset category.
        asset: &'static str,
        /// Draw requesting the asset.
        source: ForwardDrawSource,
    },
    /// The per-record tint has a non-finite component.
    InvalidTint {
        /// Draw requesting the tint.
        source: ForwardDrawSource,
    },
    /// An instance names a different asset generation or compiled variant.
    MaterialInstanceVariant {
        /// Draw requesting the material instance.
        source: ForwardDrawSource,
    },
    /// A named instance value is missing, non-finite, or has the wrong type.
    InvalidMaterialParameter {
        /// Draw requesting the value.
        source: ForwardDrawSource,
        /// Parameter in the reflected binding schema.
        parameter: String,
    },
    /// A record transform has a non-finite component.
    InvalidTransform {
        /// Draw requesting the transform.
        source: ForwardDrawSource,
    },
    /// Transmission thickness cannot be represented safely after the validated
    /// world-scale conversion.
    InvalidTransmissionTransform {
        /// Draw requesting volume transmission.
        source: ForwardDrawSource,
    },
    /// A retained view has a non-finite component or empty extent.
    InvalidView,
    /// Geometry stream count differs from its declared runtime layout.
    GeometryStreamCount {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Declared stream count.
        expected: usize,
        /// Actual CPU stream count.
        actual: usize,
    },
    /// Geometry declared an inverted vertex range.
    InvalidVertexRange {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// The instance stream does not match the selected affine-column layout.
    InstanceStreamUnsupported {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Stream index.
        stream: usize,
    },
    /// The declared instance stream does not supply all four transform columns.
    MissingInstanceTransformColumns {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// An instance transform has an invalid affine matrix or normal transform.
    InvalidInstanceTransform {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// The failed instance's index in the draw.
        index: u32,
    },
    /// The draw cannot use one front-face state for both instance orientations.
    MixedInstanceOrientation {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// The selected material cannot share the geometry instance draw.
    InstanceMaterialUnsupported {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// A CPU stream cannot satisfy its selected draw's physical attribute footprint.
    GeometryPayloadTooShort {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
        /// Stream index.
        stream: usize,
    },
    /// Geometry does not provide the required `POSITION` semantic.
    MissingPosition {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// `UV` and `COLOR` must occur together for the first optional-attribute template.
    IncompleteOptionalAttributes {
        /// Draw requesting the geometry.
        source: ForwardDrawSource,
    },
    /// PBR geometry did not expose the required `NORMAL` semantic.
    MissingNormal {
        /// Draw requesting the PBR normal.
        source: ForwardDrawSource,
    },
    /// UV0-evaluated PBR geometry did not expose exactly one `UV0` vec2 semantic.
    MissingUv0 {
        /// Draw requesting the PBR UV0 evaluation context.
        source: ForwardDrawSource,
    },
    /// A consumed PBR normal was non-finite or had zero length.
    InvalidNormal {
        /// Draw requesting the PBR normal.
        source: ForwardDrawSource,
        /// Stream containing the invalid normal.
        stream: usize,
    },
    /// Normal-map PBR requires a finite non-zero tangent with a non-zero handedness.
    InvalidTangent {
        /// Draw requesting the tangent.
        source: ForwardDrawSource,
        /// Stream containing the invalid tangent.
        stream: usize,
    },
    /// Normal-map PBR geometry did not expose `TANGENT` as Float32x4.
    MissingTangent {
        /// Draw requesting the tangent.
        source: ForwardDrawSource,
    },
    /// PBR needs an affine world transform with an invertible linear part.
    NonInvertibleNormalTransform {
        /// Draw requesting the normal transform.
        source: ForwardDrawSource,
    },
    /// A Material contribution is outside the initial Forward binding schema.
    MaterialBindingSchema {
        /// Draw requesting the Material.
        source: ForwardDrawSource,
    },
    /// The current renderer does not yet consume an explicitly requested surface effect.
    UnsupportedSurfaceExtensions {
        /// Draw requesting the unsupported effect.
        source: ForwardDrawSource,
    },
    /// Environment transmission needs the selected radiance chain and BRDF LUT.
    TransmissionRequiresSpecularEnvironment {
        /// Draw requesting environment refraction.
        source: ForwardDrawSource,
    },
    /// The selected scene refraction texture and sampler must be actual material bindings.
    InvalidSceneRefractionSource {
        /// Draw requesting the capture texture.
        source: ForwardDrawSource,
    },
    /// The submitted pipeline does not consume a scene-color refraction receiver.
    UnsupportedSceneRefractionReceiver {
        /// Draw requesting a capture receiver.
        source: ForwardDrawSource,
    },
    /// Material, Shader, or geometry logical validation failed.
    Material(MaterialPreparationError),
    /// Texture dimensions or payload length are invalid.
    InvalidTexture {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
    },
    /// A legacy CPU refusal retained for callers that classify preparation
    /// errors. Linear and sRGB RGBA8 now use the same format-preserving Graph
    /// generation path; device-specific refusals originate from RHI instead.
    UnsupportedTextureMipGeneration {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
        /// Transfer encoding whose mip generation has no defined color path.
        encoding: TextureEncoding,
        /// Requested total mip levels.
        mip_levels: u32,
    },
    /// A material texture resource appears more than once in one draw input.
    DuplicateTextureResource {
        /// Draw requesting the material resource.
        source: ForwardDrawSource,
        /// Duplicate material resource name.
        resource: String,
    },
    /// A material-declared texture resource has no supplied source.
    MissingTextureResource {
        /// Draw requesting the material resource.
        source: ForwardDrawSource,
        /// Missing material resource name.
        resource: String,
    },
    /// A draw supplied a source for no material-declared texture resource.
    UnexpectedTextureResource {
        /// Draw requesting the material resource.
        source: ForwardDrawSource,
        /// Unexpected material resource name.
        resource: String,
    },
    /// An asset texture's transfer encoding differs from its material requirement.
    TextureEncodingMismatch {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
        /// Material texture resource name.
        resource: String,
        /// Material-required encoding.
        expected: TextureEncoding,
        /// Asset-declared encoding.
        actual: TextureEncoding,
    },
    /// A supplied source kind cannot provide the material binding's exact view dimension.
    TextureResourceDimensionMismatch {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
        /// Material texture resource name.
        resource: String,
        /// Material-reflected view dimension.
        expected: fluxel_rhi::api::TextureViewDimension,
        /// Dimension provided by the supplied source kind.
        actual: fluxel_rhi::api::TextureViewDimension,
    },
    /// A supplied source cannot provide the material binding's reflected sample aspect.
    TextureResourceSampleMismatch {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
        /// Material texture resource name.
        resource: String,
        /// Material-reflected sample type.
        expected: fluxel_rhi::api::TextureSampleType,
        /// Sample type provided by the supplied source kind.
        actual: fluxel_rhi::api::TextureSampleType,
    },
    /// A supplied source cannot provide the material binding's reflected access role.
    TextureResourceAccessMismatch {
        /// Draw requesting the texture.
        source: ForwardDrawSource,
        /// Material texture resource name.
        resource: String,
    },
    /// A draw supplied sampler state for a material without a matching sampler.
    UnexpectedSamplerResource {
        /// Draw requesting the binding.
        source: ForwardDrawSource,
        /// Unexpected sampler resource name.
        resource: String,
    },
    /// A draw supplied more than one override for the same sampler resource.
    DuplicateSamplerResource {
        /// Draw requesting the binding.
        source: ForwardDrawSource,
        /// Repeated sampler resource name.
        resource: String,
    },
    /// A draw named a different sampler resource than the material declares.
    SamplerResourceMismatch {
        /// Draw requesting the binding.
        source: ForwardDrawSource,
        /// Material-declared sampler resource name.
        expected: String,
        /// Supplied sampler resource name.
        actual: String,
    },
    /// A material-declared comparison sampler has no explicit descriptor.
    MissingSamplerResource {
        /// Draw requesting the binding.
        source: ForwardDrawSource,
        /// Missing material sampler resource name.
        resource: String,
    },
    /// An explicit sampler descriptor violates the portable contract.
    InvalidSamplerDescriptor {
        /// Draw requesting the binding.
        source: ForwardDrawSource,
    },
    /// An ordinary Forward draw attempted to consume a same-frame capture slot.
    CaptureTextureOutsideReceiver {
        /// Draw requesting the capture product.
        source: ForwardDrawSource,
        /// Material resource requested from the capture product.
        resource: String,
    },
    /// A capture receiver supplied no same-frame capture source.
    CaptureReceiverRequiresSlot {
        /// Draw requesting the capture product.
        source: ForwardDrawSource,
    },
    /// A capture source named an empty producer slot.
    MissingCaptureSlot {
        /// Draw requesting the capture product.
        source: ForwardDrawSource,
        /// Requested producer slot name.
        slot: String,
    },
    /// Reserved for a renderer-owned capture-slot validation failure.
    InvalidCaptureSlot {
        /// Draw requesting the capture product.
        source: ForwardDrawSource,
        /// Requested producer slot name.
        slot: String,
    },
}

impl fmt::Display for ForwardPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Forward preparation refused: {self:?}")
    }
}

impl std::error::Error for ForwardPreparationError {}
