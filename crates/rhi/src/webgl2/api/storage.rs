//! Independently validated SSBO and shader-image bindings.

use super::{BufferId, GlError, GlFamilyApi, GlFormat, GlFormatTable, TextureId};

/// The access a program declares for a storage buffer it binds.
///
/// Ordered and hashed because [`GlShaderResourceKind`](super::GlShaderResourceKind)
/// carries one and is both: a layout is compared and keyed on, so everything a
/// layout is made of has to be as comparable as the layout.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlStorageBufferUsage {
    ReadOnly,
    ReadWrite,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlStorageBufferLimits {
    pub max_bindings: u32,
    pub max_block_size: u64,
    pub offset_alignment: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlStorageBufferRange {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
    pub usage: GlStorageBufferUsage,
}
impl GlStorageBufferRange {
    pub(crate) fn validate(
        self,
        binding: u32,
        limits: GlStorageBufferLimits,
    ) -> Result<(), GlError> {
        if binding >= limits.max_bindings
            || self.size == 0
            || self.size > limits.max_block_size
            || limits.offset_alignment == 0
            || self.offset % limits.offset_alignment != 0
        {
            return Err(GlError::Validation {
                operation: "bind_storage_buffer",
                message: "binding, aligned range, or storage-buffer limit is invalid".into(),
            });
        }
        Ok(())
    }
}

/// The access a program declares for a storage image it binds.
///
/// The same derives and the same reason as [`GlStorageBufferUsage`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlStorageImageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlStorageImageLimits {
    pub max_image_units: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlStorageImageBinding {
    pub texture: TextureId,
    pub level: u32,
    pub sample_count: u32,
    pub layered: bool,
    /// Required for a non-layered view, forbidden for a layered view.
    pub layer: Option<u32>,
    pub format: GlFormat,
    pub access: GlStorageImageAccess,
}
impl GlStorageImageBinding {
    pub(crate) fn validate(
        self,
        binding: u32,
        limits: GlStorageImageLimits,
        formats: &GlFormatTable,
    ) -> Result<(), GlError> {
        let layering_ok = if self.layered {
            self.layer.is_none()
        } else {
            self.layer.is_some()
        };
        let facts = formats.get(self.format, self.sample_count);
        let access_ok = facts.is_some_and(|f| match self.access {
            GlStorageImageAccess::ReadOnly => f.storage_read,
            GlStorageImageAccess::WriteOnly => f.storage_write,
            GlStorageImageAccess::ReadWrite => f.storage_read && f.storage_write,
        });
        if binding >= limits.max_image_units || !layering_ok || !access_ok {
            return Err(GlError::Validation {
                operation: "bind_storage_image",
                message: "image unit, layer selection, or exact format access fact is invalid"
                    .into(),
            });
        }
        Ok(())
    }
}

/// SSBO support does not imply shader-image support.
pub(crate) trait GlStorageBufferApi: GlFamilyApi {
    fn bind_storage_buffer(
        &mut self,
        binding: u32,
        range: GlStorageBufferRange,
    ) -> Result<(), GlError>;
}
/// Shader-image support does not imply SSBO support.
pub(crate) trait GlStorageImageApi: GlFamilyApi {
    fn bind_storage_image(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
mod tests {
    use super::{GlStorageBufferLimits, GlStorageBufferRange, GlStorageBufferUsage};
    use crate::webgl2::api::{BufferId, ContextEpoch, ContextStamp, DeviceIdentity};
    #[test]
    fn rejects_unaligned_storage_buffer_range() {
        let stamp = ContextStamp::new(DeviceIdentity::new(1).unwrap(), ContextEpoch::INITIAL);
        let range = GlStorageBufferRange {
            buffer: BufferId::new(stamp, 1, 1),
            offset: 3,
            size: 4,
            usage: GlStorageBufferUsage::ReadOnly,
        };
        assert!(
            range
                .validate(
                    0,
                    GlStorageBufferLimits {
                        max_bindings: 1,
                        max_block_size: 4,
                        offset_alignment: 4
                    }
                )
                .is_err()
        );
    }
}
