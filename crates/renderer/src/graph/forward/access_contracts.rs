//! Portable access contracts and copy footprints for Forward.
use super::*;

pub(super) fn whole() -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..1, 0..1)
}
pub(super) fn texture_copy_end(copy: &crate::residency::ForwardTextureUpload) -> Result<u64> {
    copy.source_offset
        .checked_add(
            u64::from(copy.bytes_per_row)
                .checked_mul(u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?,
        )
        .ok_or(Error::InvalidRange)
}
pub(super) fn graph_texture_copy_end(copy: &ForwardGraphTextureUpload) -> Result<u64> {
    copy.source_offset
        .checked_add(
            u64::from(copy.bytes_per_row)
                .checked_mul(u64::from(copy.rows_per_image))
                .ok_or(Error::InvalidRange)?,
        )
        .ok_or(Error::InvalidRange)
}
pub(super) fn buffer(uses: BufferUses, usage: BufferUsage) -> BufferAccessContract {
    BufferAccessContract {
        uses,
        descriptor_usage: usage,
    }
}
pub(super) fn texture(
    uses: TextureUses,
    layout: ImageLayout,
    usage: TextureUsage,
) -> TextureAccessContract {
    TextureAccessContract {
        uses,
        layout,
        descriptor_usage: usage,
    }
}
pub(super) fn copy_read_uses() -> BufferAccessContract {
    buffer(
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        BufferUsage::COPY_SRC,
    )
}
pub(super) fn copy_write_uses() -> BufferAccessContract {
    buffer(
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        BufferUsage::COPY_DST,
    )
}
pub(super) fn vertex_read_uses() -> BufferAccessContract {
    buffer(
        BufferUses {
            stages: StageMask::VERTEX_INPUT,
            accesses: AccessMask::VERTEX_READ,
        },
        BufferUsage::VERTEX,
    )
}
pub(super) fn shader_read_uses() -> BufferAccessContract {
    buffer(
        BufferUses {
            stages: StageMask::VERTEX | StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        BufferUsage::UNIFORM,
    )
}
pub(super) fn texture_read_uses() -> TextureAccessContract {
    texture(
        TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        ImageLayout::ShaderReadOnly,
        TextureUsage::TEXTURE_BINDING,
    )
}
pub(super) fn texture_copy_write_uses() -> TextureAccessContract {
    texture(
        TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        ImageLayout::CopyDst,
        TextureUsage::COPY_DST,
    )
}
pub(super) fn texture_copy_read_uses() -> TextureAccessContract {
    texture(
        TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        },
        ImageLayout::CopySrc,
        TextureUsage::COPY_SRC,
    )
}
pub(super) fn color_write_uses() -> TextureAccessContract {
    texture(
        TextureUses {
            stages: StageMask::COLOR_OUTPUT,
            accesses: AccessMask::COLOR_WRITE,
        },
        ImageLayout::ColorAttachment,
        TextureUsage::RENDER_ATTACHMENT,
    )
}
