//! Per-frame logical declaration and binding facts for the read-set fixture.
use super::*;

impl ReadSetRasterFrame {
    pub(crate) fn compile(destination: Range<u64>) -> Result<Self> {
        if !destination.start.is_multiple_of(256)
            || destination.end.checked_sub(destination.start) != Some(256)
            || u32::try_from(destination.start).is_err()
        {
            return Err(Error::InvalidRange);
        }
        let uniform_range = destination.start
            ..destination
                .start
                .checked_add(16)
                .ok_or(Error::InvalidRange)?;
        let mut graph = GraphBuilder::default();
        let upload = graph
            .import_buffer_desc(
                BufferDesc {
                    label: None,
                    size: UPLOAD_BYTES,
                    usage: BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
                    memory: MemoryClass::Upload,
                },
                host(),
            )
            .map_err(|_| Error::InvalidState)?;
        let uniform = graph
            .create_buffer_desc(BufferDesc {
                label: None,
                size: destination.end,
                usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
                memory: MemoryClass::Device,
            })
            .map_err(|_| Error::InvalidState)?;
        let source = graph
            .create_texture(tex_desc(
                "read-set source",
                [1, 1],
                TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            ))
            .map_err(|_| Error::InvalidState)?;
        let target = graph
            .create_texture(tex_desc(
                "read-set target",
                EXTENT,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            ))
            .map_err(|_| Error::InvalidState)?;
        let readback_buffer = graph
            .create_buffer_desc(BufferDesc {
                label: None,
                size: u64::from(ROW_PITCH) * u64::from(EXTENT[1]),
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .map_err(|_| Error::InvalidState)?;
        let (upload_pass, (uniform, source)) = graph
            .add_pass("read-set-upload", |pass| {
                pass.read_with_use(upload, 0..16, copy_read(), BufferUsage::COPY_SRC)?;
                pass.read_with_use(
                    upload,
                    TEXEL_OFFSET..TEXEL_OFFSET + 4,
                    copy_read(),
                    BufferUsage::COPY_SRC,
                )?;
                let uniform = pass.write_with_use(
                    uniform,
                    uniform_range.clone(),
                    copy_write(),
                    BufferUsage::COPY_DST,
                )?;
                let source = pass.write_texture_with_use(
                    source,
                    whole(),
                    copy_tex_write(),
                    ImageLayout::CopyDst,
                    TextureUsage::COPY_DST,
                )?;
                Ok((uniform, source))
            })
            .map_err(|_| Error::InvalidState)?;
        let (raster, target) = graph
            .add_pass("read-set-raster", |pass| {
                pass.read_with_use(
                    uniform,
                    uniform_range.clone(),
                    shader_read(),
                    BufferUsage::UNIFORM,
                )?;
                pass.read_texture_with_use(
                    source,
                    whole(),
                    shader_tex_read(),
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.write_texture_with_use(
                    target,
                    whole(),
                    color_write(),
                    ImageLayout::ColorAttachment,
                    TextureUsage::RENDER_ATTACHMENT,
                )
            })
            .map_err(|_| Error::InvalidState)?;
        let (readback, ()) = graph
            .add_pass("read-set-readback", |pass| {
                pass.read_texture_with_use(
                    target,
                    whole(),
                    copy_tex_read(),
                    ImageLayout::CopySrc,
                    TextureUsage::COPY_SRC,
                )?;
                pass.write_with_use(
                    readback_buffer,
                    0..u64::from(ROW_PITCH) * u64::from(EXTENT[1]),
                    copy_write(),
                    BufferUsage::COPY_DST,
                )?;
                pass.side_effect();
                Ok(())
            })
            .map_err(|_| Error::InvalidState)?;
        let execution = CompiledExecution::compile(&graph)?;
        if execution.logical().passes() != [upload_pass, raster, readback] {
            return Err(Error::InvalidState);
        }
        Ok(Self {
            execution,
            uniform_range,
            upload: upload_pass,
            raster,
            readback,
            source_slot: upload.into(),
            uniform_slot: uniform.into(),
            readback_slot: readback_buffer.into(),
            texture_slot: source.into(),
            target_slot: target.into(),
        })
    }

    pub(super) fn bindings<'a, B: ResourceBackend>(
        &'a self,
        device: DeviceId,
        buffers: &'a ReadSetRasterBuffers<B>,
    ) -> std::result::Result<FrameBindings<'a, B>, BindingError> {
        FrameBindings::new(
            &self.execution,
            device,
            vec![
                BoundBuffer::new(
                    self.source_slot,
                    buffers.upload_instance,
                    &buffers.upload,
                    &buffers.upload_desc,
                    host(),
                ),
                BoundBuffer::new(
                    self.uniform_slot,
                    buffers.uniform_instance,
                    &buffers.uniform,
                    &buffers.uniform_desc,
                    empty(),
                ),
                BoundBuffer::new(
                    self.readback_slot,
                    buffers.readback_instance,
                    &buffers.readback,
                    &buffers.readback_desc,
                    empty(),
                ),
            ],
            vec![
                BoundTexture::new(
                    self.texture_slot,
                    buffers.source_instance,
                    &buffers.source,
                    &buffers.source_desc,
                    buffers.source_initial,
                    buffers.source_layout,
                ),
                BoundTexture::new(
                    self.target_slot,
                    buffers.target_instance,
                    &buffers.target,
                    &buffers.target_desc,
                    buffers.target_initial,
                    buffers.target_layout,
                ),
            ],
        )
    }

    pub(crate) fn validate_bindings<B: ResourceBackend>(
        &self,
        device: DeviceId,
        buffers: &ReadSetRasterBuffers<B>,
    ) -> std::result::Result<(), BindingError> {
        self.bindings(device, buffers).map(|_| ())
    }
}
