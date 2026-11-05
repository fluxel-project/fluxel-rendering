//! Surface output route for the Deferred Graph.
//!
//! This only changes the terminal owner: the upload/geometry/lighting Graph
//! transactions are the same as the offscreen route and the lease is recorded
//! after that Graph has been compiled.

use super::surface_validate::validate_surface_inputs;
use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(crate) fn declare_surface_with<F>(
        device: &Device<B>,
        frame: &DeferredSurfaceFrame<B>,
        declare: F,
    ) -> Result<DeferredDeclaration>
    where
        F: FnOnce(
            &mut crate::DeferredPhaseContext<'_, '_>,
        ) -> std::result::Result<(), crate::PipelineError>,
    {
        validate_surface_inputs(device, frame)?;
        Self::declare_body(
            device,
            super::record_view::DeferredRecordView::surface(frame),
            declare,
        )
    }

    pub(crate) fn validate_surface_frame(&self, frame: &DeferredSurfaceFrame<B>) -> Result<()> {
        validate_surface_inputs(&self.device, frame)?;
        if !matches!(self.output, DeferredOutputPlan::Surface { .. })
            || frame.extent != self.extent
            || frame.draw.transform_bytes != self.transform_bytes
            || self.stream_slots.len() != frame.resident.streams.len()
            || self.deformation_pass.is_some() != frame.deformation.is_some()
        {
            return Err(Error::InvalidState);
        }
        self.surface_bindings(frame).map(|_| ())
    }

    pub(super) fn surface_bindings<'a>(
        &'a self,
        frame: &'a DeferredSurfaceFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        let uses = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|item| item.slot == slot)
        };
        let mut buffers = vec![
            BoundBuffer::new(
                self.frame_upload_slot,
                frame.frame.upload_instance,
                &frame.frame.upload,
                &frame.frame.upload_desc,
                frame.frame.upload_incoming,
            ),
            BoundBuffer::new(
                self.uniform_slot,
                frame.frame.uniform_instance,
                &frame.frame.uniform,
                &frame.frame.uniform_desc,
                frame.frame.uniform_incoming,
            ),
        ];
        if uses(self.resident_upload_slot) {
            buffers.push(BoundBuffer::new(
                self.resident_upload_slot,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            ));
        }
        for (slot, stream) in self
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            if uses(slot) {
                buffers.push(BoundBuffer::new(
                    slot,
                    stream.instance,
                    &stream.buffer,
                    &stream.desc,
                    stream.incoming,
                ));
            }
        }
        if let (Some(slots), Some(deformation)) = (self.deformation_slots, &frame.deformation) {
            for (slot, owner) in [
                (slots[0], &deformation.upload),
                (slots[1], &deformation.source),
                (slots[2], &deformation.product),
                (slots[3], &deformation.parameters),
            ] {
                if uses(slot) {
                    buffers.push(BoundBuffer::new(
                        slot,
                        owner.instance,
                        &owner.buffer,
                        &owner.desc,
                        owner.incoming,
                    ));
                }
            }
        }
        let textures = std::iter::once(BoundTexture::new(
            self.texture_slot,
            frame.resident.textures[0].instance,
            &frame.resident.textures[0].texture,
            &frame.resident.textures[0].desc,
            frame.resident.textures[0].incoming,
            frame.resident.textures[0].layout,
        ))
        .chain(
            self.gbuffer_slots
                .iter()
                .copied()
                .zip(&frame.gbuffer)
                .map(|(slot, texture)| {
                    BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    )
                }),
        )
        .collect();
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }

    pub(crate) fn surface_record_lighting(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
        image: &fluxel_rhi::api::SurfaceImage<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        if image.device() != self.id.device {
            return Err(Error::WrongDevice);
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated);
        }
        if image.extent() != frame.extent || image.format() != frame.color_format {
            return Err(Error::InvalidRange);
        }
        let bindings = self.surface_bindings(frame)?;
        let pass = bindings.pass(self.lighting_pass)?;
        for (slot, range) in [
            (self.gbuffer_slots[0], whole_color()),
            (self.gbuffer_slots[1], whole_color()),
            (self.gbuffer_slots[2], whole_depth()),
        ] {
            pass.texture(slot, range, AccessKind::Read, sampled_read())?;
        }
        pass.transition(encoder)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("fluxel deferred surface lighting"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: frame.extent[0],
                height: frame.extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: frame.extent[0] as f32,
            height: frame.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        raster.scissor(Rect {
            x: 0,
            y: 0,
            width: frame.extent[0],
            height: frame.extent[1],
        })?;
        raster.set_pipeline(&frame.lighting_pipeline)?;
        raster.set_bind_group(0, &frame.lighting_group, &[])?;
        raster.draw(0..3, 0..1)?;
        raster.finish()
    }

    // These entry points deliberately remain separate from the output route:
    // their Graph passes are shared with offscreen Deferred.  The Surface
    // recorder supplies the same concrete owners, so an absent deformation is
    // an ordinary no-op rather than a second Graph shape.
    pub(crate) fn surface_record_deformation(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_deformation(
            encoder,
            super::record_view::DeferredRecordView::surface(frame),
        )
    }

    pub(crate) fn surface_record_upload(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_body_upload(
            encoder,
            super::record_view::DeferredRecordView::surface(frame),
        )
    }
    pub(crate) fn surface_record_geometry(
        &self,
        encoder: &mut CommandEncoder<B>,
        frame: &DeferredSurfaceFrame<B>,
    ) -> Result<()>
    where
        B: RasterBackend,
    {
        self.record_body_geometry(
            encoder,
            super::record_view::DeferredRecordView::surface(frame),
        )
    }
}
