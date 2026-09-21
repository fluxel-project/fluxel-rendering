//! Mock framebuffer, raster-command and batch-draw domains.
//!
//! Covers the pass lifetime (attachments, draw buffers, blit) and the commands
//! issued inside it, including the two optional raster words: the one carrying
//! a base-vertex/base-instance offset, and the batch that may be issued as one
//! combined command or as its single draws.

use super::*;

/// Reports whether a draw payload would read nothing or run no instance.
///
/// Every executable provider's draw path refuses that before it issues
/// anything, and the plain, advanced and decomposed batch routes all funnel
/// into it here, so the rule cannot drift between routes that must agree.
fn draw_is_empty(draw: GlDrawCommand) -> bool {
    match draw {
        GlDrawCommand::NonIndexed(x) => x.vertex_count == 0 || x.instance_count == 0,
        GlDrawCommand::Indexed(x) => x.index_count == 0 || x.instance_count == 0,
    }
}

impl GlFramebufferApi for MockGlFamilyApi {
    fn create_framebuffer(
        &mut self,
        d: &GlFramebufferDescriptor,
    ) -> Result<FramebufferId, GlError> {
        self.ready("create-framebuffer")?;
        // Attachments are proven *first*, which is this verb's order in both
        // providers, and it is deliberately not the order the pass entry point
        // below uses.  A pass is entered over a framebuffer whose views already
        // passed this, so there the descriptor is compared against one that is
        // known to name live storage; a creation is where that is established,
        // and a per-view check of a view whose allocation is not yet resolved is
        // the check that has nothing to say.
        //
        // This comment used to say the reverse -- "attachments are proven last,
        // as both providers do" -- and the cost of the recorder disagreeing was
        // the one thing a shared oracle must not do.  For a descriptor that is
        // bad in two ways at once the providers report the view while the
        // recorder reported the descriptor, so a differential test could not
        // compare them, and the sentence claiming otherwise was what kept the
        // divergence from looking like one.
        for v in d
            .color_attachments
            .iter()
            .copied()
            .chain(d.depth_stencil_attachment)
        {
            self.validate_attachment("create-framebuffer", v)?;
        }
        d.validate(
            self.discovery.limits().max_color_attachments,
            self.discovery.limits().max_draw_buffers,
            self.stamp,
        )
        .map_err(|_| GlError::Validation {
            operation: "create-framebuffer",
            message: "invalid framebuffer descriptor".into(),
        })?;
        // A descriptor asking for several views per attachment is refused here,
        // before the framebuffer identity exists, unless this context proved a
        // view count that can serve it.  The gate reads the same
        // `max_multiview_view_count` both providers read, at the same point in
        // the order, so a differential test of the multiview rule has the same
        // oracle on all three implementations.
        if d.validate_multiview(self.discovery.max_multiview_view_count())
            .is_err()
        {
            return self.invalid("create-framebuffer", "multiview view count is not proved");
        }
        let id = FramebufferId::new(self.stamp, self.slot()?, 0);
        self.framebuffers.insert(id, d.clone());
        self.calls.push(MockCall::CreateFramebuffer(id));
        Ok(id)
    }
    fn destroy_framebuffer(&mut self, id: FramebufferId) -> Result<(), GlError> {
        self.ready("destroy-framebuffer")?;
        self.live("destroy-framebuffer", id, |this| {
            this.framebuffers.contains_key(&id)
        })?;
        self.framebuffers.remove(&id);
        self.calls.push(MockCall::DestroyFramebuffer(id));
        Ok(())
    }
    fn begin_render_pass(&mut self, d: &GlRenderPassDescriptor) -> Result<(), GlError> {
        self.ready("begin-render-pass")?;
        let framebuffer = match d.target {
            GlRenderTarget::Offscreen(framebuffer) => framebuffer,
            GlRenderTarget::Default(_) => {
                return Err(GlError::Unsupported {
                    operation: "begin-render-pass",
                    reason: "the API mock has no acquired default framebuffer",
                });
            }
        };
        self.live("begin-render-pass", framebuffer, |this| {
            this.framebuffers.contains_key(&framebuffer)
        })?;
        if self.pass_active {
            return self.invalid("begin-render-pass", "render pass already active");
        }
        let descriptor = match self.framebuffers.get(&framebuffer).cloned() {
            Some(framebuffer) => framebuffer,
            None => return self.invalid("begin-render-pass", "framebuffer is not live"),
        };
        d.validate(
            Some(&descriptor),
            self.discovery.limits().max_color_attachments,
            self.discovery.limits().max_draw_buffers,
            self.stamp,
        )
        .map_err(|_| GlError::Validation {
            operation: "begin-render-pass",
            message: "render pass does not match its framebuffer descriptor".into(),
        })?;
        // The framebuffer already passed this gate when it was created, but a
        // pass is where the multiview rule is actually obeyed: both providers
        // re-check it here, so a recorder that only checked at creation would
        // accept a pass naming a view count this context never proved.
        if d.validate_multiview(self.discovery.max_multiview_view_count())
            .is_err()
        {
            return self.invalid("begin-render-pass", "multiview view count is not proved");
        }
        for a in &d.color_attachments {
            let GlPassAttachmentView::Allocated(view) = a.view else {
                return self.invalid(
                    "begin-render-pass",
                    "default attachment needs default target",
                );
            };
            self.validate_attachment("begin-render-pass", view)?;
            if let Some(v) = a.resolve_target {
                self.validate_attachment("begin-render-pass", v)?;
            }
        }
        self.pass_active = true;
        self.pass_framebuffer = Some(framebuffer);
        self.calls.push(MockCall::BeginRenderPass(framebuffer));
        Ok(())
    }
    fn end_render_pass(&mut self) -> Result<(), GlError> {
        self.ready("end-render-pass")?;
        if !self.pass_active {
            return self.invalid("end-render-pass", "no active render pass");
        }
        // The pass is consumed *before* the framebuffer is looked up, which is
        // the order both executable providers use: their `end` takes the pass,
        // then validates the framebuffer and reads the driver's error queue.
        // So a failure here leaves no pass active, exactly as it does on real
        // hardware -- and that is the state a caller cannot see from the error
        // alone, which is why the mock has to model it rather than only the
        // lifecycle refusals that happen before the take.
        self.pass_active = false;
        let framebuffer = self
            .pass_framebuffer
            .take()
            .expect("an active pass always names its framebuffer");
        self.live("end-render-pass", framebuffer, |this| {
            this.framebuffers.contains_key(&framebuffer)
        })?;
        // The pass end forgets the installed pipeline, exactly as the executable
        // backends do, so a draw outside a pass is refused rather than recorded
        // as a draw the driver would reject.  The current program is *not*
        // cleared: nothing in a pass end changes it.
        self.installed_raster_program = None;
        self.bound_vertex_array = None;
        self.calls.push(MockCall::EndRenderPass);
        Ok(())
    }
    fn blit_framebuffer(
        &mut self,
        source: FramebufferId,
        destination: FramebufferId,
        region: GlBlitRegion,
        filter: GlFilterMode,
        masks: GlBlitMask,
    ) -> Result<(), GlError> {
        self.ready("blit-framebuffer")?;
        self.live("blit-framebuffer", source, |this| {
            this.framebuffers.contains_key(&source)
        })?;
        self.live("blit-framebuffer", destination, |this| {
            this.framebuffers.contains_key(&destination)
        })?;
        if source == destination {
            return self.invalid(
                "blit-framebuffer",
                "blit source and destination are identical",
            );
        }
        if masks.is_empty() {
            return self.invalid(
                "blit-framebuffer",
                "blit selects no color/depth/stencil plane",
            );
        }
        region.validate().map_err(|_| GlError::Validation {
            operation: "blit-framebuffer",
            message: "invalid blit region".into(),
        })?;
        let sample_count = |descriptor: &GlFramebufferDescriptor| {
            descriptor
                .color_attachments
                .first()
                .map(|view| view.sample_count)
                .or_else(|| {
                    descriptor
                        .depth_stencil_attachment
                        .as_ref()
                        .map(|view| view.sample_count)
                })
                .unwrap_or(1)
        };
        let shape = |descriptor: &GlFramebufferDescriptor| {
            descriptor
                .color_attachments
                .first()
                .copied()
                .or(descriptor.depth_stencil_attachment)
                .map(|view| (view.width, view.height))
                .unwrap_or((0, 0))
        };
        // Both descriptors were proven live above; sample counts and extents
        // come from the recorded attachment views exactly as a real
        // completeness check would.
        let (source_shape, destination_shape) = {
            let source_shape = self.framebuffers.get(&source).map(shape).unwrap_or((0, 0));
            let destination_shape = self
                .framebuffers
                .get(&destination)
                .map(shape)
                .unwrap_or((0, 0));
            (source_shape, destination_shape)
        };
        let within = |offset: [u32; 2], extent: [u32; 2], shape: (u32, u32)| {
            offset[0]
                .checked_add(extent[0])
                .is_some_and(|end| end <= shape.0)
                && offset[1]
                    .checked_add(extent[1])
                    .is_some_and(|end| end <= shape.1)
        };
        if !within(region.src_offset, region.src_extent, source_shape) {
            return self.invalid("blit-framebuffer", "blit source leaves its framebuffer");
        }
        if !within(region.dst_offset, region.dst_extent, destination_shape) {
            return self.invalid(
                "blit-framebuffer",
                "blit destination leaves its framebuffer",
            );
        }
        if filter != GlFilterMode::Nearest {
            let multisampled = |id: FramebufferId| {
                self.framebuffers
                    .get(&id)
                    .map(sample_count)
                    .map(|count| count > 1)
                    .unwrap_or(false)
            };
            if multisampled(source) || multisampled(destination) {
                return self.invalid(
                    "blit-framebuffer",
                    "multisampled blit targets only accept nearest filtering",
                );
            }
        }
        // Depth/stencil planes never scale and never filter.
        if (masks.depth || masks.stencil)
            && (filter != GlFilterMode::Nearest || region.src_extent != region.dst_extent)
        {
            return self.invalid(
                "blit-framebuffer",
                "depth/stencil blits require nearest filtering and identical extents",
            );
        }
        self.calls.push(MockCall::BlitFramebuffer {
            source,
            destination,
        });
        Ok(())
    }
}
impl GlRasterCommandApi for MockGlFamilyApi {
    fn set_raster_pipeline(&mut self, p: &GlRasterPipeline) -> Result<(), GlError> {
        self.ready("set-raster-pipeline")?;
        if !self.pass_active {
            return self.invalid("set-raster-pipeline", "no active render pass");
        }
        self.live("set-raster-pipeline", p.program, |this| {
            this.programs.contains(&p.program)
        })?;
        self.live("set-raster-pipeline", p.vertex_array, |this| {
            this.vaos.contains(&p.vertex_array)
        })?;
        self.select_program(p.program);
        self.installed_raster_program = Some(p.program);
        self.bound_vertex_array = Some(p.vertex_array);
        self.calls.push(MockCall::SetRasterPipeline {
            program: p.program,
            vertex_array: p.vertex_array,
        });
        Ok(())
    }
    fn draw_raster(&mut self, d: GlDrawCommand) -> Result<(), GlError> {
        self.ready("draw-raster")?;
        if !self.pass_active {
            return self.invalid("draw-raster", "no active render pass");
        }
        if draw_is_empty(d) {
            return self.invalid("draw-raster", "draw count and instances must be nonzero");
        }
        // The recorder models no installed pipeline for the draw domains -- a
        // provider's "no pipeline is installed" refusal has its own tests against
        // the provider -- but where a pipeline *was* installed this models the two
        // things a draw has to restore: a compute install or a link may have taken
        // the current-program slot since, and the vertex array the pipeline named
        // may have been destroyed and replaced by a later input reconcile, which
        // is what an uncached run does on every request.  Both are refusals the
        // executable backends make at the draw, so a recorder that skipped them
        // would let a differential pass here and fail on hardware.
        if let Some(program) = self.installed_raster_program {
            self.select_program(program);
        }
        if let Some(array) = self.bound_vertex_array {
            self.live("draw-raster", array, |this| this.vaos.contains(&array))?;
        }
        self.calls.push(MockCall::DrawRaster(d));
        Ok(())
    }
}
impl GlAdvancedRasterApi for MockGlFamilyApi {
    fn draw_advanced_raster(&mut self, draw: GlAdvancedDrawCommand) -> Result<(), GlError> {
        const OP: &str = "draw-advanced-raster";
        self.ready(OP)?;
        if !self.pass_active {
            return self.invalid(OP, "no active render pass");
        }
        // Same rule as the plain raster word and checked in the same place: a
        // payload that reads nothing or runs no instance is not a draw on any
        // route, and reporting an offset failure first would name the wrong
        // problem.
        if draw_is_empty(draw.draw) {
            return self.invalid(OP, "draw count and instances must be nonzero");
        }
        if draw.validate(self.advanced_raster).is_err() {
            // An offset this context never proved is a capability fact, not a
            // malformed command, so it is reported the way every other
            // unproved-domain rejection is: the caller has to stop asking, not
            // fix the arguments.
            return self.error_result(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the requested optional draw offset",
            });
        }
        self.calls.push(MockCall::DrawAdvancedRaster(draw));
        Ok(())
    }
}
impl GlMultiDrawApi for MockGlFamilyApi {
    fn multi_draw(&mut self, command: &GlMultiDraw) -> Result<(), GlError> {
        const OP: &str = "multi-draw";
        self.ready(OP)?;
        if !self.pass_active {
            return self.invalid(OP, "no active render pass");
        }
        // The trace reports the route this context proved, never the caller's
        // intent: a context that proved the combined command records the batch
        // as one command, every other context records the single draws the
        // batch decomposes into.  Both providers choose between those same two
        // shapes, so this is the one place a state-machine test can see which
        // submission shape actually happened.  The browser provider additionally
        // needs an installed pipeline on the combined route; the recorder does
        // not model the pipeline installation, so it accepts on the pass alone
        // rather than inventing a second condition it cannot observe.
        if self
            .discovery
            .capabilities()
            .supports(GlCapability::MultiDraw)
        {
            self.calls.push(MockCall::MultiDraw(command.clone()));
            return Ok(());
        }
        issue_single_draws(self, command)
    }
}
