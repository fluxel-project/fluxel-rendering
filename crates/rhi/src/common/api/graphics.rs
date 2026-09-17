//! The graphics family: raster passes, pipelines, vertex input and draws.
//!
//! This is the first family trait, and it follows the four conventions section
//! 11.9 of the lead 3F plan fixes:
//!
//! 1. it takes `fluxel_rendergraph`'s descriptors, which are already generic over
//!    the resource type, so the pass vocabulary is not restated here;
//! 2. it declares its own [`Self::Error`], because backends disagree about what a
//!    raster command can fail on;
//! 3. it is bounded on the base through [`FamilyApi`], never the other way;
//! 4. it adds **vocabulary only**. Whether a device can raster is not answered
//!    here -- that is the ledger's answer, read once by `require`.
//!
//! The handle *is* the recording context: every verb takes `&mut self`, so a
//! backend does not have to expose an encoder type in this vocabulary. A backend
//! that records through a command buffer it owns internally (Vulkan, DX12) keeps
//! that buffer private, and one whose context is immediate (GL) has nothing to
//! expose.
//!
//! # The pass bracket is the portable one
//!
//! [`Self::begin_raster`] takes `RasterPassDescriptor<'_, TextureId>`, the same
//! descriptor the GL family already instantiates with its own texture id. It
//! carries an optional depth-stencil attachment, and a backend whose retained
//! recipes declare no depth-stencil state refuses that attachment by name rather
//! than ignoring it -- refusing is the behavior the preserved-semantics table
//! requires, and a descriptor that could not express depth would have hidden the
//! choice instead of recording it.
//!
//! # Instance ranges, and what is deliberately not here
//!
//! [`Self::draw`] and [`Self::draw_indexed`] take an instance range because plain
//! instancing is part of the draw vocabulary and every backend serves it; what no
//! retained artifact declares is a *per-instance vertex stream*, which is a recipe
//! change and not a vocabulary one. A range that starts at a non-zero instance is
//! a different thing again -- the `FirstInstance` family -- and a backend that has
//! not proved that row refuses it.
//!
//! Base vertex is likewise absent: it is the `BaseVertex` family, it exists on
//! desktop GL only among the accepted profiles, and no artifact needs it.

use std::ops::Range;

use fluxel_rendergraph::{IndexFormat, RasterPassDescriptor, ScissorRect, Viewport};

use crate::common::api::handle::FamilyApi;
use crate::common::base::resource::{BufferId, TextureId};

/// The raster vocabulary, once a device has proved the graphics family.
pub(crate) trait GraphicsApi: FamilyApi {
    /// Why a command in this family was refused or failed.
    type Error;
    /// A raster pipeline this backend created.
    type Pipeline;
    /// A binding set this backend created.
    type Bindings;

    /// Opens a raster pass with the descriptor's declared attachments.
    fn begin_raster(
        &mut self,
        descriptor: &RasterPassDescriptor<'_, TextureId>,
    ) -> Result<(), Self::Error>;

    /// Closes the open raster pass.
    fn end_raster(&mut self) -> Result<(), Self::Error>;

    /// Selects the pipeline subsequent draws record through.
    fn set_raster_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<(), Self::Error>;

    /// Applies a binding set to the open pass.
    fn set_bindings(&mut self, bindings: &Self::Bindings) -> Result<(), Self::Error>;

    /// Binds the vertex buffer that occupies `slot`.
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: BufferId,
        offset: u64,
    ) -> Result<(), Self::Error>;

    /// Binds the index buffer subsequent indexed draws read.
    fn set_index_buffer(
        &mut self,
        buffer: BufferId,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), Self::Error>;

    /// Sets the viewport for subsequent draws.
    fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Self::Error>;

    /// Sets the scissor rectangle for subsequent draws.
    fn set_scissor(&mut self, scissor: ScissorRect) -> Result<(), Self::Error>;

    /// Records a non-indexed draw over `vertices`, once per instance in `instances`.
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<(), Self::Error>;

    /// Records an indexed draw over `indices`, once per instance in `instances`.
    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>)
    -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::common::api::family::{Graphics, UnmetReason};
    use crate::common::api::handle::verify_buffer;
    use crate::common::api::negotiate::{CapabilitySource, Provides, require};
    use crate::common::base::stamp::DeviceStamp;
    use crate::common::caps::{
        Capability, CapabilityEvidence, CapabilityFact, CapabilityLedger, OperationProbe,
    };
    use fluxel_rendergraph::{DeviceIdentity, PhysicalResourceIdentity};

    /// A device that records the raster verbs reached through its handle.
    #[derive(Debug)]
    struct MockRasterDevice {
        ledger: CapabilityLedger,
        stamp: DeviceStamp,
        calls: RefCell<Vec<&'static str>>,
    }

    impl MockRasterDevice {
        fn new() -> Self {
            Self {
                ledger: CapabilityLedger::default(),
                stamp: DeviceStamp::initial(DeviceIdentity::new(1)),
                calls: RefCell::new(Vec::new()),
            }
        }

        fn prove_graphics(&mut self) {
            self.ledger.record(
                Capability::Graphics,
                CapabilityFact {
                    evidence: Some(CapabilityEvidence::Core),
                    limits_satisfied: true,
                    operation_probe: OperationProbe::Passed,
                },
            );
        }

        fn calls(&self) -> Vec<&'static str> {
            self.calls.borrow().clone()
        }
    }

    impl CapabilitySource for MockRasterDevice {
        fn ledger(&self) -> &CapabilityLedger {
            &self.ledger
        }
    }

    /// The handle a proved graphics family yields.
    #[derive(Debug)]
    struct MockGraphics<'d>(&'d MockRasterDevice);

    impl FamilyApi for MockGraphics<'_> {
        fn stamp(&self) -> DeviceStamp {
            self.0.stamp
        }
    }

    impl GraphicsApi for MockGraphics<'_> {
        type Error = ();
        type Pipeline = ();
        type Bindings = ();

        fn begin_raster(
            &mut self,
            descriptor: &RasterPassDescriptor<'_, TextureId>,
        ) -> Result<(), ()> {
            assert!(descriptor.depth_stencil.is_none(), "no depth recipe exists");
            self.0.calls.borrow_mut().push("begin");
            Ok(())
        }

        fn end_raster(&mut self) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("end");
            Ok(())
        }

        fn set_raster_pipeline(&mut self, _: &()) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("pipeline");
            Ok(())
        }

        fn set_bindings(&mut self, _: &()) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("bindings");
            Ok(())
        }

        fn set_vertex_buffer(&mut self, _: u32, _: BufferId, _: u64) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("vertex");
            Ok(())
        }

        fn set_index_buffer(&mut self, _: BufferId, _: u64, _: IndexFormat) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("index");
            Ok(())
        }

        fn set_viewport(&mut self, _: Viewport) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("viewport");
            Ok(())
        }

        fn set_scissor(&mut self, _: ScissorRect) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("scissor");
            Ok(())
        }

        fn draw(&mut self, _: Range<u32>, _: Range<u32>) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("draw");
            Ok(())
        }

        fn draw_indexed(&mut self, _: Range<u32>, _: Range<u32>) -> Result<(), ()> {
            self.0.calls.borrow_mut().push("draw-indexed");
            Ok(())
        }
    }

    impl Provides<Graphics> for MockRasterDevice {
        type Api<'d> = MockGraphics<'d>;

        fn provide(&self) -> MockGraphics<'_> {
            MockGraphics(self)
        }
    }

    #[test]
    fn an_unproved_graphics_family_yields_no_handle() {
        let device = MockRasterDevice::new();
        let error = require::<_, Graphics>(&device).expect_err("graphics was never examined");
        assert_eq!(error.reason, UnmetReason::NotExamined);
        assert!(device.calls().is_empty());
    }

    #[test]
    fn a_proved_graphics_family_records_the_whole_raster_bracket() {
        let mut device = MockRasterDevice::new();
        device.prove_graphics();
        let mut api = require::<_, Graphics>(&device).expect("graphics was proved");
        let descriptor = RasterPassDescriptor {
            label: "contract",
            colors: &[],
            depth_stencil: None,
        };
        api.begin_raster(&descriptor).expect("pass opens");
        api.set_raster_pipeline(&()).expect("pipeline selects");
        api.set_viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .expect("viewport sets");
        api.set_scissor(ScissorRect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        })
        .expect("scissor sets");
        api.draw(0..3, 0..1).expect("draw records");
        api.draw_indexed(0..6, 0..1).expect("indexed draw records");
        api.end_raster().expect("pass closes");
        assert_eq!(
            device.calls(),
            vec![
                "begin", "pipeline", "viewport", "scissor", "draw", "draw-indexed", "end"
            ]
        );
    }

    #[test]
    fn the_handle_is_the_stamp_a_resource_id_is_verified_against() {
        let mut device = MockRasterDevice::new();
        device.prove_graphics();
        let api = require::<_, Graphics>(&device).expect("graphics was proved");
        let own = BufferId::new(device.stamp, PhysicalResourceIdentity::new(1));
        assert_eq!(verify_buffer(&api, own), Ok(()));
        let foreign = BufferId::new(
            device.stamp.next_generation(),
            PhysicalResourceIdentity::new(1),
        );
        assert!(verify_buffer(&api, foreign).is_err());
    }
}
