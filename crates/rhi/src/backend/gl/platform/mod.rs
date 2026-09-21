//! Provider and execution-domain adaptation for adopted GL-family contexts.
//!
//! GL, GLES and WebGL do not have a portable instance/device split.  A host
//! owns a current context (or a browser owner thread) and adopts it here.  The
//! provider consequently does not enumerate adapters; its one request produces
//! a logical v13 device around that already selected context.

mod device;
mod presentation;
mod provider;
mod request;

#[allow(unused_imports, reason = "host glue is the future external caller")]
pub(crate) use device::{
    GlBindGroupEntry, GlBindGroupPacket, GlBindGroupRef, GlBindingResource, GlBufferRef,
    GlComputePipelinePacket, GlComputePipelineRef, GlDevice, GlExecutionDriver, GlLossSink,
    GlObjectKind, GlObjectName, GlQuerySetRef, GlRasterPipelinePacket, GlRasterPipelineRef,
    GlSamplerRef, GlShaderRef, GlSubmissionBatch, GlSubmissionPlan, GlTextureRef, GlTextureViewRef,
};
pub(crate) use presentation::{GlAcquiredFramebuffer, GlPresentationLease, framebuffer_ref};
#[allow(unused_imports, reason = "host glue is the future external caller")]
pub(crate) use provider::{GlAdoptedContext, GlProvider};

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::api::capability::CapabilityFacts;
    use crate::api::error::RhiResult;
    use crate::api::identity::DeviceInstanceId;
    use crate::api::platform::backend::{ProviderBackend, RequestProgress};
    use crate::api::platform::{BackendKind, DeviceStatus};
    use crate::api::submission::{CompletionState, LaneWorkDomains};

    use super::{GlAdoptedContext, GlExecutionDriver, GlProvider};

    struct Driver;
    impl GlExecutionDriver for Driver {
        fn dispatch(&self, _: &'static str) -> RhiResult<()> {
            Ok(())
        }
    }

    struct BufferDriver(Mutex<Vec<(super::device::GlObjectKind, u32)>>);
    impl GlExecutionDriver for BufferDriver {
        fn dispatch(&self, _: &'static str) -> RhiResult<()> {
            Ok(())
        }
        fn create_buffer(
            &self,
            _: &crate::api::resource::buffer::BufferDescriptor,
        ) -> RhiResult<super::device::GlObjectName> {
            super::device::GlObjectName::new(17, "BufferDriver::create_buffer")
        }
        fn destroy(&self, kind: super::device::GlObjectKind, name: super::device::GlObjectName) {
            self.0.lock().unwrap().push((kind, name.raw()));
        }
    }

    #[test]
    fn adopted_context_does_not_fabricate_adapter_enumeration() {
        let context = GlAdoptedContext::new(
            BackendKind::WebGl2,
            "browser WebGL2",
            CapabilityFacts::empty(),
            Arc::new(Driver),
        )
        .unwrap();
        let provider = GlProvider::adopt(DeviceInstanceId::new(91), context);
        assert!(provider.enumerate_adapters().unwrap().is_none());
    }

    #[test]
    fn adopted_request_creates_one_ordered_raster_copy_device() {
        let context = GlAdoptedContext::new(
            BackendKind::OpenGl,
            "GL 4.x",
            CapabilityFacts::empty(),
            Arc::new(Driver),
        )
        .unwrap();
        let provider = GlProvider::adopt(DeviceInstanceId::new(92), context);
        let descriptor = crate::api::platform::DeviceRequestDescriptor::new(
            crate::api::platform::AdapterSelection::Default,
            crate::api::platform::requirements::DeviceRequirements::new(),
        );
        let mut request = provider.request_device(&descriptor).unwrap();
        let device = match request
            .poll_or_register_waker(std::task::Waker::noop())
            .unwrap()
        {
            RequestProgress::Ready(device) => device,
            RequestProgress::Pending => {
                assert!(false, "adopted contexts must be ready on the first poll");
                return;
            }
        };
        assert_eq!(device.backend_kind(), BackendKind::OpenGl);
        assert_eq!(device.submission_capabilities().lanes().len(), 1);
        assert!(
            device.submission_capabilities().lanes()[0]
                .domains()
                .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
        );
        // A GL device always exposes the presentation facet; unsupported is a
        // driver answer for a particular adopted target, not an absent façade.
        assert!(device.presentation().is_some());
    }

    #[test]
    fn loss_is_terminal_and_changes_unknown_completion_to_device_lost() {
        let context = GlAdoptedContext::new(
            BackendKind::WebGl2,
            "browser WebGL2",
            CapabilityFacts::empty(),
            Arc::new(Driver),
        )
        .unwrap();
        let provider = GlProvider::adopt(DeviceInstanceId::new(93), context);
        let descriptor = crate::api::platform::DeviceRequestDescriptor::new(
            crate::api::platform::AdapterSelection::Default,
            crate::api::platform::requirements::DeviceRequirements::new(),
        );
        let mut request = provider.request_device(&descriptor).unwrap();
        let device = match request
            .poll_or_register_waker(std::task::Waker::noop())
            .unwrap()
        {
            RequestProgress::Ready(device) => device,
            RequestProgress::Pending => {
                assert!(false, "adopted contexts must be ready on the first poll");
                return;
            }
        };
        let native = device
            .as_any()
            .downcast_ref::<super::device::GlDevice>()
            .unwrap();
        native.mark_lost("webglcontextlost");
        assert_eq!(device.status(), DeviceStatus::Lost);
        assert_eq!(device.loss_info().unwrap().message(), "webglcontextlost");
        assert!(matches!(
            device.completion(5),
            CompletionState::DeviceLost(_)
        ));
    }

    #[test]
    fn returned_native_object_is_retained_then_destroyed_on_its_owner_driver() {
        let driver = Arc::new(BufferDriver(Mutex::new(Vec::new())));
        let context = GlAdoptedContext::new(
            BackendKind::OpenGl,
            "GL 4.x",
            CapabilityFacts::empty(),
            driver.clone(),
        )
        .unwrap();
        let provider = GlProvider::adopt(DeviceInstanceId::new(94), context);
        let descriptor = crate::api::platform::DeviceRequestDescriptor::new(
            crate::api::platform::AdapterSelection::Default,
            crate::api::platform::requirements::DeviceRequirements::new(),
        );
        let mut request = provider.request_device(&descriptor).unwrap();
        let device = match request
            .poll_or_register_waker(std::task::Waker::noop())
            .unwrap()
        {
            RequestProgress::Ready(device) => device,
            RequestProgress::Pending => {
                assert!(false, "adopted contexts must be ready");
                return;
            }
        };
        let buffer = device
            .create_buffer(&crate::api::resource::BufferDescriptor::new(
                4,
                crate::api::resource::BufferUsage::COPY_SRC,
            ))
            .unwrap();
        assert!(driver.0.lock().unwrap().is_empty());
        drop(buffer);
        assert!(matches!(
            driver.0.lock().unwrap().as_slice(),
            [(super::device::GlObjectKind::Buffer, 17)]
        ));
    }

    #[test]
    fn zero_native_name_is_rejected_before_a_public_handle_can_exist() {
        assert!(super::device::GlObjectName::new(0, "test").is_err());
    }
}
