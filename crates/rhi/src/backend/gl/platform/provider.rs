use std::sync::Arc;

use crate::api::capability::{AvailableCapabilities, CapabilityFacts};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::DeviceInstanceId;
use crate::api::platform::backend::{DeviceRequestBackend, ProviderBackend};
use crate::api::platform::provider::{AdapterInfo, BackendKind};
use crate::api::platform::request::DeviceRequestDescriptor;
use crate::api::presentation::PresentationTarget;

use super::device::GlExecutionDriver;
use super::request::GlRequest;

/// Host-provided facts for one adopted GL-family context.
///
/// This is deliberately backend-private.  A WGL/EGL/browser integration obtains
/// the profile, extension and entry-point evidence, builds the complete facts
/// table, and then adopts the context.  It is not an invitation to expose a GL
/// context or browser token through the public RHI.
pub(crate) struct GlAdoptedContext {
    pub(crate) backend: BackendKind,
    pub(crate) name: String,
    pub(crate) vendor_id: Option<u32>,
    pub(crate) device_id: Option<u32>,
    pub(crate) facts: CapabilityFacts,
    pub(crate) driver: Arc<dyn GlExecutionDriver>,
}

impl GlAdoptedContext {
    pub(crate) fn new(
        backend: BackendKind,
        name: impl Into<String>,
        facts: CapabilityFacts,
        driver: Arc<dyn GlExecutionDriver>,
    ) -> RhiResult<Self> {
        if !matches!(backend, BackendKind::OpenGl | BackendKind::WebGl2) {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "an adopted GL context must identify as OpenGl or WebGl2",
            ));
        }
        Ok(Self {
            backend,
            name: name.into(),
            vendor_id: None,
            device_id: None,
            facts,
            driver,
        })
    }
}

/// Provider for an already-owned WGL/EGL/GLX, GLES, or WebGL2 context.
///
/// Enumeration is intentionally `None`: a context is selected by the host and
/// there may be no native adapter object at all (notably WebGL2).
pub(crate) struct GlProvider {
    context: GlAdoptedContext,
    adapter: AdapterInfo,
}

impl GlProvider {
    pub(crate) fn adopt(instance: DeviceInstanceId, context: GlAdoptedContext) -> Self {
        let adapter = AdapterInfo::new(
            crate::api::platform::AdapterId::new(instance.as_u64(), 0),
            context.name.clone(),
            context.backend,
            context.vendor_id,
            context.device_id,
            AvailableCapabilities::from_facts(context.facts.clone()),
        );
        Self { context, adapter }
    }
}

impl ProviderBackend for GlProvider {
    fn enumerate_adapters(&self) -> RhiResult<Option<Vec<AdapterInfo>>> {
        Ok(None)
    }

    fn supports_presentation(
        &self,
        _adapter: crate::api::platform::AdapterId,
        target: &PresentationTarget,
    ) -> RhiResult<bool> {
        self.context.driver.supports_presentation(target)
    }

    fn request_device(
        &self,
        _descriptor: &DeviceRequestDescriptor,
    ) -> RhiResult<Box<dyn DeviceRequestBackend>> {
        Ok(Box::new(GlRequest::new(
            self.context.backend,
            self.adapter.clone(),
            self.context.facts.clone(),
            Arc::clone(&self.context.driver),
        )))
    }
}
