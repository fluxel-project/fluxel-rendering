//! Opt-in mock wrapper over the proved compute and storage domains.
//!
//! The wrapper exists so ordinary compute-free fixtures cannot accidentally
//! reach optional domains: constructing it fails unless the bound discovery
//! snapshot proved both capabilities, mirroring the provider-side rule.
//!
//! Indirect dispatch lives here rather than with the other indirect domains
//! because it is the one indirect command that is not a raster command: it needs
//! the installed compute program, which only this wrapper carries, exactly as
//! the native provider's single compute type carries it.

use super::*;

/// Explicit opt-in domains, available only after their snapshot proved both capabilities.
#[derive(Debug)]
pub struct MockComputeStorageApi {
    inner: MockGlFamilyApi,
}
impl MockComputeStorageApi {
    pub fn new(inner: MockGlFamilyApi) -> Result<Self, GlError> {
        let caps = inner.discovery().capabilities();
        if caps.supports(GlCapability::Compute) && caps.supports(GlCapability::StorageBuffer) {
            Ok(Self { inner })
        } else {
            Err(GlError::Unsupported {
                operation: "mock-compute-storage",
                reason: "discovery did not prove compute and storage-buffer support",
            })
        }
    }
    pub fn calls(&self) -> &[MockCall] {
        self.inner.calls()
    }
    pub fn into_inner(self) -> MockGlFamilyApi {
        self.inner
    }
}
impl GlFamilyApi for MockComputeStorageApi {
    fn profile(&self) -> GlFamilyProfile {
        self.inner.profile()
    }
    fn context_stamp(&self) -> ContextStamp {
        self.inner.context_stamp()
    }
    fn lifecycle(&self) -> GlContextLifecycle {
        self.inner.lifecycle()
    }
    fn owner_thread(&self) -> OwnerThreadIdentity {
        self.inner.owner_thread()
    }
    fn assert_owner_thread(&self, op: &'static str) -> Result<(), GlError> {
        self.inner.assert_owner_thread(op)
    }
    fn discovery(&self) -> &GlDiscoverySnapshot {
        self.inner.discovery()
    }
    fn context_lost(&mut self) -> Result<(), GlError> {
        self.inner.context_lost()
    }
    fn context_restored(&mut self) -> Result<ContextStamp, GlError> {
        self.inner.context_restored()
    }
}
impl GlComputeDispatchApi for MockComputeStorageApi {
    fn set_compute_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        self.inner.ready("set-compute-program")?;
        self.inner.live("set-compute-program", program, |this| {
            this.programs.contains(&program)
        })?;
        self.inner.installed_compute_program = Some(program);
        // The verb installs as well as records, so the modelled driver's current
        // program moves here too.  A recorder that only recorded the intent would
        // accept a dispatch the real provider cannot perform.
        self.inner.select_program(program);
        self.inner.calls.push(MockCall::SetComputeProgram(program));
        Ok(())
    }
    fn dispatch(&mut self, g: GlDispatchGroups) -> Result<(), GlError> {
        self.inner.ready("dispatch")?;
        let Some(program) = self.inner.installed_compute_program else {
            return self
                .inner
                .invalid("dispatch", "no compute program is installed");
        };
        // A raster install since the compute install took the current-program
        // slot, so the selection is re-asserted, exactly as the provider does.
        self.inner.select_program(program);
        g.validate(GlComputeLimits {
            max_group_count: self.inner.discovery.limits().max_compute_work_group_count,
            max_group_size: self.inner.discovery.limits().max_compute_work_group_size,
            max_group_invocations: self
                .inner
                .discovery
                .limits()
                .max_compute_work_group_invocations,
        })?;
        self.inner.calls.push(MockCall::Dispatch(g));
        Ok(())
    }
    fn memory_barrier(&mut self, b: GlMemoryBarrier) -> Result<(), GlError> {
        self.inner.ready("memory-barrier")?;
        b.validate_nonempty()
    }
}
impl GlDispatchIndirectApi for MockComputeStorageApi {
    fn dispatch_indirect(&mut self, command: GlDispatchIndirectCommand) -> Result<(), GlError> {
        const OP: &str = "dispatch-indirect";
        self.inner.ready(OP)?;
        self.inner.require_indirect_capability(
            OP,
            GlCapability::IndirectDispatch,
            "this context did not prove the indirect-dispatch capability",
        )?;
        // The installed program is what makes the record's work-group triple
        // meaningful, so the provider checks it after the capability row and
        // before the record layout: a context that never proved dispatch must
        // not be told to install a program first.
        let Some(program) = self.inner.installed_compute_program else {
            return self.inner.invalid(OP, "no compute program is installed");
        };
        self.inner
            .live(OP, program, |this| this.programs.contains(&program))?;
        if let Err(error) = command.validate(OP) {
            return self.inner.error_result(error);
        }
        self.inner.indirect_buffer(OP, command.range)?;
        self.inner.select_program(program);
        self.inner.calls.push(MockCall::DispatchIndirect(command));
        Ok(())
    }
}
impl GlStorageBufferApi for MockComputeStorageApi {
    fn bind_storage_buffer(
        &mut self,
        binding: u32,
        r: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        self.inner.ready("bind-storage-buffer")?;
        let desc = self.inner.buffer("bind-storage-buffer", r.buffer)?;
        r.validate(
            binding,
            GlStorageBufferLimits {
                max_bindings: self.inner.discovery.limits().max_storage_buffer_bindings,
                max_block_size: self.inner.discovery.limits().max_storage_block_size,
                offset_alignment: self
                    .inner
                    .discovery
                    .limits()
                    .storage_buffer_offset_alignment,
            },
        )?;
        GlBufferRange {
            buffer: r.buffer,
            offset: r.offset,
            size: r.size,
        }
        .validate_for(desc)
        .map_err(|_| GlError::Validation {
            operation: "bind-storage-buffer",
            message: "storage buffer range is outside the allocation".into(),
        })?;
        if !desc.usage.contains(GlBufferUsage::STORAGE) {
            return self
                .inner
                .invalid("bind-storage-buffer", "buffer lacks storage usage");
        }
        self.inner.calls.push(MockCall::BindStorageBuffer {
            binding,
            buffer: r.buffer,
            offset: r.offset,
            size: r.size,
        });
        Ok(())
    }
}
impl GlStorageImageApi for MockComputeStorageApi {
    fn bind_storage_image(
        &mut self,
        binding: u32,
        i: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        self.inner.ready("bind-storage-image")?;
        self.inner.validate_storage_image_binding(binding, i)?;
        self.inner.calls.push(MockCall::BindStorageImage {
            binding,
            texture: i.texture,
        });
        Ok(())
    }
}
