//! Fine-grained native command entry-point evidence.

use crate::backend::gl::api::{GlFamilyProfile, GlKnownExtension};
use crate::backend::gl::features::{GlFeature, required_functions};
use core::ffi::c_void;

/// One optional native command family.  The table intentionally names semantic
/// groups rather than pretending one loaded symbol proves every related call.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NativeEntryPoint {
    Compute,
    StorageBuffer,
    StorageImage,
    IndirectDispatch,
    DrawIndirect,
    TimerQuery,
    DebugOutput,
    Robustness,
}

/// Loaded entry-point evidence for one current context generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeEntryPoints {
    compute: bool,
    storage_buffer: bool,
    storage_image: bool,
    indirect_dispatch: bool,
    draw_indirect: bool,
    timer_query: bool,
    debug_output: bool,
    robustness: bool,
}

impl NativeEntryPoints {
    /// Resolves every optional group through the loader associated with the
    /// exact current WGL/EGL context.  A null result leaves only that group
    /// unavailable; no group is inferred from a similarly named symbol.
    ///
    /// # Safety
    ///
    /// `load` must use the current context's proc-address mechanism and every
    /// non-null pointer must be callable with the registered ABI.
    pub(crate) unsafe fn load(mut load: impl FnMut(&str) -> *const c_void) -> Self {
        let functions = |feature: GlFeature, load: &mut dyn FnMut(&str) -> *const c_void| {
            required_functions(feature).iter().all(|function| {
                function
                    .native_symbol()
                    .is_some_and(|name| !load(name).is_null())
            })
        };
        Self {
            compute: functions(GlFeature::Compute, &mut load),
            storage_buffer: functions(GlFeature::ShaderStorageBuffers, &mut load),
            storage_image: functions(GlFeature::ImageLoadStore, &mut load),
            indirect_dispatch: functions(GlFeature::IndirectDispatch, &mut load),
            draw_indirect: functions(GlFeature::IndirectDraw, &mut load),
            timer_query: functions(GlFeature::TimerQuery, &mut load),
            debug_output: functions(GlFeature::DebugMarkers, &mut load),
            // No feature registry route currently publishes robustness. Keep
            // the legacy observation fail-closed rather than load an orphan
            // symbol outside the authoritative feature vocabulary.
            robustness: false,
        }
    }

    pub(crate) const fn has(self, point: NativeEntryPoint) -> bool {
        match point {
            NativeEntryPoint::Compute => self.compute,
            NativeEntryPoint::StorageBuffer => self.storage_buffer,
            NativeEntryPoint::StorageImage => self.storage_image,
            NativeEntryPoint::IndirectDispatch => self.indirect_dispatch,
            NativeEntryPoint::DrawIndirect => self.draw_indirect,
            NativeEntryPoint::TimerQuery => self.timer_query,
            NativeEntryPoint::DebugOutput => self.debug_output,
            NativeEntryPoint::Robustness => self.robustness,
        }
    }

    /// Native admission is the intersection of a complete callable function
    /// group and independently recorded core-or-extension evidence.  `extension`
    /// must be `Some` only after the API discovery ledger recorded that exact
    /// extension as acquired (and, where required, probed); a raw extension
    /// string is deliberately insufficient.
    ///
    /// This method does not manufacture extension evidence.  It checks family
    /// legality so a platform adapter cannot accidentally replay a WebGL-only
    /// or desktop-only ledger row against a native context.
    pub(crate) const fn admits(
        self,
        point: NativeEntryPoint,
        profile: GlFamilyProfile,
        core: bool,
        acquired_extension: Option<GlKnownExtension>,
    ) -> bool {
        if !self.has(point) {
            return false;
        }
        core || match acquired_extension {
            Some(extension) => extension.is_legal_for(profile),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeEntryPoint, NativeEntryPoints};
    use crate::backend::gl::api::{GlFamilyProfile, GlKnownExtension};
    use core::ffi::c_void;

    #[test]
    fn one_missing_symbol_closes_only_its_group() {
        // SAFETY: the loader returns sentinel non-null values and no symbol is called.
        let table = unsafe {
            NativeEntryPoints::load(|name| {
                if name == "glMemoryBarrier" {
                    core::ptr::null()
                } else {
                    1usize as *const c_void
                }
            })
        };
        assert!(!table.has(NativeEntryPoint::StorageImage));
        assert!(table.has(NativeEntryPoint::DrawIndirect));
    }

    #[test]
    fn typed_extension_cannot_cross_profile_family() {
        let table = NativeEntryPoints {
            compute: true,
            ..NativeEntryPoints::default()
        };
        assert!(!table.admits(
            NativeEntryPoint::Compute,
            GlFamilyProfile::Embedded { major: 3, minor: 1 },
            false,
            Some(GlKnownExtension::ArbComputeShader),
        ));
        // An acquired extension is not a substitute for actual acquisition:
        // `None` is the representation of reported-only evidence.
        assert!(!table.admits(
            NativeEntryPoint::Compute,
            GlFamilyProfile::Desktop { major: 4, minor: 2 },
            false,
            None,
        ));
    }
}
