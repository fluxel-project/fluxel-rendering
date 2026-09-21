//! Per-context GL feature observations; route policy lives in `features`.

use super::profile::GlProfile;
use std::collections::BTreeSet;

pub(crate) use super::features::SelectedRoute;
pub(crate) use super::features::{GlExtension, GlFeature, GlFunction};

#[derive(Clone, Debug)]
pub(crate) struct GlFeatureProbe {
    profile: GlProfile,
    extensions: BTreeSet<GlExtension>,
    functions: BTreeSet<GlFunction>,
}

impl GlFeatureProbe {
    pub(crate) fn new(profile: GlProfile) -> Self {
        Self {
            profile,
            extensions: BTreeSet::new(),
            functions: BTreeSet::new(),
        }
    }
    pub(crate) fn profile(&self) -> GlProfile {
        self.profile
    }
    pub(crate) fn report_extension(&mut self, extension: GlExtension) {
        self.extensions.insert(extension);
    }
    pub(crate) fn report_function(&mut self, function: GlFunction) {
        self.functions.insert(function);
    }
    pub(crate) fn supports(&self, feature: GlFeature) -> bool {
        super::features::supports(feature, self.profile, &self.extensions, &self.functions)
    }
    pub(crate) fn missing_function(&self, feature: GlFeature) -> Option<GlFunction> {
        super::features::missing_function(feature, self.profile, &self.extensions, &self.functions)
    }
    pub(crate) fn selected_route(&self, feature: GlFeature) -> Option<SelectedRoute> {
        super::features::selected_route(feature, self.profile, &self.extensions)
    }
}
