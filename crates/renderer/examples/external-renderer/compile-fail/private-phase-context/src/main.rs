use fluxel_renderer::{
    ForwardCompositionContext, ForwardFeaturePass, ForwardPhaseContext, RenderFeature,
};

fn main() {
    let _ = ForwardPhaseContext::new;
}

// These signatures only receive renderer-issued scoped values; they never
// construct Graph state or backend objects.  The field reads are the intended
// privacy violations: external code must not extract either authority.
fn feature_scope(pass: &ForwardFeaturePass<'_, '_>) {
    let _ = &pass.scope;
}

fn composition_phase<B, F>(context: &ForwardCompositionContext<'_, '_, '_, B, F>)
where
    B: fluxel_rhi::backend::ResourceBackend,
    F: RenderFeature,
{
    let _ = &context.phase;
}
