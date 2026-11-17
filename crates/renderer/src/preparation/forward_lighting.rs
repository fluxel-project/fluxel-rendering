//! Scene and Feature selection for the shared material preparation path.
use super::*;

pub(crate) fn prepare_forward_with_environment(
    frame: &PreparedFrame<'_, ForwardResources>,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    prepare_forward_with_lighting(frame, assets, composer, environment, false)
}

#[cfg(test)]
pub(crate) fn prepare_feature_draws_with_environment(
    frame: &PreparedFrame<'_, ForwardResources>,
    feature: FeatureHandle,
    requests: &[ForwardDrawRequest],
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    prepare_feature_draws_with_lighting(
        frame,
        feature,
        requests,
        assets,
        composer,
        environment,
        false,
    )
}

/// Resolves every selected candidate into a retained logical Forward draw input.
///
/// The supplied [`PreparedFrame`] already binds candidate identities to one
/// immutable scene version. This function resolves the exact asset generations
/// named by those records and keeps cloned snapshots in its output so later GPU
/// preparation cannot accidentally switch to newer content.
pub fn prepare_forward(
    frame: &PreparedFrame<'_, ForwardResources>,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    prepare_forward_with_environment(frame, assets, composer, false)
}

pub(crate) fn prepare_forward_with_lighting(
    frame: &PreparedFrame<'_, ForwardResources>,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
    specular: bool,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    let view = frame.view_record();
    if view.extent.contains(&0) || !finite_transform(view.transform) {
        return Err(ForwardPreparationError::InvalidView);
    }
    let mut inputs = Vec::with_capacity(frame.candidates().len());
    for &record_id in frame.candidates() {
        let record =
            frame
                .record(record_id)
                .map_err(|_| ForwardPreparationError::InvalidTransform {
                    source: ForwardDrawSource::Scene(record_id),
                })?;
        inputs.push(prepare_draw(
            view,
            ForwardDrawSource::Scene(record_id),
            0,
            &record.resources,
            record.transform,
            assets,
            composer,
            environment,
            specular,
        )?);
    }
    Ok(inputs)
}

/// Resolves feature-owned Forward requests against the selected frame view.
///
/// The requests retain exact asset generations and are validated through the
/// same path as scene-owned records. Their local identities are checked before
/// any asset lookup or shader work so invalid feature payloads have no GPU-side
/// effect.
#[cfg(test)]
pub(crate) fn prepare_feature_draws(
    frame: &PreparedFrame<'_, ForwardResources>,
    feature: FeatureHandle,
    requests: &[ForwardDrawRequest],
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    prepare_feature_draws_with_environment(frame, feature, requests, assets, composer, false)
}

pub(crate) fn prepare_feature_draws_with_lighting(
    frame: &PreparedFrame<'_, ForwardResources>,
    feature: FeatureHandle,
    requests: &[ForwardDrawRequest],
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
    specular: bool,
) -> Result<Vec<ForwardDrawInput>, ForwardPreparationError> {
    let view = frame.view_record();
    if view.extent.contains(&0) || !finite_transform(view.transform) {
        return Err(ForwardPreparationError::InvalidView);
    }
    let mut local_ids = std::collections::HashSet::with_capacity(requests.len());
    for request in requests {
        if !local_ids.insert(request.local_id) {
            return Err(ForwardPreparationError::DuplicateFeatureLocalId {
                feature,
                local_id: request.local_id,
            });
        }
    }
    requests
        .iter()
        .map(|request| {
            prepare_draw(
                view,
                ForwardDrawSource::Feature {
                    feature,
                    local_id: request.local_id,
                },
                request.order,
                &request.resources,
                request.transform,
                assets,
                composer,
                environment,
                specular,
            )
        })
        .collect()
}
