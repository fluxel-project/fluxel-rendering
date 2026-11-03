use super::{
    CpuPhase, CpuPhaseContributions, FeatureError, TypedFeatureSnapshot, validate_context,
};
use crate::RenderScene;

#[derive(Clone)]
struct R;
type Scene = RenderScene<R>;

#[test]
fn issuers_storage_and_contributions_reject_foreign_contexts() {
    let first = Scene::new().unwrap();
    let second = Scene::new().unwrap();
    let first_snapshot = first.snapshot();
    let second_snapshot = second.snapshot();
    let issuer_a = first.feature_issuer().unwrap();
    let issuer_b = first.feature_issuer().unwrap();
    let handle_a = issuer_a.issue("a").unwrap();
    let handle_b = issuer_b.issue("b").unwrap();
    let context_a = issuer_a
        .context(&first_snapshot, CpuPhase::Reference, handle_a)
        .unwrap();
    assert!(
        issuer_b
            .context(&first_snapshot, CpuPhase::Reference, handle_a)
            .is_err()
    );
    assert!(
        issuer_a
            .context(&second_snapshot, CpuPhase::Reference, handle_a)
            .is_err()
    );
    let foreign_issuer = second.feature_issuer().unwrap();
    let foreign_context = foreign_issuer
        .context(
            &second_snapshot,
            CpuPhase::Reference,
            foreign_issuer.issue("foreign").unwrap(),
        )
        .unwrap();
    let foreign = TypedFeatureSnapshot::for_context(&foreign_context, 1_u32);
    assert_eq!(
        validate_context(&context_a, &foreign, handle_a),
        Err(FeatureError::ForeignScene)
    );
    let mut storage = issuer_a.storage();
    assert!(storage.publish(&context_a, 1_u32).is_ok());
    let context_b = issuer_b
        .context(&first_snapshot, CpuPhase::Reference, handle_b)
        .unwrap();
    assert!(storage.publish(&context_b, 2_u32).is_err());
    let mut contributions = CpuPhaseContributions::default();
    let contribution = context_a.contribute();
    assert!(contributions.accept(&context_a, contribution).is_ok());
    assert!(contributions.accept(&context_b, contribution).is_err());
}
