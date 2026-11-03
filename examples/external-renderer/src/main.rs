//! Independent CPU reference for the public RenderScene batch boundary.

mod adapters;

use adapters::{EcsStyleAdapter, ObjectAdapter, ReferenceAdapter};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (direct, _) = run(
        Path::new("target/external-renderer-evidence"),
        ReferenceAdapter::expected_evidence(),
    )?;
    println!(
        "external-renderer CPU reference published scene versions {} and {}",
        direct.initial.version, direct.updated.version
    );
    Ok(())
}

fn run(
    evidence: &Path,
    expected: adapters::EvidenceFacts,
) -> Result<(adapters::ReferenceResult, adapters::ReferenceResult), Box<dyn std::error::Error>> {
    let direct_attempt = ReferenceAdapter::begin_evidence(
        evidence,
        "object-adapter",
        "two object-map records, one view, and one material replacement",
        expected.clone(),
    )?;
    let direct = match ObjectAdapter::reference_scene() {
        Ok(value) => value,
        Err(error) => {
            ReferenceAdapter::fail_evidence(&direct_attempt, &error)?;
            return Err(error);
        }
    };
    ReferenceAdapter::complete_evidence(&direct_attempt, &direct)?;

    let ecs_attempt = ReferenceAdapter::begin_evidence(
        evidence,
        "ecs-adapter",
        "two ECS-shaped component rows, one view, and one material replacement",
        expected,
    )?;
    let ecs = match EcsStyleAdapter::reference_scene() {
        Ok(value) => value,
        Err(error) => {
            ReferenceAdapter::fail_evidence(&ecs_attempt, &error)?;
            return Err(error);
        }
    };
    ReferenceAdapter::complete_evidence(&ecs_attempt, &ecs)?;
    ReferenceAdapter::assert_equivalent(&direct, &ecs)?;
    Ok((direct, ecs))
}

#[cfg(test)]
#[path = "tests/contract.rs"]
mod tests;
