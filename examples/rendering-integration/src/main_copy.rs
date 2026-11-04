//! Copy workload fixture and its independent evidence.
use super::main_support::*;
use super::*;

pub(super) fn run_copy<B>(
    device: Device<B>,
    adapter: &str,
    output: &Path,
) -> Result<fluxel_renderer::Submission, Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    fs::create_dir_all(output)?;
    write_bytes(&output.join("selected-adapter.txt"), adapter.as_bytes())?;

    let store = AssetStore::<Bytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    // The payloads deliberately start at nonzero, changing offsets in source
    // buffers with asymmetric sentinel prefixes and suffixes. Compatible Copy
    // uses must refresh frame bindings while retaining one compiled topology.
    let inputs = [
        framed_copy_source(&[0xa1, 0xa2, 0xa3, 0xa4], 17, 16, &[0xb1, 0xb2, 0xb3]),
        framed_copy_source(
            &[0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8],
            18,
            24,
            &[0xd1, 0xd2, 0xd3, 0xd4, 0xd5],
        ),
        framed_copy_source(
            &[0xe1, 0xe2, 0xe3, 0xe4],
            19,
            12,
            &[0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6],
        ),
    ];
    let source_ranges: [Range<u64>; 3] = [4..20, 8..32, 4..16];
    let expected = [vec![17; 16], vec![18; 24], vec![19; 12]];
    for index in 0..inputs.len() {
        write_bytes(&output.join(format!("input-{index}.bin")), &inputs[index])?;
        write_bytes(
            &output.join(format!("expected-{index}.bin")),
            &expected[index],
        )?;
    }
    let first = producer(&store, &handle, inputs[0].clone())?;
    let second = replacement(&store, &handle, inputs[1].clone())?;
    let third = replacement(&store, &handle, inputs[2].clone())?;
    let generations = [
        first.generation().get(),
        second.generation().get(),
        third.generation().get(),
    ];
    let snapshots = [
        Arc::downgrade(&first.value_arc()),
        Arc::downgrade(&second.value_arc()),
        Arc::downgrade(&third.value_arc()),
    ];
    if first.id() != second.id()
        || second.id() != third.id()
        || !(first.generation() < second.generation() && second.generation() < third.generation())
    {
        return Err("asset replacement did not preserve identity and advance generations".into());
    }

    let mut renderer = Renderer::new(device.clone())?;
    let before = device.stats().snapshot();
    let empty_store = AssetStore::<Bytes, Vec<u8>, String>::new();
    let empty_handle = empty_store.create()?;
    let empty = producer(&empty_store, &empty_handle, Vec::new())?;
    if !matches!(
        block_on(renderer.submit(empty)),
        Err(RendererError::EmptyAsset)
    ) {
        return Err("empty asset did not receive the structured EmptyAsset refusal".into());
    }
    if device.stats().snapshot() != before {
        return Err("empty asset changed RHI request counters".into());
    }

    let submit_started = Instant::now();
    let first_submission = block_on(renderer.submit_copy_range(first, source_ranges[0].clone()))?;
    let plan = renderer
        .plan_id()
        .ok_or("copy plan identity was not retained")?;
    let submissions = [
        first_submission,
        block_on(renderer.submit_copy_range(second, source_ranges[1].clone()))?,
        block_on(renderer.submit_copy_range(third, source_ranges[2].clone()))?,
    ];
    let submit_elapsed = submit_started.elapsed();
    if renderer.pending() != 3
        || renderer.compilations() != 1
        || renderer.plan_id() != Some(plan)
        || renderer.binding_refreshes() != 3
    {
        return Err("three outstanding copy submissions did not retain one compiled plan".into());
    }

    let fourth = replacement(&store, &handle, vec![20; 20])?;
    let saturated_before = device.stats().snapshot();
    if !matches!(
        block_on(renderer.submit(fourth)),
        Err(RendererError::Backpressure)
    ) {
        return Err("slot saturation did not receive the structured Backpressure refusal".into());
    }
    if device.stats().snapshot() != saturated_before {
        return Err("backpressure refusal changed RHI request counters".into());
    }

    let mut other_renderer = Renderer::new(device.clone())?;
    if !matches!(
        block_on(other_renderer.readback(&submissions[0])),
        Err(RendererError::ForeignSubmission)
    ) {
        return Err("foreign renderer did not reject a submission before waiting".into());
    }

    drop(handle);
    drop(store);
    if snapshots
        .iter()
        .any(|snapshot| snapshot.upgrade().is_none())
    {
        return Err(
            "renderer did not retain every submitted source snapshot before completion".into(),
        );
    }
    let snapshot_retention_verified = true;
    let readback_started = Instant::now();
    for (index, submission) in submissions.iter().enumerate() {
        let readback = block_on(renderer.readback(submission))?;
        write_bytes(&output.join(format!("readback-{index}.bin")), &readback)?;
        if readback != expected[index] {
            return Err(format!(
                "readback {index} differs from its selected immutable payload range"
            )
            .into());
        }
        if snapshots[index].upgrade().is_some() {
            return Err(
                format!("snapshot {index} remained retained after its covering readback").into(),
            );
        }
        if snapshots[index + 1..]
            .iter()
            .any(|snapshot| snapshot.upgrade().is_none())
        {
            return Err(format!(
                "a later copy snapshot retired before its covering readback at frame {index}"
            )
            .into());
        }
    }
    let readback_elapsed = readback_started.elapsed();
    if renderer.pending() != 0 {
        return Err("all readbacks completed but renderer still retains submissions".into());
    }
    if !matches!(
        block_on(renderer.readback(&submissions[0])),
        Err(RendererError::StaleSubmission)
    ) {
        return Err(
            "retired submission did not receive the structured StaleSubmission refusal".into(),
        );
    }

    let after = device.stats().snapshot();
    let payload = format!(
        concat!(
            "{{\n",
            "  \"schemaVersion\": 1,\n",
            "  \"workload\": \"s0-copy-three-generations\",\n",
            "  \"adapter\": {:?},\n",
            "  \"assetGenerations\": [{}, {}, {}],\n",
            "  \"sourceByteLengths\": [{}, {}, {}],\n",
            "  \"sourceRanges\": [[{}, {}], [{}, {}], [{}, {}]],\n",
            "  \"outputByteLengths\": [{}, {}, {}],\n",
            "  \"submissions\": 3,\n",
            "  \"compiledPlans\": {},\n",
            "  \"frameBindingRefreshes\": {},\n",
            "  \"dynamicRangePlanReuseVerified\": true,\n",
            "  \"snapshotRetentionVerified\": {},\n",
            "  \"perFrameRetirementVerified\": true,\n",
            "  \"timingsNanoseconds\": {{\"submitThree\": {}, \"readbackThree\": {}}},\n",
            "  \"rhiCounters\": {{\"copies\": {}, \"submits\": {}}}\n",
            "}}\n"
        ),
        adapter,
        generations[0],
        generations[1],
        generations[2],
        inputs[0].len(),
        inputs[1].len(),
        inputs[2].len(),
        source_ranges[0].start,
        source_ranges[0].end,
        source_ranges[1].start,
        source_ranges[1].end,
        source_ranges[2].start,
        source_ranges[2].end,
        expected[0].len(),
        expected[1].len(),
        expected[2].len(),
        renderer.compilations(),
        renderer.binding_refreshes(),
        snapshot_retention_verified,
        submit_elapsed.as_nanos(),
        readback_elapsed.as_nanos(),
        after.copies.saturating_sub(before.copies),
        after.submits.saturating_sub(before.submits),
    );
    write_bytes(&output.join("workload.json"), payload.as_bytes())?;
    Ok(submissions[0].clone())
}
