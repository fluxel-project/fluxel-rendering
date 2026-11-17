//! Source-bound grid controls and independent pre-submission image expectations.

use super::*;

struct Case {
    name: &'static str,
    options: Options,
    input: inputs::GridInput,
    expected: oracle::RasterImage,
    probes: Vec<usize>,
}

pub(super) fn run_offscreen<B>(device: Device<B>, options: &Options) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    // All independent expectations are retained before creating Renderer or
    // submitting any of the controls. A failed control cannot erase its input.
    let mut cases = Vec::new();
    for (name, time, reverse) in [
        ("grid-primary", options.time, options.reverse),
        ("grid-reversed", options.time, !options.reverse),
        ("grid-animated", options.time + 0.75, options.reverse),
        (
            "grid-animated-reversed",
            options.time + 0.75,
            !options.reverse,
        ),
        ("overlap-far-first", options.time, false),
        ("overlap-near-first", options.time, true),
    ] {
        let case_options = Options {
            mode: Mode::Offscreen,
            output: options.output.join(name),
            time,
            reverse,
            exit_after_capture: true,
        };
        fs::create_dir_all(&case_options.output)?;
        let overlap = name.starts_with("overlap");
        let input = if overlap {
            inputs::overlap(reverse)?
        } else {
            inputs::grid(time, reverse)?
        };
        let expected = if overlap {
            let mut image = oracle::RasterImage::clear(EXTENT);
            image.triangle(
                [
                    [-0.8, -0.8, 0.25, 1.],
                    [0.8, -0.8, 0.25, 1.],
                    [0., 0.8, 0.25, 1.],
                ],
                [[0., 1., 1.]; 3],
            );
            image
        } else {
            oracle::grid_reference(EXTENT, input.camera, &input.transforms)
        };
        write_inputs(&case_options.output, &input, &case_options)?;
        fs::write(case_options.output.join("expected.rgba"), &expected.rgba)?;
        let probes = if overlap {
            [[256, 256], [220, 300], [292, 300], [256, 180]]
                .map(|[x, y]| (y * EXTENT[0] + x) as usize)
                .to_vec()
        } else {
            stable_probes(&expected)?
        };
        write_inputs(&case_options.output, &input, &case_options)?;
        fs::write(case_options.output.join("expected.rgba"), &expected.rgba)?;
        fs::write(
            case_options.output.join("oracle-contract.txt"),
            format!(
                "independent CPU projected RGB/depth; screen-linear NDC depth; perspective-correct RGB; tolerance=2; stable interior pixel indices={probes:?}\n"
            ),
        )?;
        cases.push(Case {
            name,
            options: case_options,
            input,
            expected,
            probes,
        });
    }
    let mut renderer = Renderer::new(device)?;
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal();
    let mut outputs = Vec::new();
    for case in &cases {
        let before_draws = renderer.forward_cache_stats().opaque_draws;
        let submission = block_on(renderer.submit_forward(
            &pipeline,
            case.input.snapshot.clone(),
            case.input.view,
            case.input.bounds,
            &case.input.assets,
        ))?;
        let rgba = block_on(renderer.readback(&submission))?;
        fs::write(case.options.output.join("forward-readback.rgba"), &rgba)?;
        if renderer.pending() != 0 {
            return Err("grid readback retained pending work".into());
        }
        verify(&rgba, &case.expected.rgba, &case.probes)?;
        let stats = renderer.forward_cache_stats();
        if stats.opaque_draws - before_draws != case.input.transforms.len() as u64 {
            return Err("actual accepted draw count differs from saved scene records".into());
        }
        fs::write(
            case.options.output.join("result.txt"),
            format!(
                "{}: independent probes passed; cumulative actual opaque draws={}\n",
                case.name, stats.opaque_draws
            ),
        )?;
        outputs.push(rgba);
    }
    // Exact order reversal is stronger than comparing two independent small
    // probe lists. These controls preserve geometry and vary actual record order.
    if outputs[0] != outputs[1] || outputs[2] != outputs[3] || outputs[4] != outputs[5] {
        return Err("reversing draw order changed the depth-tested grid".into());
    }
    if outputs[0] == outputs[2] {
        return Err("time control did not change the grid".into());
    }
    fs::write(
        options.output.join("result.json"),
        "{\"schemaVersion\":1,\"gridCases\":4,\"drawsPerCase\":125,\"independentInteriorRgbTolerance\":2,\"orderReversal\":\"exact\",\"timeControl\":\"different\",\"overlapCases\":2,\"scope\":\"Forward grid and two-colour depth controls\"}\n",
    )?;
    Ok(())
}

pub(super) fn stable_probes(image: &oracle::RasterImage) -> Result<Vec<usize>, Box<dyn Error>> {
    let expected = &image.rgba;
    let [width, height] = EXTENT;
    let mut probes = Vec::new();
    for row in 0..4 {
        for column in 0..4 {
            'candidate: for y in row * height / 4 + 3..(row + 1) * height / 4 - 3 {
                for x in column * width / 4 + 3..(column + 1) * width / 4 - 3 {
                    let at = (y * width + x) as usize;
                    let center = &expected[at * 4..at * 4 + 4];
                    if center[3] != 255 || center[..3].iter().all(|c| *c < 20) {
                        continue;
                    }
                    let stable = (-2i32..=2).all(|dy| {
                        (-2i32..=2).all(|dx| {
                            let other = ((y as i32 + dy) * width as i32 + x as i32 + dx) as usize;
                            image.owners[other] == image.owners[at]
                        })
                    });
                    if stable {
                        probes.push(at);
                        break 'candidate;
                    }
                }
            }
        }
    }
    if probes.len() < 8 {
        return Err("CPU reference has insufficient separated stable grid probes".into());
    }
    Ok(probes)
}

pub(super) fn verify(raw: &[u8], expected: &[u8], probes: &[usize]) -> Result<(), Box<dyn Error>> {
    if raw.len() != expected.len() {
        return Err("grid readback byte count mismatch".into());
    }
    for &pixel in probes {
        for c in 0..3 {
            if raw[pixel * 4 + c].abs_diff(expected[pixel * 4 + c]) > 2 {
                return Err(format!(
                    "independent grid RGB mismatch at ({},{}), found {:?}, expected {:?}",
                    pixel % EXTENT[0] as usize,
                    pixel / EXTENT[0] as usize,
                    &raw[pixel * 4..pixel * 4 + 4],
                    &expected[pixel * 4..pixel * 4 + 4]
                )
                .into());
            }
        }
    }
    Ok(())
}
