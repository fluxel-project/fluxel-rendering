use super::*;

fn max_error(a: &[Probe], b: &[Probe]) -> u8 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.expected[..3].iter().zip(&b.expected[..3]))
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0)
}

#[test]
fn controls_are_independent_and_reject_baseline_reuse() {
    let baseline = input_with_receiver(Case::Baseline, Case::Baseline).unwrap();
    let expected = reference(&baseline).unwrap();
    for (_, producer, receiver) in CONTROLS.into_iter().skip(1) {
        let changed = input_with_receiver(producer, receiver).unwrap();
        let probes = reference(&changed).unwrap();
        assert!(
            max_error(&expected, &probes) > crate::oracle::TOLERANCE,
            "producer={producer:?}, receiver={receiver:?} accepts baseline"
        );
        assert_ne!(
            baseline.producer.assets.textures[0].id(),
            changed.receiver_images[0].id(),
            "producer and receiver must never share an asset identity"
        );
    }
}

#[test]
fn complete_capture_reference_is_nonconstant_and_not_sparse() {
    let input = input(Case::Baseline).unwrap();
    let image = cpu_capture_reference(&input).unwrap();
    assert_eq!(
        image.len(),
        (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize
    );
    let nonzero = image
        .chunks_exact(4)
        .filter(|pixel| pixel[..3] != [0; 3])
        .count();
    assert!(
        nonzero > image.len() / 32,
        "capture reference is sparse or clear"
    );
    assert!(image.chunks_exact(4).any(|pixel| pixel[..3] != image[..3]));
}

#[test]
fn wrong_capture_and_wrong_srgb_decode_exceed_acceptance_tolerance() {
    let input = input(Case::Baseline).unwrap();
    let expected = reference(&input).unwrap();
    let mut clear = cpu_capture_reference(&input).unwrap();
    clear.fill(0);
    let wrong_capture = reference_from_capture(&input, &clear, false).unwrap();
    assert!(max_error(&expected, &wrong_capture) > crate::oracle::TOLERANCE);
    let capture = cpu_capture_reference(&input).unwrap();
    let wrong_decode = reference_from_capture(&input, &capture, true).unwrap();
    assert!(max_error(&expected, &wrong_decode) > crate::oracle::TOLERANCE);
    let wrong_mapping =
        reference_with_receiver_mapping(&input, &capture, false, [1, 0, 2]).unwrap();
    assert!(max_error(&expected, &wrong_mapping) > crate::oracle::TOLERANCE);
}

#[test]
fn saved_input_manifest_is_json_and_contains_actual_owner_facts() {
    let input = input(Case::Baseline).unwrap();
    let probes = reference(&input).unwrap();
    let path = std::env::temp_dir().join(format!(
        "fluxel-mixed-capture-manifest-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    write_inputs(&path, &input, &probes).unwrap();
    let bytes = std::fs::read(path.join("mixed-input.json")).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_json(text);
    assert!(text.contains("producerSlot"));
    assert!(text.contains("generation"));
    assert!(text.contains("GeometryLayout"));
    std::fs::remove_dir_all(path).unwrap();
}

fn assert_json(text: &str) {
    fn ws(bytes: &[u8], index: &mut usize) {
        while bytes.get(*index).is_some_and(u8::is_ascii_whitespace) {
            *index += 1;
        }
    }
    fn string(bytes: &[u8], index: &mut usize) {
        assert_eq!(bytes.get(*index), Some(&b'\"'));
        *index += 1;
        while let Some(&byte) = bytes.get(*index) {
            *index += 1;
            match byte {
                b'\"' => return,
                b'\\' => *index += 1,
                0..=31 => panic!("unescaped JSON control"),
                _ => {}
            }
        }
        panic!("unterminated JSON string");
    }
    fn value(bytes: &[u8], index: &mut usize) {
        ws(bytes, index);
        match bytes.get(*index).copied() {
            Some(b'\"') => string(bytes, index),
            Some(b'{') => {
                *index += 1;
                ws(bytes, index);
                if bytes.get(*index) == Some(&b'}') {
                    *index += 1;
                    return;
                }
                loop {
                    ws(bytes, index);
                    string(bytes, index);
                    ws(bytes, index);
                    assert_eq!(bytes.get(*index), Some(&b':'));
                    *index += 1;
                    value(bytes, index);
                    ws(bytes, index);
                    match bytes.get(*index) {
                        Some(b',') => *index += 1,
                        Some(b'}') => {
                            *index += 1;
                            return;
                        }
                        _ => panic!("invalid JSON object"),
                    }
                }
            }
            Some(b'[') => {
                *index += 1;
                ws(bytes, index);
                if bytes.get(*index) == Some(&b']') {
                    *index += 1;
                    return;
                }
                loop {
                    value(bytes, index);
                    ws(bytes, index);
                    match bytes.get(*index) {
                        Some(b',') => *index += 1,
                        Some(b']') => {
                            *index += 1;
                            return;
                        }
                        _ => panic!("invalid JSON array"),
                    }
                }
            }
            Some(_) => {
                let start = *index;
                while bytes
                    .get(*index)
                    .is_some_and(|b| !b" \t\r\n,]}".contains(b))
                {
                    *index += 1;
                }
                assert!(*index > start, "missing JSON primitive");
            }
            None => panic!("missing JSON value"),
        }
    }
    let bytes = text.as_bytes();
    let mut index = 0;
    value(bytes, &mut index);
    ws(bytes, &mut index);
    assert_eq!(index, bytes.len());
}
