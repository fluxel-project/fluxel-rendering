use super::coalesce_adjacent;

#[test]
fn opaque_batches_preserve_overlapping_draw_order() {
    // The incompatible program is an intervening overlapping draw. The final
    // compatible item must form a new run instead of moving before it.
    let runs = coalesce_adjacent(
        vec![
            ("program-b", (9_u32, 1_u32)),
            ("program-a", (4, 2)),
            ("program-b", (3, 7)),
            ("program-a", (2, 9)),
        ],
        |entry| entry.0,
    );
    assert_eq!(
        runs,
        vec![
            ("program-b", vec![("program-b", (9, 1))]),
            ("program-a", vec![("program-a", (4, 2))]),
            ("program-b", vec![("program-b", (3, 7))]),
            ("program-a", vec![("program-a", (2, 9))]),
        ]
    );
}

#[test]
fn opaque_batches_merge_only_adjacent_compatible_programs() {
    let runs = coalesce_adjacent(vec![(1_u8, 0_u8), (1, 1), (2, 2), (2, 3)], |entry| entry.0);
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0], (1, vec![(1, 0), (1, 1)]));
    assert_eq!(runs[1], (2, vec![(2, 2), (2, 3)]));
}
