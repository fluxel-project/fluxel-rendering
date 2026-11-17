use super::binary16_round;

#[test]
fn effects_half_rounding_uses_even_ties_and_preserves_subnormal_carry() {
    // Adjacent values above 0.5 are spaced by 2^-11. The halfway
    // positions must round to the endpoint with an even low mantissa bit.
    assert_eq!(binary16_round(0.5 + 2_f32.powi(-12)), 0.5);
    assert_eq!(
        binary16_round(0.5 + 3. * 2_f32.powi(-12)),
        0.5 + 2_f32.powi(-10)
    );
    assert_eq!(binary16_round(2_f32.powi(-25)), 0.);
    assert_eq!(binary16_round(3. * 2_f32.powi(-25)), 2_f32.powi(-23));
    assert_eq!(
        binary16_round(2_f32.powi(-14) - 2_f32.powi(-25)),
        2_f32.powi(-14)
    );
}
