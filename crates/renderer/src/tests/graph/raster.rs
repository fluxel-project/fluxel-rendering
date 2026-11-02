    use super::*;

    #[test]
    fn readback_rows_obey_the_portable_256_byte_alignment() {
        assert_eq!(aligned_row_pitch(1).unwrap(), 256);
        assert_eq!(aligned_row_pitch(64).unwrap(), 256);
        assert_eq!(aligned_row_pitch(65).unwrap(), 512);
        assert_eq!(aligned_row_pitch(u32::MAX), Err(Error::InvalidRange));
    }
