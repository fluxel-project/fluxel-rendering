//! Version parsing shared by WGL and EGL discovery.

use crate::backend::gl::api::GlFamilyProfile;

/// Why a native GL version string cannot enter the accepted profile set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeProfileError {
    Empty,
    Malformed(String),
    BelowDesktopFloor { major: u8, minor: u8 },
    BelowEmbeddedFloor { major: u8, minor: u8 },
}

/// Parses the exact GL_VERSION spelling without treating a desktop string as
/// ES just because a vendor name happens to contain "ES".
pub(crate) fn parse_native_profile(version: &str) -> Result<GlFamilyProfile, NativeProfileError> {
    let value = version.trim();
    if value.is_empty() {
        return Err(NativeProfileError::Empty);
    }
    let (embedded, number) = if let Some(number) = value.strip_prefix("OpenGL ES ") {
        (true, number)
    } else {
        (false, value)
    };
    let number = number.split_whitespace().next().unwrap_or_default();
    let mut parts = number.split('.');
    let parse = |part: Option<&str>| part.and_then(|value| value.parse::<u8>().ok());
    let (Some(major), Some(minor)) = (parse(parts.next()), parse(parts.next())) else {
        return Err(NativeProfileError::Malformed(value.into()));
    };
    if embedded {
        if major < 3 {
            Err(NativeProfileError::BelowEmbeddedFloor { major, minor })
        } else {
            Ok(GlFamilyProfile::Embedded { major, minor })
        }
    } else if major < 4 {
        Err(NativeProfileError::BelowDesktopFloor { major, minor })
    } else {
        Ok(GlFamilyProfile::Desktop { major, minor })
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeProfileError, parse_native_profile};
    use crate::backend::gl::api::GlFamilyProfile;

    #[test]
    fn keeps_newer_versions_instead_of_downgrading_to_floor() {
        assert_eq!(
            parse_native_profile("4.6.0 NVIDIA 555"),
            Ok(GlFamilyProfile::Desktop { major: 4, minor: 6 })
        );
        assert_eq!(
            parse_native_profile("OpenGL ES 3.2 V@1"),
            Ok(GlFamilyProfile::Embedded { major: 3, minor: 2 })
        );
    }

    #[test]
    fn rejects_older_native_profiles_before_any_command_path() {
        assert_eq!(
            parse_native_profile("3.3"),
            Err(NativeProfileError::BelowDesktopFloor { major: 3, minor: 3 })
        );
        assert_eq!(
            parse_native_profile("OpenGL ES 2.0"),
            Err(NativeProfileError::BelowEmbeddedFloor { major: 2, minor: 0 })
        );
    }

    #[test]
    fn malformed_es_prefix_does_not_become_desktop_gl() {
        assert!(matches!(
            parse_native_profile("OpenGL ES-CM 1.1"),
            Err(NativeProfileError::Malformed(_))
        ));
    }
}
