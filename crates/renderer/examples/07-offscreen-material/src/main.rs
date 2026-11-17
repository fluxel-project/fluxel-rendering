//! Same-Graph captured-texture Material consumer with exact local asset admission.
//!
//! Native execution renders the immutable producer/receiver inputs through
//! Renderer. Visible mode captures the real window and compares it with a
//! separately retained same-input offscreen companion.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use renderer_07_offscreen_material::gltf;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;

const RHI_REVISION: &str = "ebdcb614140296197fcf9abe2924ab0faf5cc490";
const DRAGON_SHA256: &str = "8E69E1A2337BABF6A508BFA44D7AED41F92D85B04C9C3C2EEB5BA4059509A2E5";
const PLANE_SHA256: &str = "F77C1B7D558A26610CDE672EFC53EAC8975CA68AAB631EFE8AAC6D34EE98E10E";

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args()?;
    let dragon_hash = verify_gltf("dragon", &options.dragon, DRAGON_SHA256)?;
    let plane_hash = verify_gltf("plane", &options.plane, PLANE_SHA256)?;
    let dragon = gltf::load(&fs::read(&options.dragon)?)?;
    let plane = gltf::load(&fs::read(&options.plane)?)?;
    fs::create_dir_all(&options.output)?;
    fs::write(
        options.output.join("source.json"),
        format!(
            "{{\"schemaVersion\":1,\"workload\":\"renderer-07-offscreen-material\",\"referenceRhiRevision\":\"{RHI_REVISION}\",\"dragon\":{{\"path\":{:?},\"sha256\":\"{dragon_hash}\"}},\"plane\":{{\"path\":{:?},\"sha256\":\"{plane_hash}\"}}}}\n",
            options.dragon, options.plane
        ),
    )?;
    write_source_snapshot(&options.output)?;
    #[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
    return native::run(
        options.mode,
        options.exit_after_capture,
        dragon,
        plane,
        &options.output,
    );
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    let _ = (dragon, plane);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    Err("Renderer 07 execution requires Windows with --features dx12 or native-gl-wgl".into())
}

fn write_source_snapshot(output: &Path) -> Result<(), Box<dyn Error>> {
    for (name, bytes) in [
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("Cargo.lock", include_bytes!("../Cargo.lock").as_slice()),
        ("source-main.rs", include_bytes!("main.rs").as_slice()),
        ("source-native.rs", include_bytes!("native.rs").as_slice()),
        (
            "source-diagnostics.rs",
            include_bytes!("../../rendering-integration/src/diagnostics.rs").as_slice(),
        ),
        (
            "source-window-oracle.rs",
            include_bytes!("window_oracle.rs").as_slice(),
        ),
        ("source-runtime.rs", include_bytes!("runtime.rs").as_slice()),
        ("source-inputs.rs", include_bytes!("inputs.rs").as_slice()),
        (
            "source-evidence.rs",
            include_bytes!("evidence.rs").as_slice(),
        ),
        ("source-oracle.rs", include_bytes!("oracle.rs").as_slice()),
        ("source-scene.rs", include_bytes!("scene.rs").as_slice()),
        ("source-gltf.rs", include_bytes!("gltf.rs").as_slice()),
    ] {
        fs::write(output.join(name), bytes)?;
    }
    Ok(())
}

#[cfg_attr(
    not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))),
    allow(dead_code)
)]
struct Options {
    mode: Mode,
    exit_after_capture: bool,
    dragon: PathBuf,
    plane: PathBuf,
    output: PathBuf,
}

#[derive(Clone, Copy)]
enum Mode {
    Offscreen,
    Visible,
}

fn parse_args() -> Result<Options, Box<dyn Error>> {
    let mut dragon = None;
    let mut plane = None;
    let mut output = PathBuf::from("target/renderer-07-offscreen-material");
    let mut mode = None;
    let mut exit_after_capture = false;
    let mut args = env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_string_lossy().as_ref() {
            "--offscreen" => mode = Some(Mode::Offscreen),
            "--visible" => mode = Some(Mode::Visible),
            "--exit-after-capture" => exit_after_capture = true,
            "--dragon" => dragon = Some(args.next().ok_or("--dragon requires a path")?.into()),
            "--plane" => plane = Some(args.next().ok_or("--plane requires a path")?.into()),
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => {
                return Err(format!("unsupported argument: {}", argument.to_string_lossy()).into());
            }
        }
    }
    Ok(Options {
        mode: mode.ok_or("select exactly one of --offscreen or --visible")?,
        exit_after_capture,
        dragon: dragon.ok_or("--dragon is required; this asset is not redistributed")?,
        plane: plane.ok_or("--plane is required; this asset is not redistributed")?,
        output,
    })
}

fn verify_gltf(name: &str, path: &Path, expected: &str) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if !bytes.starts_with(b"{")
        || !bytes
            .windows(b"\"version\" : \"2.0\"".len())
            .any(|v| v == b"\"version\" : \"2.0\"")
    {
        return Err(format!("{name} must be an embedded glTF 2.0 JSON source").into());
    }
    let actual = sha256(&bytes);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(format!("{name} SHA-256 mismatch: expected {expected}, got {actual}").into());
    }
    Ok(actual)
}

fn sha256(input: &[u8]) -> String {
    const H: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&((input.len() as u64).wrapping_mul(8)).to_be_bytes());
    let mut h = H;
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().expect("chunk"));
        }
        for i in 16..64 {
            let x = w[i - 15];
            let y = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut z) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let t1 = z
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ ((!e) & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            z = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in h.iter_mut().zip([a, b, c, d, e, f, g, z]) {
            *state = state.wrapping_add(value);
        }
    }
    h.into_iter().map(|word| format!("{word:08X}")).collect()
}

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
fn json_string(value: &str) -> String {
    format!("{:?}", value)
}
