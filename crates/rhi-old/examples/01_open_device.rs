//! Open and inspect one headless native device.
//!
//! Run on Windows with `cargo run -p fluxel-rhi --example 01_open_device -- dx12`
//! or replace `dx12` with `vulkan`.

use fluxel_rhi::{Backend, Device, DeviceOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = match std::env::args().nth(1).as_deref() {
        Some("dx12") => Backend::Dx12,
        Some("vulkan") => Backend::Vulkan,
        Some(other) => {
            return Err(format!("unknown backend `{other}`; use `dx12` or `vulkan`").into());
        }
        None => {
            return Err(
                "usage: cargo run -p fluxel-rhi --example 01_open_device -- <dx12|vulkan>".into(),
            );
        }
    };

    let device = Device::open(backend, DeviceOptions::default())?;
    println!("hardware:\n{:#?}", device.hardware());
    println!("capabilities:\n{:#?}", device.capabilities());
    Ok(())
}
