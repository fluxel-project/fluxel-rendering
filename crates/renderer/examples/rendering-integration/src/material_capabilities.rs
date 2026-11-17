//! M0 material-path capability probe.
//!
//! This fixture records the selected device's answers for the resource shapes
//! required by the first PBR material milestone.  It deliberately records an
//! unsupported answer instead of guessing from a backend name: the later
//! renderer admission path must refuse the dependent allocation or recording.

use std::{error::Error, fs, ops::Range, path::Path};

use fluxel_rhi::{
    api::{
        AddressMode, Device, FilterMode, SamplerDesc, Support, TextureDesc, TextureDimension,
        TextureFormat, TextureUsage,
    },
    backend::ResourceBackend,
};

pub(crate) fn run<B: ResourceBackend>(
    device: &Device<B>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;

    let candidates = [
        ("surface_color", color_desc("M0 surface color", 1)),
        ("gbuffer", float_color_desc("M0 G-buffer", 1)),
        ("scene_capture", capture_desc("M0 scene capture")),
        ("depth_sample", depth_desc("M0 sampled depth")),
        (
            "peel_accumulation",
            float_color_desc("M0 peel accumulation", 1),
        ),
        ("multisample_color", color_desc("M0 multisample color", 4)),
        ("resolve_target", color_desc("M0 resolve target", 1)),
    ];
    let rows = candidates
        .iter()
        .map(|(name, desc)| support_row(name, desc, device))
        .collect::<Vec<_>>();

    let invalid_transient = TextureDesc {
        label: Some("M0 invalid transient negative control".into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TRANSIENT_ATTACHMENT
            | TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    };
    let invalid_transient_reason = match device.texture_support(&invalid_transient) {
        Support::Unsupported(reason) => reason,
        Support::Supported(_) => {
            return Err("invalid transient attachment descriptor was admitted".into());
        }
    };

    let sampler = SamplerDesc {
        label: Some("M0 linear material sampler".into()),
        address: [AddressMode::Repeat; 3],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        lod: Range {
            start: 0.0,
            end: 16.0,
        },
        anisotropy: 1,
        compare: None,
        border: None,
    };
    let sampler_status = match device.create_sampler(&sampler) {
        Ok(_) => "{\"status\":\"supported\"}".to_owned(),
        Err(error) => format!(
            "{{\"status\":\"unsupported\",\"reason\":\"{}\"}}",
            escape(&error.to_string())
        ),
    };
    let mrt_required = 3_u32;
    let mrt_status = if device.capabilities().limits.max_color_attachments >= mrt_required {
        "supported"
    } else {
        "unsupported"
    };
    fs::write(
        output.join("m0-material-capabilities.json"),
        format!(
            concat!(
                "{{\n  \"schemaVersion\": 1,\n",
                "  \"probe\": \"m0-material-resource-admission\",\n",
                "  \"deviceId\": \"{:?}\",\n",
                "  \"requirements\": [{}],\n",
                "  \"sampler\": {{\"linearMipmapped\": {}}},\n",
                "  \"mrt\": {{\"requiredColorAttachments\": {}, \"reportedLimit\": {}, \"status\": \"{}\"}},\n",
                "  \"negativeControls\": {{\"invalidTransientAttachment\": {{\"status\": \"rejected-before-allocation\", \"reason\": \"{}\"}}}},\n",
                "  \"notes\": [\"Texture support is a device fact; blendability is reported per color format. A successful multisample source and single-sample target does not itself prove a resolve pass.\"]\n",
                "}}\n"
            ),
            device.id(),
            rows.join(","),
            sampler_status,
            mrt_required,
            device.capabilities().limits.max_color_attachments,
            mrt_status,
            escape(invalid_transient_reason),
        ),
    )?;
    Ok(())
}

fn color_desc(label: &str, samples: u32) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn float_color_desc(label: &str, samples: u32) -> TextureDesc {
    TextureDesc {
        format: TextureFormat::Rgba16Float,
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn capture_desc(label: &str) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::TEXTURE_BINDING
            | TextureUsage::COPY_SRC,
        view_formats: Vec::new(),
    }
}

fn depth_desc(label: &str) -> TextureDesc {
    TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn support_row<B: ResourceBackend>(name: &str, desc: &TextureDesc, device: &Device<B>) -> String {
    match device.texture_support(desc) {
        Support::Supported(support) => match device.create_texture(desc) {
            Ok(_) => format!(
                "{{\"name\":\"{name}\",\"format\":\"{:?}\",\"samples\":{},\"status\":\"supported-and-allocated\",\"filterable\":{},\"blendable\":{},\"storage\":{},\"sampleCounts\":[{}]}}",
                desc.format,
                desc.samples,
                support.filterable,
                support.blendable,
                support.storage,
                support
                    .sample_counts
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Err(error) => format!(
                "{{\"name\":\"{name}\",\"format\":\"{:?}\",\"samples\":{},\"status\":\"allocation-rejected\",\"reason\":\"{}\"}}",
                desc.format,
                desc.samples,
                escape(&error.to_string())
            ),
        },
        Support::Unsupported(reason) => format!(
            "{{\"name\":\"{name}\",\"format\":\"{:?}\",\"samples\":{},\"status\":\"unsupported\",\"reason\":\"{}\"}}",
            desc.format,
            desc.samples,
            escape(reason)
        ),
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
