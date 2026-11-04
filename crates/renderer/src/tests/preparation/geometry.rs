use super::{
    GeometryAttribute, GeometryBufferOwner, GeometryDraw, GeometryError, GeometryLayout,
    GeometryStream, GeometryStreamBinding, PreparedGeometry,
};
use fluxel_rhi::api::{
    AccessMask, BufferDesc, BufferUsage, BufferUses, DeviceId, MemoryClass, StageMask,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
use fluxel_shader::{ScalarType, VertexInputDeclaration, VertexType};

fn requirement(semantic: &str, location: u32, ty: VertexType) -> VertexInputDeclaration {
    VertexInputDeclaration::new(semantic, location, ty)
}

fn attribute(semantic: &str, offset: u64, format: VertexFormat) -> GeometryAttribute {
    GeometryAttribute {
        semantic: semantic.into(),
        offset,
        format,
    }
}

fn stream(stride: u64, attributes: Vec<GeometryAttribute>) -> GeometryStream {
    GeometryStream {
        stride,
        step_mode: VertexStepMode::Vertex,
        attributes,
    }
}

fn owner(device: DeviceId, size: u64, usage: BufferUsage) -> GeometryBufferOwner {
    GeometryBufferOwner::new(
        device,
        BufferDesc {
            label: None,
            size,
            usage,
            memory: MemoryClass::Device,
        },
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
    )
}

fn binding(owner: GeometryBufferOwner, start: u64, end: u64) -> GeometryStreamBinding {
    GeometryStreamBinding {
        owner,
        range: start..end,
    }
}

#[test]
fn preparation_resolves_interleaved_separate_and_mixed_streams() {
    let requirements = [
        requirement("POSITION", 0, VertexType::Vec3(ScalarType::F32)),
        requirement("UV", 1, VertexType::Vec2(ScalarType::F32)),
        requirement("COLOR", 2, VertexType::Vec4(ScalarType::F32)),
    ];
    let interleaved = GeometryLayout {
        streams: vec![stream(
            36,
            vec![
                attribute("POSITION", 0, VertexFormat::Float32(3)),
                attribute("UV", 12, VertexFormat::Float32(2)),
                attribute("COLOR", 20, VertexFormat::Float32(4)),
            ],
        )],
    };
    let separate = GeometryLayout {
        streams: vec![
            stream(12, vec![attribute("POSITION", 0, VertexFormat::Float32(3))]),
            stream(8, vec![attribute("UV", 0, VertexFormat::Float32(2))]),
            stream(16, vec![attribute("COLOR", 0, VertexFormat::Float32(4))]),
        ],
    };
    let mixed = GeometryLayout {
        streams: vec![
            stream(
                20,
                vec![
                    attribute("POSITION", 0, VertexFormat::Float32(3)),
                    attribute("UV", 12, VertexFormat::Float32(2)),
                ],
            ),
            stream(16, vec![attribute("COLOR", 0, VertexFormat::Float32(4))]),
        ],
    };
    for layout in [interleaved, separate, mixed] {
        let prepared = layout.prepare(&requirements).expect("compatible geometry");
        let locations: Vec<_> = prepared
            .streams
            .iter()
            .flat_map(|stream| stream.attributes.iter().map(|attribute| attribute.location))
            .collect();
        assert_eq!(locations, [0, 1, 2]);
    }
}

#[test]
fn preparation_rejects_missing_mismatched_and_invalid_attributes() {
    let position = requirement("POSITION", 0, VertexType::Vec3(ScalarType::F32));
    assert_eq!(
        GeometryLayout::default()
            .prepare(&[position.clone()])
            .unwrap_err(),
        GeometryError::MissingStreams
    );
    let mismatch = GeometryLayout {
        streams: vec![stream(
            8,
            vec![attribute("POSITION", 0, VertexFormat::Float32(2))],
        )],
    };
    assert_eq!(
        mismatch.prepare(&[position.clone()]).unwrap_err(),
        GeometryError::TypeMismatch {
            semantic: "POSITION".into(),
            location: 0
        }
    );
    let invalid = GeometryLayout {
        streams: vec![stream(
            12,
            vec![attribute("POSITION", 4, VertexFormat::Float32(3))],
        )],
    };
    assert_eq!(
        invalid.prepare(&[position]).unwrap_err(),
        GeometryError::AttributeOutOfBounds {
            stream: 0,
            semantic: "POSITION".into()
        }
    );
}

#[test]
fn preparation_rejects_duplicate_semantics() {
    let duplicate = GeometryLayout {
        streams: vec![
            stream(12, vec![attribute("POSITION", 0, VertexFormat::Float32(3))]),
            stream(12, vec![attribute("POSITION", 0, VertexFormat::Float32(3))]),
        ],
    };
    assert_eq!(
        duplicate.prepare(&[]).unwrap_err(),
        GeometryError::DuplicateSemantic {
            semantic: "POSITION".into()
        }
    );
}

#[test]
fn bindings_use_owner_facts_and_validate_nonzero_base_footprints() {
    let layout = GeometryLayout {
        streams: vec![stream(
            16,
            vec![attribute("POSITION", 4, VertexFormat::Float32(3))],
        )],
    };
    let prepared = layout
        .prepare(&[requirement(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        )])
        .expect("logical layout");
    let device = DeviceId::allocate().expect("device id");
    let bindings = [binding(owner(device, 96, BufferUsage::VERTEX), 32, 80)];
    let draw = GeometryDraw {
        vertices: 1..3,
        instances: 0..1,
    };
    let bound = prepared
        .bind(device, &bindings, draw.clone())
        .expect("final position fetch ends exactly at the concrete binding end");
    assert_eq!(bound.draw(), &draw);
    assert_eq!(bound.accessed_ranges(), &[Some(52..80)]);
    assert_eq!(bound.bindings()[0].owner.desc().size, 96);
    assert_eq!(
        bound.bindings()[0].owner.incoming(),
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        }
    );
    assert_eq!(bound.geometry().streams.len(), 1);

    let short = [binding(owner(device, 96, BufferUsage::VERTEX), 32, 79)];
    assert_eq!(
        prepared.bind(device, &short, draw).unwrap_err(),
        GeometryError::DrawFootprintOutsideBinding { stream: 0 }
    );
}

#[test]
fn bindings_reject_foreign_or_incompatible_owner_facts() {
    let layout = GeometryLayout {
        streams: vec![stream(
            12,
            vec![attribute("POSITION", 0, VertexFormat::Float32(3))],
        )],
    };
    let prepared = layout
        .prepare(&[requirement(
            "POSITION",
            0,
            VertexType::Vec3(ScalarType::F32),
        )])
        .expect("logical layout");
    let device = DeviceId::allocate().expect("device id");
    let foreign = DeviceId::allocate().expect("foreign device id");
    let draw = GeometryDraw {
        vertices: 0..1,
        instances: 0..1,
    };
    assert_eq!(
        prepared
            .bind(
                device,
                &[binding(owner(foreign, 12, BufferUsage::VERTEX), 0, 12)],
                draw.clone(),
            )
            .unwrap_err(),
        GeometryError::ForeignDevice { stream: 0 }
    );
    assert_eq!(
        prepared
            .bind(
                device,
                &[binding(owner(device, 12, BufferUsage::COPY_DST), 0, 12)],
                draw.clone(),
            )
            .unwrap_err(),
        GeometryError::MissingVertexUsage { stream: 0 }
    );
    assert_eq!(
        prepared
            .bind(
                device,
                &[binding(owner(device, 12, BufferUsage::VERTEX), 1, 13)],
                draw,
            )
            .unwrap_err(),
        GeometryError::BindingRangeOutsideOwner { stream: 0 }
    );
}

#[test]
fn binding_rejects_publicly_malformed_prepared_layouts() {
    let device = DeviceId::allocate().expect("device id");
    let malformed = PreparedGeometry {
        streams: vec![VertexBufferLayout {
            stride: 4,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 2,
                format: VertexFormat::Float32(1),
            }],
        }],
    };
    assert_eq!(
        malformed
            .bind(
                device,
                &[binding(owner(device, 8, BufferUsage::VERTEX), 0, 8)],
                GeometryDraw {
                    vertices: 0..1,
                    instances: 0..1,
                },
            )
            .unwrap_err(),
        GeometryError::InvalidPreparedLayout { stream: 0 }
    );
}
