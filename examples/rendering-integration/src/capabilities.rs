//! Exhaustive, structured evidence for the selected RHI capability contract.
//!
//! The pinned RHI intentionally does not derive `Serialize` for its public
//! capability types. The exhaustive destructure below is therefore deliberate:
//! adding an RHI capability makes this consumer stop compiling until its
//! retained evidence schema is updated.

use std::{error::Error, fs, path::Path};

use fluxel_rhi::{
    api::{
        CooperativeMatrixComponentType, CooperativeMatrixConfiguration, CooperativeMatrixScope,
        Device, DeviceCapabilities, DeviceLimits, LaneClass, LaneId, LaneInfo, WaitRoute,
    },
    backend::Backend,
};

use crate::json_string;

macro_rules! boolean_capability_fields {
    ($visitor:ident) => {
        $visitor!(
            debug_markers,
            debug_markers_encoder,
            debug_markers_raster_pass,
            debug_markers_compute_pass,
            compute,
            sparse,
            mesh,
            acceleration_structure,
            acceleration_structure_binding_arrays,
            runtime_sized_acceleration_structure_binding_arrays,
            partially_bound_acceleration_structure_binding_arrays,
            ray,
            ray_query,
            shader_early_depth_test,
            shader_primitive_index,
            shader_draw_index,
            shader_barycentrics,
            shader_barycentrics_khr,
            shader_barycentrics_nv,
            shader_barycentrics_amd,
            shader_float32_atomic_buffer_add,
            shader_float32_atomic_workgroup_add,
            shader_float32_atomic_workgroup_min,
            shader_float32_atomic_workgroup_max,
            shader_float32_atomic_buffer_min,
            shader_float32_atomic_buffer_max,
            shader_int64_atomic_buffer_add,
            shader_int64_atomic_buffer_min,
            shader_int64_atomic_buffer_max,
            shader_int64_atomic_buffer_umin,
            shader_int64_atomic_buffer_umax,
            shader_int64_atomic_buffer_xor,
            shader_int64_atomic_buffer_compare_exchange,
            shader_int64_atomic_buffer_exchange,
            shader_int64_atomic_buffer_and,
            shader_int64_atomic_buffer_or,
            shader_int64_atomic_workgroup_add,
            shader_int64_atomic_workgroup_min,
            shader_int64_atomic_workgroup_max,
            shader_int64_atomic_workgroup_xor,
            shader_int64_atomic_workgroup_and,
            shader_int64_atomic_workgroup_or,
            shader_int64_atomic_workgroup_umin,
            shader_int64_atomic_workgroup_umax,
            shader_int64_atomic_workgroup_exchange,
            shader_int64_atomic_workgroup_compare_exchange,
            shader_int64_atomic_image_r64uint_add,
            shader_int64_atomic_image_r64uint_umin,
            shader_int64_atomic_image_r64uint_umax,
            shader_memory_coherent_spirv,
            shader_memory_volatile_spirv,
            multi_queue,
            indirect_draw,
            indirect_first_instance,
            multi_draw,
            multi_draw_count,
            mesh_multi_indirect,
            mesh_indirect_count,
            push_constants,
            storage_buffers,
            storage_textures,
            texture_atomic_r32uint_compute,
            texture_atomic_r32sint_compute,
            texture_atomic_r32sint_min_compute,
            texture_atomic_r32sint_max_compute,
            texture_atomic_r32uint_min_compute,
            texture_atomic_r32uint_max_compute,
            texture_atomic_r32uint_xor_compute,
            texture_atomic_r32uint_and_compute,
            texture_atomic_r32uint_or_compute,
            binding_arrays,
            runtime_sized_binding_arrays,
            nonuniform_sampled_texture_and_storage_buffer_indexing,
            nonuniform_uniform_buffer_indexing,
            nonuniform_storage_buffer_indexing,
            nonuniform_sampled_texture_indexing,
            nonuniform_storage_texture_indexing,
            nonuniform_sampler_indexing,
            nonuniform_acceleration_structure_indexing,
            timestamp,
            occlusion,
            pipeline_statistics,
            multiview,
            placed_memory,
            external_memory,
            secondary_commands,
            native_interop,
            memory_observation,
            dynamic_depth_bias,
            dynamic_depth_bias_clamp,
            rect_attachment_clear
        )
    };
}

pub(crate) fn write<B: Backend>(
    device: &Device<B>,
    backend: &str,
    adapter: &str,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    macro_rules! destructure_and_write {
        ($($field:ident),* $(,)?) => {{
            let DeviceCapabilities {
                cooperative_matrix_configurations,
                $($field,)*
                lanes,
                limits: DeviceLimits {
                    max_bind_groups, max_bindings_per_group, max_color_attachments,
                    max_texture_dimension_2d, max_texture_array_layers, max_buffer_bytes,
                    max_vertex_buffers, max_vertex_attributes, max_compute_workgroups,
                    max_compute_workgroup_size, max_push_constant_bytes, max_multiview_views,
                    max_mesh_indirect_count, uniform_offset_alignment, storage_offset_alignment,
                },
            } = device.capabilities().clone();
            let capability_bools = [$(format!("{}:{}", json_string(stringify!($field)), $field)),*]
                .join(",");
            let limits = format!(
                concat!(
                    "{{\"max_bind_groups\":{},\"max_bindings_per_group\":{},",
                    "\"max_color_attachments\":{},\"max_texture_dimension_2d\":{},",
                    "\"max_texture_array_layers\":{},\"max_buffer_bytes\":{},",
                    "\"max_vertex_buffers\":{},\"max_vertex_attributes\":{},",
                    "\"max_compute_workgroups\":[{},{},{}],\"max_compute_workgroup_size\":[{},{},{}],",
                    "\"max_push_constant_bytes\":{},\"max_multiview_views\":{},",
                    "\"max_mesh_indirect_count\":{},\"uniform_offset_alignment\":{},",
                    "\"storage_offset_alignment\":{}}}"
                ),
                max_bind_groups, max_bindings_per_group, max_color_attachments,
                max_texture_dimension_2d, max_texture_array_layers, max_buffer_bytes,
                max_vertex_buffers, max_vertex_attributes,
                max_compute_workgroups[0], max_compute_workgroups[1], max_compute_workgroups[2],
                max_compute_workgroup_size[0], max_compute_workgroup_size[1], max_compute_workgroup_size[2],
                max_push_constant_bytes, max_multiview_views, max_mesh_indirect_count,
                uniform_offset_alignment, storage_offset_alignment,
            );
            let device_capabilities = format!(
                "{{{capability_bools},\"cooperative_matrix_configurations\":{},\"lanes\":{},\"limits\":{limits}}}",
                cooperative_matrices(&cooperative_matrix_configurations), lanes_json(&lanes),
            );
            let root_limits = format!(
                "{{\"maxBufferBytes\":{},\"maxVertexBuffers\":{},\"maxVertexAttributes\":{},\"maxColorAttachments\":{}}}",
                max_buffer_bytes, max_vertex_buffers, max_vertex_attributes, max_color_attachments,
            );
            fs::write(output.join("capabilities.json"), format!(
                concat!(
                    "{{\n  \"schemaVersion\": 2,\n  \"backend\": {},\n  \"adapter\": {},\n",
                    "  \"deviceId\": {},\n  \"compute\": {},\n  \"storageBuffers\": {},\n",
                    "  \"storageTextures\": {},\n  \"nativeInterop\": {},\n  \"debugMarkers\": {},\n",
                    "  \"multiQueue\": {},\n  \"lanes\": {},\n  \"limits\": {},\n",
                    "  \"deviceCapabilities\": {}\n}}\n"
                ),
                json_string(backend), json_string(adapter), json_string(&format!("{:?}", device.id())),
                device.capabilities().compute, device.capabilities().storage_buffers,
                device.capabilities().storage_textures, device.capabilities().native_interop,
                device.capabilities().debug_markers, device.capabilities().multi_queue,
                lanes.len(), root_limits, device_capabilities,
            ))?;
            Ok(())
        }};
    }
    boolean_capability_fields!(destructure_and_write)
}

fn cooperative_matrices(configurations: &[CooperativeMatrixConfiguration]) -> String {
    format!("[{}]", configurations.iter().map(|config| {
        let CooperativeMatrixConfiguration {
            m, n, k, a_type, b_type, c_type, result_type, saturating_accumulation, scope,
        } = config;
        format!(
        concat!("{{\"m\":{},\"n\":{},\"k\":{},\"a_type\":{},\"b_type\":{},",
                "\"c_type\":{},\"result_type\":{},\"saturating_accumulation\":{},\"scope\":{}}}"),
        m, n, k, json_string(component_name(*a_type)), json_string(component_name(*b_type)),
        json_string(component_name(*c_type)), json_string(component_name(*result_type)),
        saturating_accumulation, json_string(scope_name(*scope)),
    )}).collect::<Vec<_>>().join(","))
}

fn lanes_json(lanes: &[LaneInfo]) -> String {
    format!(
        "[{}]",
        lanes
            .iter()
            .map(|lane| {
                let LaneInfo { id, class, routes } = lane;
                format!(
                    "{{\"id\":{},\"class\":{},\"routes\":[{}]}}",
                    lane_id_json(*id),
                    json_string(lane_class_name(*class)),
                    routes
                        .iter()
                        .map(|(target, route)| format!(
                            "{{\"target\":{},\"route\":{}}}",
                            lane_id_json(*target),
                            json_string(route_name(*route))
                        ))
                        .collect::<Vec<_>>()
                        .join(","),
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn lane_id_json(id: LaneId) -> String {
    let LaneId { device, ordinal } = id;
    format!(
        "{{\"device\":{},\"ordinal\":{ordinal}}}",
        json_string(&format!("{device:?}"))
    )
}

fn component_name(value: CooperativeMatrixComponentType) -> &'static str {
    match value {
        CooperativeMatrixComponentType::Float16 => "float16",
        CooperativeMatrixComponentType::Float32 => "float32",
        CooperativeMatrixComponentType::Float64 => "float64",
        CooperativeMatrixComponentType::Sint8 => "sint8",
        CooperativeMatrixComponentType::Sint16 => "sint16",
        CooperativeMatrixComponentType::Sint32 => "sint32",
        CooperativeMatrixComponentType::Sint64 => "sint64",
        CooperativeMatrixComponentType::Uint8 => "uint8",
        CooperativeMatrixComponentType::Uint16 => "uint16",
        CooperativeMatrixComponentType::Uint32 => "uint32",
        CooperativeMatrixComponentType::Uint64 => "uint64",
        CooperativeMatrixComponentType::BFloat16 => "bfloat16",
    }
}

fn scope_name(value: CooperativeMatrixScope) -> &'static str {
    match value {
        CooperativeMatrixScope::Subgroup => "subgroup",
        CooperativeMatrixScope::Workgroup => "workgroup",
        CooperativeMatrixScope::Device => "device",
        CooperativeMatrixScope::QueueFamily => "queue_family",
    }
}

fn lane_class_name(value: LaneClass) -> &'static str {
    match value {
        LaneClass::Graphics => "graphics",
        LaneClass::Compute => "compute",
        LaneClass::Transfer => "transfer",
    }
}

fn route_name(value: WaitRoute) -> &'static str {
    match value {
        WaitRoute::SameQueue => "same_queue",
        WaitRoute::NativeGpuWait => "native_gpu_wait",
        WaitRoute::OrderedByNativeQueue => "ordered_by_native_queue",
        WaitRoute::Unsupported => "unsupported",
    }
}
