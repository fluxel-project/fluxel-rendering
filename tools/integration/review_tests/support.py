"""Regression tests for the bounded S0-S3 integration evidence reviewer."""
import importlib.util
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "review", Path(__file__).resolve().parents[1] / "review.py"
)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": review.sha256(path) or ""}


CAPABILITY_DECLARATION = """pub struct DeviceCapabilities {
    pub compute: bool,
    pub storage_buffers: bool,
    pub storage_textures: bool,
    pub native_interop: bool,
    pub debug_markers: bool,
    pub multi_queue: bool,
    pub lanes: Arc<[LaneInfo]>,
    pub cooperative_matrix_configurations: Vec<CooperativeMatrixConfiguration>,
    pub limits: DeviceLimits,
}
pub struct DeviceLimits {
    pub max_buffer_bytes: u64,
    pub max_vertex_buffers: u32,
    pub max_vertex_attributes: u32,
    pub max_color_attachments: u32,
    pub max_compute_workgroups: [u32; 3],
}
pub struct CooperativeMatrixConfiguration {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: CooperativeMatrixComponentType,
    pub b_type: CooperativeMatrixComponentType,
    pub c_type: CooperativeMatrixComponentType,
    pub result_type: CooperativeMatrixComponentType,
    pub saturating_accumulation: bool,
    pub scope: CooperativeMatrixScope,
}
pub enum CooperativeMatrixComponentType {
    Float16,
    BFloat16,
}
pub enum CooperativeMatrixScope {
    Subgroup,
    QueueFamily,
}
"""


def capability_fixture(compute: object = True, device_id: str = "DeviceId(1)") -> dict:
    return {"backend": "native-gl-wgl", "adapter": "actual adapter", "deviceId": device_id, "compute": compute,
            "storageBuffers": True, "storageTextures": False, "nativeInterop": True,
            "debugMarkers": True, "multiQueue": False, "lanes": 1,
            "limits": {"maxBufferBytes": 4096, "maxVertexBuffers": 4, "maxVertexAttributes": 8, "maxColorAttachments": 4},
            "deviceCapabilities": {"compute": compute, "storage_buffers": True, "storage_textures": False,
                "native_interop": True, "debug_markers": True, "multi_queue": False,
                "lanes": [{"id": {"device": device_id, "ordinal": 0}, "class": "graphics", "routes": []}],
                "cooperative_matrix_configurations": [{"m": 16, "n": 16, "k": 16, "a_type": "float16", "b_type": "bfloat16",
                    "c_type": "float16", "result_type": "float16", "saturating_accumulation": False, "scope": "queue_family"}],
                "limits": {"max_buffer_bytes": 4096, "max_vertex_buffers": 4, "max_vertex_attributes": 8,
                           "max_color_attachments": 4, "max_compute_workgroups": [64, 64, 64]}}}
