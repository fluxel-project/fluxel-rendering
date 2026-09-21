# Fluxel RHI 0.16

## 本版本完成的功能

0.16 延续 v13 公共 RHI 契约，重点完成了跨平台测试与示例基础设施，而
不是重新打开公共 API 冻结：

- 公共 provider 组合入口：`create_dx12_provider`、
  `create_vulkan_provider`、`create_metal_provider`，以及 wasm WebGPU 入口。
  上层 examples 不再需要访问 crate-private provider 或 native handle。
- 公共 conformance workload 矩阵，覆盖 triangle、indexed triangle、
  uniform/texture/sampler、storage compute、transfer/readback、depth/stencil、
  MSAA/resolve、indirect、occlusion query、multiview、mapping、BC/ETC2/ASTC
  和 presentation resize/recreate 13 个类别，并为每项固定 capability gate
  与 `Pass/Unsupported/Skipped/Failure` 判定规则。
- examples 入口：`triangle`、`textured_cube`、`compute`、`msaa`、
  `indirect`、`query`，以及既有的 `offscreen_triangle`、`headless_compute`、
  `clear_present`、`provider_probe`。
- GL/WGL 公共 fixture 接入；GL/GLES/WebGL2 仍由 host-owned context 适配，
  native context 不进入 RHI 公共对象模型。
- DX12 completion 顺序修复：较早的不可观察 signal 在后续 fence 成功到达时，
  按队列顺序正确发布完成；transfer resource 保留到对应 fence 完成。
- provider probe 的 Metal/WebGPU 分支补齐，wasm 分支使用浏览器 Promise
  调度，不在浏览器线程上阻塞等待。

## 测试结果

以下数字来自本版本工作树的实际运行；`cargo check` 或 no-run 不计为硬件
通过。

| 环境 | 命令/入口 | 结果 |
| --- | --- | --- |
| API-only | `cargo test -p fluxel-rhi --no-default-features --lib` | **597 passed** |
| Windows Vulkan | `cargo test -p fluxel-rhi --no-default-features --features vulkan --lib -- --test-threads=1` | **610 passed, 0 failed** |
| Windows GL/WebGL backend logic | `cargo test -p fluxel-rhi --no-default-features --features webgl2 --lib -- --test-threads=1` | **727 passed, 0 failed** |
| Chrome headed WebGL2 wasm suite | `scripts/browser/wasm_test_headed.py --features webgl2` | **13 browser tests passed**, capture produced; screenshot evidence written under `target/evidence/` |
| Examples | `cargo check -p fluxel-rhi --features examples --examples` | passed for all registered examples |
| DX12 | full run attempted with DX12+Vulkan | initial GPU work passed, then adapter entered `DEVICE_REMOVED/DEVICE_PAUSED` after TDR; subsequent failures are terminal-device cascade and are not a clean DX12 verdict |

### 尚未形成通过证据的环境

- Chrome WebGPU headed suite：本轮尚未得到完整 `test result: ok.` 证据。
- Android GLES 3.x / Vulkan：脚本和 host fixture 已存在，但本版本记录中没有
  一次完整的设备端通过日志。
- macOS Metal：完成静态编译和 API/逻辑检查，Windows 环境无法执行真机 Metal。
- DX12：需要系统恢复/重启后重新运行，当前 TDR 后的设备状态不能代表代码结果。

## 示例结果

所有 10 个 RHI examples 都以公共 API 编译通过。示例采用 host fixture 注入
方式：fixture 可以由 `fluxel-host` 或应用自己的窗口/浏览器生命周期层实现，
但 workload 只接收 `PlatformProvider`、`Device`、资源和 pipeline 等公共类型。

已具备完整公共录制/提交/readback 路径的示例：

- `headless_compute`
- `offscreen_triangle`
- `clear_present`
- `provider_probe`

新增的 `triangle`、`textured_cube`、`compute`、`msaa`、`indirect`、`query`
提供相同 workload 的公共 fixture 入口；其中具体 shader code form、窗口目标
和 capability-specific 资源由各 backend integration fixture 注入。

## 代码改动简述

- `crates/rhi/examples/`：新增六类公共 API 示例和跨平台使用说明。
- `crates/rhi/tests/`：新增 13 项 conformance matrix、GL/WGL fixture，以及
  公共 workload 状态和 capability gate 文档。
- `crates/rhi/src/lib.rs`：公开 backend provider composition functions，
  仍不暴露 DXGI/Vulkan/Metal/WebGL native handle。
- `crates/rhi/src/backend/dx12/command/spine.rs`、`failure.rs`、
  `platform/device.rs`：修复 fence 顺序完成判定和 device-loss 相关说明/回归。
- `crates/rhi/src/backend/gl/native/mod.rs`：接入 WGL host fixture 的私有 owner
  路径。
- 文档只把已证实的硬件结果写入版本记录；未完成能力继续由 capability
  fail-closed，不以模板或 no-run 编译替代实现证据。

## 0.16 的边界

公共 API 仍保持 v13 冻结。indexed raster、uniform+texture/sampler、MSAA、
multiview、压缩纹理以及 WebGPU/Android/Metal 的真机完整 workload 需要各自
的 shader/WSI fixture 和设备日志后，才能在后续版本提升为硬件 conformance
通过项；本版本不把这些未完成证据标成 Pass。
