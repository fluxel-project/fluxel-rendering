# RHI Backend Implementation Strategy

Unreal: E:\moyy\program\UnrealEngine-ue6-main
rafx: https://github.com/aclysma/rafx

你正在实现 **Fluxel RHI 0.16 backend**。

E:\moyy\program\rust\fluxel\fluxel-rendering\documents\design-rhi.md 是 v13：**Fluxel RHI public semantic contract / source of truth**

当前目标 backend：

```text
DX12
Vulkan
Metal
WebGPU
GL (GL4.x / GLES 3.x / WebGL2) 的状态机模型与优化边界以当前 `src/backend/gl` 和对应设计文档为准。
```

你的任务不是重新设计 RHI public API，而是：

> 将 Fluxel v13 定义的 portable semantic，正确、完整、高效地 lower 到各 backend。

---

## 1. 总原则

### 1.1 RHI 对象所有权与共享边界

RHI public API 与其默认实现是低层、即时的对象语义；跨帧缓存、永久渲染对象复用和资源调度属于上层资源管理器与渲染图的职责，不能反向污染 RHI 对象模型。

遵循以下规则：

```text
能不用 Arc / Rc / 引用计数器，就不用。

默认由对象直接握住其全部所有权；只有无法用唯一所有权和借用表达时，
才引入共享所有权容器。

不要为同一对象的 backend、capabilities、serials、interning 等内部状态
分别包一层 Arc；应优先收敛到同一个 inner ownership domain。
```

若公共契约明确要求低成本克隆且各副本必须共享同一 native 生命周期，例如 `Device`
或可克隆资源句柄，可以使用一个 `Arc<...Inner>`；inner 必须直接持有 identity、backend
backing 和其余对象状态。若对象不要求克隆或共享，则直接持有 `Box<dyn Backend>` 或具体
backend 值，不要仅为统一形态套 `Arc`。任何一个逻辑对象都不得拆成数个独立引用计数
所有者。

`Arc` / `Rc` / 自定义计数器仅可用于由语义证明的共享生命周期，例如可克隆 GPU 句柄、
异步状态、不可变大块数据共享，或 accepted GPU work 必须跨任务保活的 backing。能够转移
所有权、借用或直接嵌入的对象一律不用计数所有权。不要为了预防性缓存、形态统一、上层
跨帧资源管理或方便传递而给低层 RHI 对象增加引用计数。

不得将 browser session/token、asset token、或等价的引用计数架构带入公共对象模型。
资源归属始终是 Device/Context identity + generation；浏览器和 native backend 的具体
session、context、native object 都是 backend-private 实现细节。

有效的公共资源句柄必须始终拥有 native backing。随着实现完成，应消除
`Option<Arc<dyn ...>>` 这类半初始化形态；创建成功的 Buffer、Texture、View、Sampler、
Pipeline 等不得以“稍后才有 backing”为常规状态。

Transient lifetime、accepted work 保活与 retirement 必须统一归 submission/device 的执行
域管理；不要以 `Arc<Mutex<Vec<TransientLifetime>>>` 等独立 registry 发展第二套生命周期
系统。

### 1.2 公共接口以测试反审契约

每一个新增或语义变更的 public API（包括 format、capability vocabulary、descriptor、
command、future 与错误终态）必须同时提交三类测试：

```text
正面：合法调用得到承诺的结果。
反面：不支持、错误设备、非法组合或错误状态结构化失败，绝不猜测/fallback。
边界：上限、对齐、空/满、精确末端、生命周期或并发终态不漂移。
```

测试不是实现后的装饰。先用这三类可观察行为写出候选契约，再 review 测试本身：若测试
需要 backend 名称、native handle、隐式全局状态、时序猜测或无法说明的 mock 才能表达，
说明 public abstraction 不合理，应先修接口/契约而不是把 backend 细节泄漏出去。每项
测试必须证明 portable semantic；backend-specific conformance 测试另行补充，不能替代它。

实现任何 backend 功能时，统一遵循：

```text
Fluxel v13 semantic
        ↓
理解该功能的 portable contract
        ↓
研究该 backend 的成熟实现
        ↓
查官方 API / specification
        ↓
设计 Fluxel backend-private lowering
        ↓
实现 correctness-complete 最小版本
        ↓
测试 / validation
        ↓
逐步加入 backend-specific optimization
```

禁止反过来：

```text
找到某个现有项目
        ↓
照着它实现
        ↓
为了适配它而修改 Fluxel public API
```

只有能够给出明确的：

```text
DX12 / Vulkan / Metal / WebGPU / GL
底层 API semantic 反例
```

才能建议修改 frozen RHI contract。

---

# 2. 参考资料不是“模板”，而是证据

不要选择一个项目作为唯一真理。

参考实现分成四类：

```text
A. 可读架构参考
B. 当前维护的 correctness 参考
C. 大型生产级实现参考
D. 官方 specification / documentation
```

它们承担不同职责。

---

# 3. 通用参考优先级

## 3.1 rafx — 可读的多 backend RHI 参考

优先研究：

```text
https://github.com/aclysma/rafx
```

尤其：

```text
rafx-api
DX12
Vulkan
Metal
GLES
```

rafx 很适合回答：

```text
一个 RHI backend 应该如何拆模块？

Device / Queue / CommandBuffer 怎么组织？

Buffer / Texture / View 生命周期怎么表达？

backend object ownership 怎么处理？

descriptor / binding / pipeline 怎么 lower？

多 backend 如何共享 semantic，又不共享 native implementation？
```

它的优势：

```text
代码规模适中
Rust
多 backend
抽象层次接近 Fluxel
不被 WebGPU semantic 完全约束
容易阅读
```

但 rafx 已停止活跃演进。

所以：

> **主要学习 architecture / mechanics，不把它当当前 correctness 的唯一依据。**

---

## 3.2 活跃实现 — correctness reference

针对不同 backend，寻找仍活跃维护的实现进行交叉验证。

主要包括：

```text
wgpu-hal
Mesa
ANGLE
MoltenVK
Chromium / Dawn
Firefox / WebGPU implementation
平台官方 sample
```

用途：

```text
确认当前 API 使用方式
发现旧代码已经过时的地方
发现 validation requirement
发现 driver workaround
发现现代 binding / synchronization / presentation 做法
```

但必须区分：

```text
底层 API requirement
```

和：

```text
那个项目自己的 abstraction choice
```

---

## 3.3 Unreal Engine — production-hardening reference

UE 不作为逐行实现模板。

主要用来检查：

```text
大型游戏长期运行后会遇到什么问题？
```

重点研究：

```text
descriptor lifetime
deferred destruction
resource retirement
transient allocator
memory aliasing
upload allocator
multi-queue
residency
pipeline cache
device lost
GPU crash / diagnostics
presentation lifecycle
statistics / debugging
```

尤其对：

```text
DX12
Vulkan
Metal
```

生产级实现非常有参考价值。

不要复制：

```text
UE 历史兼容包袱
全局 engine state
宏体系
平台 wrapper
与 Fluxel 无关的 object model
```

---

# 4. 官方 API 是最终裁判

任何时候参考实现出现冲突：

```text
rafx 这么做
wgpu-hal 那么做
UE 又是另一种做法
```

不要投票。

回到对应官方 contract。

---

## DX12

最终依据：

```text
Microsoft Direct3D 12 / DXGI documentation
D3D12 specification
D3D12 debug layer behavior
```

---

## Vulkan

最终依据：

```text
Khronos Vulkan Specification
Vulkan Validation Layers
Vulkan Guide / official extensions
```

---

## Metal

最终依据：

```text
Apple Metal documentation
Metal Programming Guide
Metal API reference
Metal validation
```

---

## WebGPU

最终依据：

```text
W3C WebGPU Specification
GPUWeb specification
browser implementation constraints
```

不能把：

```text
wgpu native extension
```

误认为 WebGPU browser contract。

---

## OpenGL / WebGL2

最终依据：

```text
Khronos OpenGL specification
OpenGL ES specification
WebGL 2 specification
extension specification
```

特别区分：

```text
Desktop GL
OpenGL ES
WebGL2
```

不要因为 GL 可以，就认为 WebGL2 也可以。

---

# 5. 每个 Backend 的参考策略

## DX12

推荐：

```text
rafx DX12
    -> architecture / readable mechanics

wgpu-hal DX12
    -> current Rust correctness

Unreal D3D12RHI
    -> production hardening

Microsoft D3D12
    -> final authority
```

---

## Vulkan

推荐：

```text
rafx Vulkan
    -> architecture / readable mechanics

wgpu-hal Vulkan
    -> current Rust implementation

ash ecosystem / Mesa / validation layers
    -> Vulkan-specific correctness

Unreal VulkanRHI
    -> production hardening

Khronos Vulkan Spec
    -> final authority
```

Vulkan 特别关注：

```text
queue family
memory type
image layout
pipeline barrier
ownership transfer
descriptor lifetime
timeline/binary synchronization
swapchain recreation
device loss
aliasing
```

---

## Metal

推荐：

```text
rafx Metal
    -> readable Rust implementation

wgpu-hal Metal
    -> current implementation behavior

Unreal MetalRHI
    -> production hardening

Apple Metal docs
    -> final authority
```

重点关注：

```text
MTLDevice
MTLCommandQueue
MTLCommandBuffer
render/compute/blit encoder lifetime
MTLHeap
resource storage mode
argument buffers
drawable lifecycle
completion handlers
```

---

## WebGPU

主要参考：

```text
wgpu-core / wgpu-hal
Dawn
browser WebGPU implementations
GPUWeb specification
```

这里 rafx 价值较低。

必须以：

```text
WebGPU specification
```

为 capability 上限判断依据。

不要因为 Vulkan/DX12 能实现，就偷偷：

```text
shader fallback
CPU fallback
隐藏额外 pass
```

来假装 WebGPU 支持。

不支持就：

```text
Capability = Unsupported
```

---

## OpenGL

参考：

```text
rafx GLES/OpenGL backend
ANGLE
Mesa
wgpu-hal GL（如果对应功能仍适用）
Khronos specification
```

重点注意：

```text
GL 是 state machine
```

而 Fluxel 是：

```text
logical explicit RHI
```

所以 backend 需要维护自己的：

```text
state cache
binding cache
framebuffer state
pipeline-emulation state
```

不能让 GL 全局状态污染 public RHI semantic。

GL 状态机与 GL 调用封装严格采用
`E:\moyy\program\rust\参考\webgl2_performance` 的模型和优化思路；代码应按
当前 v13 identity/generation、结构化错误与 retirement 规则重写，不要求复制旧代码。
每个 context 只有一个实际接线的状态权威，不能保留一套未被 lowering 使用的旁路状态机。
pipeline 必须先做完整对象身份快判，再分别判断 program、raster、depth、stencil、blend
等不可变状态块；bind group 使用 dirty mask 并在 draw/dispatch 前集中逐槽更新；vertex/index
buffer 形成 geometry key 并复用 VAO；active texture、每个 texture/sampler/UBO/SSBO/image
槽、framebuffer、viewport/scissor、clear 与 pixel-store 状态均由该权威追踪。资源退休必须反向
失效绑定槽和依赖它的 VAO/FBO；任何绕过封装的 raw GL 调用必须声明并执行精确 invalidation，
无法证明范围时 invalidation ALL。context loss/replacement 后所有已知状态变为 Unknown。
稳定对象 ID、generation 或 canonical structural key 可以替代旧实现的指针比较；不得为了模仿
指针快判额外引入可有可无的 Arc/Rc。

---

## WebGL2

不要简单写成：

```text
OpenGL backend + wasm
```

必须单独考虑：

```text
browser ownership
WebGL validation
extension availability
default framebuffer
context loss
resource restrictions
thread restrictions
lack of compute/storage features
```

参考：

```text
WebGL2 specification
ANGLE
browser implementations
成熟 WebGL engine
```

---

# 6. 实现一个功能时的标准流程

假设当前实现：

```text
Texture creation
```

任何 backend 都执行同一流程。

---

## Step 1 — 提取 Fluxel semantic

先读 v13。

整理：

```text
输入 descriptor
合法性
Capability requirement
DeviceIdentity
resource lifetime
usage
ResourceUse
statistics
tooling
error semantic
async / sync semantic
```

形成 checklist。

**先不要写代码。**

---

## Step 2 — 看 rafx 如何组织

如果 rafx 有对应 backend，先看它。

目标不是复制代码，而是理解：

```text
需要哪些 native object？
ownership 怎么组织？
谁负责 destroy？
descriptor/view 在什么时候创建？
command 使用时需要什么状态？
```

---

## Step 3 — 找当前维护实现交叉验证

例如：

```text
DX12   -> wgpu-hal
Vulkan -> wgpu-hal
Metal  -> wgpu-hal
WebGPU -> Dawn / wgpu
GL     -> Mesa / ANGLE / wgpu-hal
```

检查：

```text
rafx 有没有过时？

现代 API 调用是否变化？

有没有新的 validation requirement？

有没有特殊 alignment / flag / synchronization 要求？
```

---

## Step 4 — 看生产级实现

如果功能涉及：

```text
memory
lifetime
descriptor
queue
sync
transient
present
device loss
```

必须再查 UE 或其它大型 engine。

问：

```text
这个功能连续运行几个小时以后会出什么问题？
```

---

## Step 5 — 查官方规范

遇到任何不确定：

```text
不要猜
不要按多数实现投票
```

查官方 API/spec。

---

# 7. 设计 Fluxel backend-private lowering

public：

```text
Buffer
Texture
TextureView
Sampler
BindGroup
Pipeline
RecordedWork
```

backend 内部可以完全不同。

例如：

```rust
Dx12Buffer
VulkanBuffer
MetalBuffer
WebGpuBuffer
GlBuffer
```

禁止 native object 泄漏到 portable API。

例如 DX12 public API 不出现：

```text
ID3D12Resource
D3D12_RESOURCE_STATES
D3D12_CPU_DESCRIPTOR_HANDLE
D3D12_GPU_DESCRIPTOR_HANDLE
D3D12_HEAP
Fence value
```

Vulkan public API 不出现：

```text
VkImage
VkBuffer
VkImageLayout
VkPipelineStageFlags
VkAccessFlags
VkSemaphore
VkFence
queue family index
```

Metal public API 不出现：

```text
MTLTexture
MTLBuffer
MTLCommandBuffer
MTLHeap
```

---

# 8. 先实现 correctness，再实现 optimization

这是 backend 实现的重要原则。

例如 Transient。

第一版：

```text
TransientAllocationSupport::Dedicated
```

即：

```text
每个 transient resource
    -> 独立 backing allocation
```

只保证：

```text
semantic 正确
lifetime 正确
completion-safe destruction
```

以后升级：

### DX12

```text
Heap
Placed Resource
Aliasing Barrier
```

### Vulkan

```text
shared VkDeviceMemory
memory requirements
aliasing
barrier / ownership
```

### Metal

```text
MTLHeap
aliasable resource
```

public API 不变。

---

# 9. Async 策略

v13 的规则：

> **真正可能等待未来事件 → async。**

> **纯 CPU/local work → sync。**

例如：

```text
request_device()              async

shader/pipeline compilation   async

submit()                      async
wait_completion()             async
wait_idle()                   async

readback.read()               async

configure presentation        async
acquire                       async
wait_present                  async
```

而：

```text
create_buffer()               sync
create_texture()              sync
create_texture_view()         sync

create_bind_group()           sync

create_recorder()             sync

set_pipeline()                sync
set_bind_group()              sync

draw()                        sync
dispatch()                    sync
copy()                        sync

capability query              sync
```

不要为了“异步风格统一”机械增加 Future。

---

# 10. Backend async implementation

backend 不得要求普通 caller：

```rust
while pending {
    device.poll();
}
```

才能正常完成 Future。

根据平台使用：

```text
Fence/Event
completion callback
Promise
Waker
OS event
worker
host event loop integration
```

驱动 Future。

`Device::poll()` 只是：

```text
opportunistic progress hook
```

不是唯一 progress engine。

---

# 11. Capability 实现原则

绝不能：

```rust
if backend == Vulkan {
    supports_x = true;
}
```

应该根据实际 instance/device：

```text
adapter facts
enabled device features
limits
formats
routes
surface facts
```

产生 Capability。

例如：

```text
Vulkan 某 GPU 支持
    !=
所有 Vulkan Device 都支持

OpenGL extension 支持
    !=
WebGL2 支持
```

---

# 12. 不允许偷偷 fallback

如果 Fluxel API 请求：

```text
Direct Blit
Storage Texture
Compute
Binding Array
Multi Queue dependency
```

而 backend 不支持：

```text
Unsupported
```

禁止 RHI backend 偷偷：

```text
插 shader pass
插 CPU copy
插额外 render pass
CPU emulate compute
```

高级 fallback 应由更高层显式选择另一条 RHI route。

---

# 13. Device Identity / Lifetime

所有 backend 必须实现统一规则：

```text
resource DeviceIdentity
    ==
consumer DeviceIdentity
```

否则：

```text
WrongDevice
```

检查发生在 native backend 调用之前。

Device loss 后：

```text
当前 Device execution domain terminal
pending completion -> DeviceLost
pending readback -> DeviceLost
presentation -> DeviceLost
```

不能继续偷偷恢复旧 object。

重新创建设备：

```text
new DeviceIdentity
```

---

# 14. Resource destruction

任何 backend 都必须遵守：

```text
Rust handle drop
    !=
native resource immediately destroy
```

必须至少满足：

```text
最后 CPU logical owner gone
+
所有 accepted GPU work terminal
```

然后才：

```text
native destroy / recycle
```

这对：

```text
DX12
Vulkan
Metal
WebGPU
GL
```

都统一成立。

---

# 15. Recording

CommandRecorder 是 portable CPU recording builder。

backend 不应该让 public Recorder 直接变成：

```text
ID3D12GraphicsCommandList
VkCommandBuffer
MTLCommandEncoder
WebGPU GPUCommandEncoder
GL immediate calls
```

可以：

```text
record portable/private command representation
```

然后 backend lowering。

这样才能统一支持：

```text
actual ResourceUse
statistics
capture
validation
deferred native realization
```

---

# 16. Submission

SubmissionPlan 是：

```text
logical execution plan
```

不是 native queue submission struct。

backend 负责 lower：

### DX12

```text
CommandQueue
Fence
Wait/Signal
```

### Vulkan

```text
VkQueue
Semaphore
SubmitInfo
```

### Metal

```text
MTLCommandQueue
MTLCommandBuffer
completion handlers
```

### WebGPU

```text
GPUQueue
submitted work completion
```

### GL/WebGL2

```text
single logical ordered lane
```

---

# 17. Backend capability 可以不同

Fluxel 不追求：

```text
所有 backend 功能一样
```

目标是：

```text
统一 semantic vocabulary
+
准确 capability facts
```

所以完全允许：

```text
DX12       feature X = Supported
Vulkan     feature X = Supported
Metal      feature X = Supported
WebGPU     feature X = Unsupported
WebGL2     feature X = Unsupported
```

这不是 abstraction failure。

这是正确的 portable RHI。

---

# 18. 每完成一个 subsystem 必须输出

实现一个 subsystem 后，不要只提交代码。

必须输出：

```text
1. Fluxel v13 semantic 摘要

2. 当前 backend 官方 API 对应物

3. rafx 中参考了什么

4. 当前维护实现中校验了什么

5. UE / 生产级实现发现了哪些风险

6. 官方 specification 最终确认

7. Fluxel backend-private lowering 设计

8. 哪些地方与参考实现不同，以及为什么

9. capability mapping

10. implementation

11. validation / tests

12. 已知限制

13. 下一步优化，但不能要求修改 frozen public API
```

---

# 19. DX12 示例

例如当前任务：

```text
实现 DX12 Buffer
```

执行：

```text
读 Fluxel Buffer semantic
        ↓
读 rafx RafxBufferDx12
        ↓
读 wgpu-hal dx12 Buffer/resource implementation
        ↓
检查 Unreal D3D12 resource lifetime /
descriptor / deferred destruction
        ↓
查 Microsoft D3D12 resource contract
        ↓
设计 Dx12Buffer private representation
        ↓
实现 committed-resource correctness path
        ↓
验证 usage / alignment / WrongDevice /
DeviceLost / retirement
```

这里 DX12 只是这套通用流程的一个实例。

Vulkan / Metal / WebGPU / GL 都执行相同方法论。

---

# 20. 最终目标

不要实现：

> “一个像 rafx 的 RHI。”

不要实现：

> “一个像 wgpu 的 RHI。”

不要实现：

> “一个缩小版 Unreal RHI。”

要实现：

> **Fluxel v13 定义的 RHI。**

并且：

```text
吸收 rafx 的清晰架构
+
吸收活跃实现的 correctness
+
吸收大型 engine 的生产经验
+
由各平台官方 specification 做最终裁决
```

所有 backend：

```text
DX12
Vulkan
Metal
WebGPU
OpenGL
WebGL2
```

最终都必须服从同一条规则：

> **Public semantic 属于 Fluxel；native mechanics 属于 backend。**

## Repository language rule

All documentation committed to GitHub must be written in English. This
includes README files, ADRs, version notes, test/example documentation, and
code comments that describe public decisions. Chinese may be used in local
conversation or scratch notes, but it must be translated before commit.
