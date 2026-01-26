# vLLM 近似热切换 (Approximate Hot-Swap) 代码修改设计文档

## 1. 概述
本功能旨在解决单张 GPU 显存不足以同时容纳两个大模型时的快速切换问题。通过利用 CUDA VMM (Virtual Memory Management) 技术，实现模型的“预初始化”：在 CPU (Host) 内存中完成模型加载和 CUDA Graph Capture，仅在需要推理时将数据快速搬运至 GPU 显存并重映射地址，从而实现亚秒级的模型切换。

## 2. 核心架构设计
采用 **"Host Backing + Remapping"** 方案，并结合 **"Sparse VMM (稀疏虚拟内存)"** 技术解决 KV Cache 动态扩容导致的 CUDA Graph 指针失效问题。

1.  **虚拟地址解耦**：利用 CUDA VMM 将 GPU 的虚拟地址 (VA) 与物理显存 (PA) 解耦。
2.  **Sparse VMM Allocation (稀疏分配)**：
    *   对于 KV Cache 等大块显存，在初始化阶段直接**保留 (Reserve)** 完整的虚拟地址空间 (例如 20GB)。
    *   但仅**映射 (Map)** 极小部分的 Host 物理内存 (例如 100MB) 以满足启动时的 Profiling 和 Warmup 需求。
    *   **关键点**：CUDA Graph 捕获的是这个固定的 VA 范围，无论后续物理内存如何变化，VA 保持不变，从而避免 Graph 失效。
3.  **Host Aliasing (预备阶段)**：
    *   模型权重和稀疏 KV Cache 映射到 Host Pinned Memory。
    *   GPU 可通过 PCIe 访问这些数据，完成 Kernel 编译和 Graph Capture。
4.  **Hot Activation (激活阶段)**：
    *   分配真实的 GPU 物理显存 (Full Size)。
    *   将 Host Memory 中的有效数据 (权重 + 稀疏 KV Cache 内容) 拷贝到 GPU 显存。
    *   修改页表映射 (Remap)，将原有的 VA 指向新的 GPU 显存。
    *   释放 Host Memory。
    *   由于 VA 不变，CUDA Graph 继续有效。

## 3. 详细代码修改

### 3.1 C++ 扩展层 (`csrc`)
修改文件：[cumem_allocator.cpp](file:///d:/Code/AI/vllm/csrc/cumem_allocator.cpp)

为了支持 Python 层精细控制内存映射，我们在 C++ 扩展中暴露了底层的 Driver API 原语。

*   **新增 VMM 原语绑定**：
    *   `reserve_address`: 保留虚拟地址范围 (`cuMemAddressReserve`)。
    *   `create_physical`: 创建 GPU 物理内存句柄 (`cuMemCreate`)。
    *   `create_host_physical`: **[新增]** 创建 Host (Pinned) 物理内存句柄 (`cuMemCreate` with `CU_MEM_LOCATION_TYPE_HOST`)。
    *   `map_memory` / `unmap_memory`: 建立/解除 VA 到 PA 的映射 (`cuMemMap`/`cuMemUnmap`)。
    *   `set_access`: 设置内存访问权限 (`cuMemSetAccess`)。
    *   `copy_memory`: **[新增]** 设备内/设备间内存拷贝（`cuMemcpyAsync` + `cuStreamSynchronize`），用于将数据从 Host Backing 拷贝到 Real GPU Memory。
    *   `update_handle`: **[新增]** 更新 C++ 层维护的句柄指针，确保 Python 对象释放时能正确释放底层的 GPU 显存而非旧的 Host 内存。
    *   `python_set_sparse_limit`: **[新增]** 设置稀疏分配限制。
    *   `make_tensor_from_ptr`: **[新增]** 将原始内存指针包装为 `torch.Tensor`，并绑定自定义 Deleter 以正确释放 VMM 内存。

*   **分配策略增强**：
    *   增加了 `g_use_host_mode` 全局标志。
    *   增加了 `g_sparse_limit`。当设置此限制时，`malloc` 会保留完整 VA，但只映射 `min(size, limit)` 的 Host 物理内存。

### 3.2 内存分配器 (`device_allocator`)
新增文件：[vmm_allocator.py](file:///d:/Code/AI/vllm/vllm/device_allocator/vmm_allocator.py)

创建了 `VMMAllocator` 类，继承自 `CuMemAllocator`，作为 Python 侧的核心管理器。

*   **`enable_aliasing()`**: 开启 Host Mapping 模式。
*   **`set_sparse_allocation_limit(bytes)`**: 设置稀疏分配上限。
*   **`allocate_tensor(size, shape, dtype, device)`**: **[新增]** 分配 Host-Backed VMM 内存并返回 PyTorch Tensor。
*   **`activate_memory(device_id)`**:
    *   **核心逻辑**：执行从 CPU 到 GPU 的热迁移，并处理稀疏张量的特殊情况。
    *   **流程**：
        1.  遍历所有记录的 Aliased 指针。
        2.  `create_physical`: 分配等大 (Full Size) 的 GPU 显存。
        3.  `copy_memory`: 将数据从 Host 拷贝到 GPU。如果是稀疏张量，只拷贝已 Backing 的部分 (Sparse Size)。
        4.  `unmap_memory`: 解除 Host 映射 (如果是稀疏张量，只解除 Sparse Size)。
        5.  `map_memory`: 将原有的 VA (Full Size) 重新映射到新的 GPU 显存。
        6.  `release_physical`: 释放 Host 内存句柄。
    *   **修复点**：针对稀疏张量，必须只 `unmap` 实际映射的大小，否则 `cuMemUnmap` 会失败；而 `map` 操作必须映射整个 Full Size。

### 3.3 Worker 集成 (`worker`)
修改文件：[gpu_worker.py](file:///d:/Code/AI/vllm/vllm/v1/worker/gpu_worker.py)

*   **`load_model` 方法**：
    *   **强制 CPU 加载**：临时将 `model_runner.device` 设为 `cpu`，使权重加载到 CPU 内存（避免 GPU 显存占用）。
    *   **权重注入 (Aliasing)**：调用 `alias_model_weights`，将 CPU 权重搬运到 Host-Backed VMM 内存，并替换模型参数。
    *   **结果**：启动时权重占用 GPU 显存为 0 (仅 VA 占用)。

*   **`initialize_from_config` 方法**：
    *   计算启动所需的最小内存 (`min_blocks`)，调用 `set_sparse_allocation_limit`。
    *   **关键修复**：将 `initialize_kv_cache` 包裹在 `VMMAllocator.use_memory_pool("kv_cache")` 中，强制使用 VMM 分配器并遵循稀疏限制。
    *   VMMAllocator 会保留 20GB VA，但只消耗 100MB Host 内存。
    *   CUDA Graph 捕获这 20GB VA 的指针。

*   **`restore_full_kv_cache` 方法**：
    *   现在为空操作 (No-op)。因为 VA 已经是完整的，且 `activate_memory` 已经完成了物理内存的填充。

### 3.4 引擎控制接口 (`engine`)
修改文件：[llm_engine.py](file:///d:/Code/AI/vllm/vllm/v1/engine/llm_engine.py)

*   **`activate_model()` 方法**：
    *   调用 `VMMAllocator.activate_memory()` 完成物理内存切换。
    *   不再需要显式的 `restore` 逻辑。

## 4. Debug Log & 问题分析

### 已解决问题
* --enable-aliased-init启动之后，发送请求，正常返回；activate_model之后，发送请求，也是正常返回 - YES!
* activate_model的时候报错，CUDA out of memory - RESOLVED (Fixed by increasing safety buffer to 4GB)
* 激活后输出全是 "!" (Token ID 0) 或乱码 - RESOLVED (Fixed by Sparse VMM).
* **服务刚启动时显存占用异常升高 (29GB)** - **RESOLVED**.
    * **原因**:
        1. 权重被直接加载到 GPU 显存（默认行为）。
        2. KV Cache 未使用 VMM 分配器，导致全量分配 GPU 显存。
    * **修复**:
        1. `load_model` 强制加载到 CPU，然后迁移至 Host-Backed VMM。
        2. `initialize_kv_cache` 强制使用 VMM 稀疏分配。
    * **预期结果**: 启动时 GPU 显存占用极低 (< 2GB)，激活后恢复正常占用。

## 5. 验证方法
请在服务器环境执行以下步骤：

1.  **编译安装**：
    为了加速编译，已在 `setup.py` 中增加了 `VLLM_BUILD_ONLY` 支持，只编译修改过的 `cumem_allocator`：
    ```bash
    # 只编译 cumem_allocator，跳过其他繁重的扩展
    export VLLM_BUILD_ONLY=cumem_allocator
    pip install -e .
    ```
    *(注意：初次编译仍建议完整运行 `pip install -e .` 以确保依赖完整)*

2.  **运行测试脚本**：
    ```bash
    export VLLM_VMM_ACTIVATE_VERIFY=1
    python -m vllm.entrypoints.openai.api_server --model <model> --enable-aliased-init ...
    ```
3.  **流程验证**：
    *   Start -> Send Request (Should work, slow)
    *   Activate Model -> Send Request (Should work, fast, correct output)
    *   观察日志，确认 `VMM activation copy verify` 通过。

## 6. 日志
启动完成之后，显存占用变大了。
```
 root@ay:/vllm-workspace# nvidia-smi
Tue Feb  3 04:50:59 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L20                     Off |   00000000:00:0C.0 Off |                  Off |
| N/A   75C    P0            124W /  350W |   29030MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            9577      C   VLLM::EngineCore                      28978MiB |
+-----------------------------------------------------------------------------------------+
```
启动日志
```
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:18 [compilation/backends.py:288] Compiling a graph for dynamic shape takes 6.71 s
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:41:18 [compilation/backends.py:768] Computation graph saved to /root/.cache/vllm/torch_compile_cache/b985b23115/rank_0_0/backbone/computation_graph.py
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:19 [compilation/monitor.py:34] torch.compile takes 11.17 s in total
(APIServer pid=9504) DEBUG 02-03 04:41:20 [v1/engine/utils.py:950] Waiting for 1 local, 0 remote core engine proc(s) to start.
(APIServer pid=9504) DEBUG 02-03 04:41:30 [v1/engine/utils.py:950] Waiting for 1 local, 0 remote core engine proc(s) to start.
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:31 [v1/worker/gpu_worker.py:362] Aliased Init: Corrected non-torch memory to 0.03 GiB and reserved 15.27 GiB in non_kv_cache_memory (compensating for CPU-offloaded weights).
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:41:31 [v1/worker/gpu_worker.py:387] Initial free memory: 47.03 GiB; Requested memory: 0.90 (util), 42.64 GiB
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:41:31 [v1/worker/gpu_worker.py:393] Free memory after profiling: 46.93 GiB (total), 42.53 GiB (within requested)
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:41:31 [v1/worker/gpu_worker.py:399] Memory profiling takes 24.71 seconds. Total non KV cache memory: 16.70GiB; torch peak memory increase: 1.40GiB; non-torch forward increase memory: -15.24GiB; weights memory: 15.27GiB.
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:31 [v1/worker/gpu_worker.py:400] Available KV cache memory: 25.94 GiB
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:31 [v1/core/kv_cache_utils.py:1286] GPU KV cache size: 188,864 tokens
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:31 [v1/core/kv_cache_utils.py:1291] Maximum concurrency for 16,384 tokens per request: 11.53x
(EngineCore_DP0 pid=9577) INFO 02-03 04:41:31 [v1/worker/gpu_worker.py:459] Aliased Init: Initializing with sparse KV cache (limit 10485760 bytes per tensor) to save memory for startup tasks while reserving full VA space.


(EngineCore_DP0 pid=9577) INFO 02-03 04:44:01 [v1/worker/gpu_model_runner.py:4477] Graph capturing finished in 150 secs, took 0.54 GiB
(EngineCore_DP0 pid=9577) INFO 02-03 04:44:01 [v1/worker/gpu_worker.py:565] Aliased Init Memory Debug: Total Requested: 42.64 GiB, Weights: 15.27 GiB, Peak Activation: 1.40 GiB, Non-Torch: 0.03 GiB, Graph: 0.54 GiB, Buffer: 4.00 GiB. Formula: Requested - (Weights + Peak + Non-Torch + Graph + Buffer) = 21.40 GiB
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:44:01 [v1/worker/gpu_worker.py:598] Free memory on device (47.03/47.37 GiB) on startup. Desired GPU memory utilization is (0.9, 42.64 GiB). Actual usage is 15.27 GiB for weight, 1.4 GiB for peak activation, 0.03 GiB for non-torch memory, and 0.54 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=22976439705` (21.4 GiB) to fit into requested memory, or `--kv-cache-memory=27692338176` (25.79 GiB) to fully utilize gpu memory. Calculated potential (deferred) KV cache memory is 25.94 GiB.
(EngineCore_DP0 pid=9577) INFO 02-03 04:44:01 [v1/engine/core.py:266] init engine (profile, create kv cache, warmup model) took 175.31 seconds
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:44:02 [utils/gc_utils.py:40] GC Debug Config. enabled:False,top_objects:-1
(APIServer pid=9504) DEBUG 02-03 04:44:02 [v1/engine/utils.py:1063] READY from local core engine process 0.
(EngineCore_DP0 pid=9577) DEBUG 02-03 04:44:02 [v1/engine/core.py:908] EngineCore waiting for work.
(APIServer pid=9504) DEBUG 02-03 04:44:02 [v1/metrics/loggers.py:246] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 11804
```
激活模型
```
(APIServer pid=9504) INFO 02-03 05:03:10 [entrypoints/openai/api_server.py:924] activating model from aliased state
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:10 [v1/engine/core.py:513] Activating model memory from Host Backing...
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:10 [device_allocator/vmm_allocator.py:87] Disabling VMM aliasing mode
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:10 [device_allocator/vmm_allocator.py:97] Activating memory for 148 tensors
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:10 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=12918456320 src_hash=15671208136963838365 dst_hash=15671208136963838365
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:10 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=12918456320 src_hash=15671208136963838365 remap_hash=15671208136963838365
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=134421034303488 src_hash=7782737544244222502 dst_hash=7782737544244222502
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=134421034303488 src_hash=7782737544244222502 remap_hash=7782737544244222502
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14172553216 src_hash=4383011679291150979 dst_hash=4383011679291150979
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14172553216 src_hash=4383011679291150979 remap_hash=4383011679291150979
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=134421084635136 src_hash=3598831830838738215 dst_hash=3598831830838738215
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=134421084635136 src_hash=3598831830838738215 remap_hash=3598831830838738215
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14164164608 src_hash=15786046522820683941 dst_hash=15786046522820683941
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14164164608 src_hash=15786046522820683941 remap_hash=15786046522820683941
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14193524736 src_hash=3321880341113990785 dst_hash=3321880341113990785
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14193524736 src_hash=3321880341113990785 remap_hash=3321880341113990785
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14394851328 src_hash=2266321428864280917 dst_hash=2266321428864280917
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14394851328 src_hash=2266321428864280917 remap_hash=2266321428864280917
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=134421149646848 src_hash=15888445142274642646 dst_hash=15888445142274642646
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=134421149646848 src_hash=15888445142274642646 remap_hash=15888445142274642646
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14495514624 src_hash=9237407865860961693 dst_hash=9237407865860961693
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14495514624 src_hash=9237407865860961693 remap_hash=9237407865860961693
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14529069056 src_hash=4525600126540330485 dst_hash=4525600126540330485
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14529069056 src_hash=4525600126540330485 remap_hash=4525600126540330485
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14730395648 src_hash=13698012329686620573 dst_hash=13698012329686620573
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14730395648 src_hash=13698012329686620573 remap_hash=13698012329686620573
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14831058944 src_hash=2135820478799822033 dst_hash=2135820478799822033
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14831058944 src_hash=2135820478799822033 remap_hash=2135820478799822033
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14898167808 src_hash=11701891251665914512 dst_hash=11701891251665914512
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14898167808 src_hash=11701891251665914512 remap_hash=11701891251665914512
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18119393280 src_hash=12825623409136692204 dst_hash=12825623409136692204
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18119393280 src_hash=12825623409136692204 remap_hash=12825623409136692204
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19730006016 src_hash=12970842062268034340 dst_hash=12970842062268034340
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19730006016 src_hash=12970842062268034340 remap_hash=12970842062268034340
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21340618752 src_hash=5603049626857464905 dst_hash=5603049626857464905
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21340618752 src_hash=5603049626857464905 remap_hash=5603049626857464905
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18354274304 src_hash=2148111576950719007 dst_hash=2148111576950719007
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18354274304 src_hash=2148111576950719007 remap_hash=2148111576950719007
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17314086912 src_hash=9890610602082318498 dst_hash=9890610602082318498
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17314086912 src_hash=9890610602082318498 remap_hash=9890610602082318498
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20568866816 src_hash=17221274877099311840 dst_hash=17221274877099311840
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20568866816 src_hash=17221274877099311840 remap_hash=17221274877099311840
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18756927488 src_hash=10296982418422891130 dst_hash=10296982418422891130
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18756927488 src_hash=10296982418422891130 remap_hash=10296982418422891130
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19662897152 src_hash=588835461139645265 dst_hash=588835461139645265
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19662897152 src_hash=588835461139645265 remap_hash=588835461139645265
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16106127360 src_hash=10296333038232177826 dst_hash=10296333038232177826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16106127360 src_hash=10296333038232177826 remap_hash=10296333038232177826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16508780544 src_hash=776778438972528542 dst_hash=776778438972528542
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16508780544 src_hash=776778438972528542 remap_hash=776778438972528542
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16911433728 src_hash=1686087450145526319 dst_hash=1686087450145526319
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16911433728 src_hash=1686087450145526319 remap_hash=1686087450145526319
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15233712128 src_hash=523533439345228962 dst_hash=523533439345228962
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15233712128 src_hash=523533439345228962 remap_hash=523533439345228962
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15133048832 src_hash=14302995926132166891 dst_hash=14302995926132166891
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15133048832 src_hash=14302995926132166891 remap_hash=14302995926132166891
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19159580672 src_hash=11954703899403145272 dst_hash=11954703899403145272
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19159580672 src_hash=11954703899403145272 remap_hash=11954703899403145272
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=14931722240 src_hash=1450671689192856237 dst_hash=1450671689192856237
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=14931722240 src_hash=1450671689192856237 remap_hash=1450671689192856237
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16542334976 src_hash=8875174252422346778 dst_hash=8875174252422346778
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16542334976 src_hash=8875174252422346778 remap_hash=8875174252422346778
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18152947712 src_hash=10407264452790975439 dst_hash=10407264452790975439
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18152947712 src_hash=10407264452790975439 remap_hash=10407264452790975439
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19763560448 src_hash=2038566270771430178 dst_hash=2038566270771430178
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19763560448 src_hash=2038566270771430178 remap_hash=2038566270771430178
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21374173184 src_hash=18053754864875177200 dst_hash=18053754864875177200
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21374173184 src_hash=18053754864875177200 remap_hash=18053754864875177200
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22984785920 src_hash=17071926989387502209 dst_hash=17071926989387502209
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22984785920 src_hash=17071926989387502209 remap_hash=17071926989387502209
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24595398656 src_hash=3392359710593384989 dst_hash=3392359710593384989
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24595398656 src_hash=3392359710593384989 remap_hash=3392359710593384989
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26206011392 src_hash=10179484149253868552 dst_hash=10179484149253868552
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26206011392 src_hash=10179484149253868552 remap_hash=10179484149253868552
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27816624128 src_hash=14414825278242479787 dst_hash=14414825278242479787
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27816624128 src_hash=14414825278242479787 remap_hash=14414825278242479787
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20770193408 src_hash=1357780694181919887 dst_hash=1357780694181919887
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20770193408 src_hash=1357780694181919887 remap_hash=1357780694181919887
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20870856704 src_hash=337905935774950717 dst_hash=337905935774950717
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20870856704 src_hash=337905935774950717 remap_hash=337905935774950717
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16039018496 src_hash=3128233069363172244 dst_hash=3128233069363172244
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16039018496 src_hash=3128233069363172244 remap_hash=3128233069363172244
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17649631232 src_hash=10583686732010898722 dst_hash=10583686732010898722
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17649631232 src_hash=10583686732010898722 remap_hash=10583686732010898722
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19260243968 src_hash=6980176788817677004 dst_hash=6980176788817677004
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19260243968 src_hash=6980176788817677004 remap_hash=6980176788817677004
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22481469440 src_hash=10413674406368815440 dst_hash=10413674406368815440
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22481469440 src_hash=10413674406368815440 remap_hash=10413674406368815440
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24092082176 src_hash=10236438643042051141 dst_hash=10236438643042051141
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24092082176 src_hash=10236438643042051141 remap_hash=10236438643042051141
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25702694912 src_hash=15785096464550576313 dst_hash=15785096464550576313
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25702694912 src_hash=15785096464550576313 remap_hash=15785096464550576313
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27313307648 src_hash=15744662158354658071 dst_hash=15744662158354658071
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27313307648 src_hash=15744662158354658071 remap_hash=15744662158354658071
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15535702016 src_hash=17732926292685240175 dst_hash=17732926292685240175
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15535702016 src_hash=17732926292685240175 remap_hash=17732926292685240175
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20367540224 src_hash=7245836012195746905 dst_hash=7245836012195746905
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20367540224 src_hash=7245836012195746905 remap_hash=7245836012195746905
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17146314752 src_hash=1750047234865831995 dst_hash=1750047234865831995
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17146314752 src_hash=1750047234865831995 remap_hash=1750047234865831995
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21978152960 src_hash=2612131876199655024 dst_hash=2612131876199655024
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21978152960 src_hash=2612131876199655024 remap_hash=2612131876199655024
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23588765696 src_hash=6361103532953250975 dst_hash=6361103532953250975
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23588765696 src_hash=6361103532953250975 remap_hash=6361103532953250975
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25199378432 src_hash=13318189299189893512 dst_hash=13318189299189893512
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25199378432 src_hash=13318189299189893512 remap_hash=13318189299189893512
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26809991168 src_hash=18435469604009676027 dst_hash=18435469604009676027
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26809991168 src_hash=18435469604009676027 remap_hash=18435469604009676027
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28420603904 src_hash=1711358137021637615 dst_hash=1711358137021637615
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28420603904 src_hash=1711358137021637615 remap_hash=1711358137021637615
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22380806144 src_hash=6628294524556008295 dst_hash=6628294524556008295
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22380806144 src_hash=6628294524556008295 remap_hash=6628294524556008295
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17716740096 src_hash=1942816383091628248 dst_hash=1942816383091628248
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17716740096 src_hash=1942816383091628248 remap_hash=1942816383091628248
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19327352832 src_hash=1721986194209873288 dst_hash=1721986194209873288
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19327352832 src_hash=1721986194209873288 remap_hash=1721986194209873288
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20937965568 src_hash=12559937844078897504 dst_hash=12559937844078897504
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20937965568 src_hash=12559937844078897504 remap_hash=12559937844078897504
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22548578304 src_hash=13978068351486301520 dst_hash=13978068351486301520
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22548578304 src_hash=13978068351486301520 remap_hash=13978068351486301520
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24159191040 src_hash=10724951352572593546 dst_hash=10724951352572593546
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24159191040 src_hash=10724951352572593546 remap_hash=10724951352572593546
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25769803776 src_hash=11540540526164779693 dst_hash=11540540526164779693
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25769803776 src_hash=11540540526164779693 remap_hash=11540540526164779693
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27380416512 src_hash=8539569756703100826 dst_hash=8539569756703100826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27380416512 src_hash=8539569756703100826 remap_hash=8539569756703100826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23991418880 src_hash=3816886355032186834 dst_hash=3816886355032186834
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23991418880 src_hash=3816886355032186834 remap_hash=3816886355032186834
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20971520000 src_hash=14843992497393826718 dst_hash=14843992497393826718
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20971520000 src_hash=14843992497393826718 remap_hash=14843992497393826718
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16139681792 src_hash=5514863305268690017 dst_hash=5514863305268690017
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16139681792 src_hash=5514863305268690017 remap_hash=5514863305268690017
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17750294528 src_hash=18217825145471451010 dst_hash=18217825145471451010
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17750294528 src_hash=18217825145471451010 remap_hash=18217825145471451010
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19360907264 src_hash=15103746361205067019 dst_hash=15103746361205067019
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19360907264 src_hash=15103746361205067019 remap_hash=15103746361205067019
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22582132736 src_hash=8850100431985861467 dst_hash=8850100431985861467
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22582132736 src_hash=8850100431985861467 remap_hash=8850100431985861467
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24192745472 src_hash=824731718187165652 dst_hash=824731718187165652
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24192745472 src_hash=824731718187165652 remap_hash=824731718187165652
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25803358208 src_hash=16102088309805136183 dst_hash=16102088309805136183
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25803358208 src_hash=16102088309805136183 remap_hash=16102088309805136183
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27413970944 src_hash=1913602450545170389 dst_hash=1913602450545170389
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27413970944 src_hash=1913602450545170389 remap_hash=1913602450545170389
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15636365312 src_hash=14453659308165324065 dst_hash=14453659308165324065
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15636365312 src_hash=14453659308165324065 remap_hash=14453659308165324065
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20468203520 src_hash=6189328552785638151 dst_hash=6189328552785638151
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20468203520 src_hash=6189328552785638151 remap_hash=6189328552785638151
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17246978048 src_hash=8108790977061817110 dst_hash=8108790977061817110
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17246978048 src_hash=8108790977061817110 remap_hash=8108790977061817110
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18857590784 src_hash=16912571999926249318 dst_hash=16912571999926249318
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18857590784 src_hash=16912571999926249318 remap_hash=16912571999926249318
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22078816256 src_hash=12832514636088738837 dst_hash=12832514636088738837
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22078816256 src_hash=12832514636088738837 remap_hash=12832514636088738837
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23689428992 src_hash=17450181998077300435 dst_hash=17450181998077300435
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23689428992 src_hash=17450181998077300435 remap_hash=17450181998077300435
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25300041728 src_hash=1540790018704764234 dst_hash=1540790018704764234
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25300041728 src_hash=1540790018704764234 remap_hash=1540790018704764234
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26910654464 src_hash=14590057907721784655 dst_hash=14590057907721784655
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26910654464 src_hash=14590057907721784655 remap_hash=14590057907721784655
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28521267200 src_hash=6082128268745289770 dst_hash=6082128268745289770
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28521267200 src_hash=6082128268745289770 remap_hash=6082128268745289770
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25602031616 src_hash=3899929450899759200 dst_hash=3899929450899759200
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25602031616 src_hash=3899929450899759200 remap_hash=3899929450899759200
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19964887040 src_hash=4055684889731658182 dst_hash=4055684889731658182
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19964887040 src_hash=4055684889731658182 remap_hash=4055684889731658182
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16743661568 src_hash=14439114527868253876 dst_hash=14439114527868253876
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16743661568 src_hash=14439114527868253876 remap_hash=14439114527868253876
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21575499776 src_hash=16846300317388271338 dst_hash=16846300317388271338
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21575499776 src_hash=16846300317388271338 remap_hash=16846300317388271338
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23186112512 src_hash=8084892835540478496 dst_hash=8084892835540478496
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23186112512 src_hash=8084892835540478496 remap_hash=8084892835540478496
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24796725248 src_hash=8429866384775467319 dst_hash=8429866384775467319
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24796725248 src_hash=8429866384775467319 remap_hash=8429866384775467319
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26407337984 src_hash=6723054489582445013 dst_hash=6723054489582445013
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26407337984 src_hash=6723054489582445013 remap_hash=6723054489582445013
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28017950720 src_hash=1954249929898027090 dst_hash=1954249929898027090
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28017950720 src_hash=1954249929898027090 remap_hash=1954249929898027090
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27212644352 src_hash=6859988831770125149 dst_hash=6859988831770125149
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27212644352 src_hash=6859988831770125149 remap_hash=6859988831770125149
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15703474176 src_hash=11476307816336172155 dst_hash=11476307816336172155
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15703474176 src_hash=11476307816336172155 remap_hash=11476307816336172155
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18924699648 src_hash=13142846499346697725 dst_hash=13142846499346697725
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18924699648 src_hash=13142846499346697725 remap_hash=13142846499346697725
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20535312384 src_hash=9798607876826529583 dst_hash=9798607876826529583
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20535312384 src_hash=9798607876826529583 remap_hash=9798607876826529583
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22145925120 src_hash=8763440715628791114 dst_hash=8763440715628791114
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22145925120 src_hash=8763440715628791114 remap_hash=8763440715628791114
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23756537856 src_hash=2824767039952212747 dst_hash=2824767039952212747
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23756537856 src_hash=2824767039952212747 remap_hash=2824767039952212747
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25367150592 src_hash=13291005814139417991 dst_hash=13291005814139417991
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25367150592 src_hash=13291005814139417991 remap_hash=13291005814139417991
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26977763328 src_hash=5826395634608007033 dst_hash=5826395634608007033
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26977763328 src_hash=5826395634608007033 remap_hash=5826395634608007033
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22951231488 src_hash=11324000924947799285 dst_hash=11324000924947799285
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22951231488 src_hash=11324000924947799285 remap_hash=11324000924947799285
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15737028608 src_hash=934091724534631250 dst_hash=934091724534631250
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15737028608 src_hash=934091724534631250 remap_hash=934091724534631250
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17347641344 src_hash=5933753265082404856 dst_hash=5933753265082404856
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17347641344 src_hash=5933753265082404856 remap_hash=5933753265082404856
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18958254080 src_hash=16141703600900582603 dst_hash=16141703600900582603
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18958254080 src_hash=16141703600900582603 remap_hash=16141703600900582603
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22179479552 src_hash=6458180725766354606 dst_hash=6458180725766354606
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22179479552 src_hash=6458180725766354606 remap_hash=6458180725766354606
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23790092288 src_hash=13758514528881018843 dst_hash=13758514528881018843
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23790092288 src_hash=13758514528881018843 remap_hash=13758514528881018843
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=25400705024 src_hash=16488104071272512939 dst_hash=16488104071272512939
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=25400705024 src_hash=16488104071272512939 remap_hash=16488104071272512939
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27011317760 src_hash=3550600666978220656 dst_hash=3550600666978220656
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27011317760 src_hash=3550600666978220656 remap_hash=3550600666978220656
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20065550336 src_hash=1083805424042195607 dst_hash=1083805424042195607
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20065550336 src_hash=1083805424042195607 remap_hash=1083805424042195607
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16844324864 src_hash=17345241621065814510 dst_hash=17345241621065814510
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16844324864 src_hash=17345241621065814510 remap_hash=17345241621065814510
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18454937600 src_hash=14570017188070498298 dst_hash=14570017188070498298
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18454937600 src_hash=14570017188070498298 remap_hash=14570017188070498298
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21676163072 src_hash=6159056969302541011 dst_hash=6159056969302541011
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21676163072 src_hash=6159056969302541011 remap_hash=6159056969302541011
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23286775808 src_hash=5695061298792312478 dst_hash=5695061298792312478
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23286775808 src_hash=5695061298792312478 remap_hash=5695061298792312478
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24897388544 src_hash=13288355675191787719 dst_hash=13288355675191787719
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24897388544 src_hash=13288355675191787719 remap_hash=13288355675191787719
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26508001280 src_hash=883532267841129297 dst_hash=883532267841129297
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26508001280 src_hash=883532267841129297 remap_hash=883532267841129297
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28118614016 src_hash=13760254030569807405 dst_hash=13760254030569807405
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28118614016 src_hash=13760254030569807405 remap_hash=13760254030569807405
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24561844224 src_hash=6774119192254633997 dst_hash=6774119192254633997
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24561844224 src_hash=6774119192254633997 remap_hash=6774119192254633997
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17951621120 src_hash=9300525428119125853 dst_hash=9300525428119125853
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17951621120 src_hash=9300525428119125853 remap_hash=9300525428119125853
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=19562233856 src_hash=14003491518757388685 dst_hash=14003491518757388685
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=19562233856 src_hash=14003491518757388685 remap_hash=14003491518757388685
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16341008384 src_hash=1954073745060476805 dst_hash=1954073745060476805
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16341008384 src_hash=1954073745060476805 remap_hash=1954073745060476805
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21172846592 src_hash=2487958810946652067 dst_hash=2487958810946652067
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21172846592 src_hash=2487958810946652067 remap_hash=2487958810946652067
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22783459328 src_hash=5824558279029848350 dst_hash=5824558279029848350
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:11 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22783459328 src_hash=5824558279029848350 remap_hash=5824558279029848350
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24394072064 src_hash=602047939918040944 dst_hash=602047939918040944
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24394072064 src_hash=602047939918040944 remap_hash=602047939918040944
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26004684800 src_hash=6355220925980045343 dst_hash=6355220925980045343
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26004684800 src_hash=6355220925980045343 remap_hash=6355220925980045343
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27615297536 src_hash=3126593182087483150 dst_hash=3126593182087483150
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27615297536 src_hash=3126593182087483150 remap_hash=3126593182087483150
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26172456960 src_hash=11888175020526600681 dst_hash=11888175020526600681
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26172456960 src_hash=11888175020526600681 remap_hash=11888175020526600681
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15300820992 src_hash=1732199600093756687 dst_hash=1732199600093756687
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15300820992 src_hash=1732199600093756687 remap_hash=1732199600093756687
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18522046464 src_hash=186401526167109163 dst_hash=186401526167109163
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18522046464 src_hash=186401526167109163 remap_hash=186401526167109163
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20132659200 src_hash=7709649987560496874 dst_hash=7709649987560496874
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20132659200 src_hash=7709649987560496874 remap_hash=7709649987560496874
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21743271936 src_hash=4478713947157202722 dst_hash=4478713947157202722
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21743271936 src_hash=4478713947157202722 remap_hash=4478713947157202722
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23353884672 src_hash=14139062074562132102 dst_hash=14139062074562132102
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23353884672 src_hash=14139062074562132102 remap_hash=14139062074562132102
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24964497408 src_hash=12118584880576086144 dst_hash=12118584880576086144
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24964497408 src_hash=12118584880576086144 remap_hash=12118584880576086144
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26575110144 src_hash=4281476683342223825 dst_hash=4281476683342223825
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26575110144 src_hash=4281476683342223825 remap_hash=4281476683342223825
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28185722880 src_hash=16145261536608853803 dst_hash=16145261536608853803
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28185722880 src_hash=16145261536608853803 remap_hash=16145261536608853803
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15334375424 src_hash=10290744761432342220 dst_hash=10290744761432342220
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15334375424 src_hash=10290744761432342220 remap_hash=10290744761432342220
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=20166213632 src_hash=9588495247478492508 dst_hash=9588495247478492508
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=20166213632 src_hash=9588495247478492508 remap_hash=9588495247478492508
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16944988160 src_hash=12717068512679055594 dst_hash=12717068512679055594
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16944988160 src_hash=12717068512679055594 remap_hash=12717068512679055594
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18555600896 src_hash=9500982213457792391 dst_hash=9500982213457792391
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18555600896 src_hash=9500982213457792391 remap_hash=9500982213457792391
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21776826368 src_hash=4125304726290974390 dst_hash=4125304726290974390
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21776826368 src_hash=4125304726290974390 remap_hash=4125304726290974390
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=23387439104 src_hash=15634187710954214004 dst_hash=15634187710954214004
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=23387439104 src_hash=15634187710954214004 remap_hash=15634187710954214004
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24998051840 src_hash=13437920169281068764 dst_hash=13437920169281068764
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24998051840 src_hash=13437920169281068764 remap_hash=13437920169281068764
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26608664576 src_hash=9710865373601010619 dst_hash=9710865373601010619
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26608664576 src_hash=9710865373601010619 remap_hash=9710865373601010619
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=28219277312 src_hash=10244375870815165887 dst_hash=10244375870815165887
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=28219277312 src_hash=10244375870815165887 remap_hash=10244375870815165887
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27783069696 src_hash=11967924853766049826 dst_hash=11967924853766049826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27783069696 src_hash=11967924853766049826 remap_hash=11967924853766049826
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=16441671680 src_hash=4304939370921595592 dst_hash=4304939370921595592
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=16441671680 src_hash=4304939370921595592 remap_hash=4304939370921595592
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=18052284416 src_hash=15393897787926713938 dst_hash=15393897787926713938
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=18052284416 src_hash=15393897787926713938 remap_hash=15393897787926713938
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=21273509888 src_hash=643748989552917946 dst_hash=643748989552917946
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=21273509888 src_hash=643748989552917946 remap_hash=643748989552917946
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=22884122624 src_hash=9554802461931473847 dst_hash=9554802461931473847
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=22884122624 src_hash=9554802461931473847 remap_hash=9554802461931473847
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=24494735360 src_hash=9936599067038751539 dst_hash=9936599067038751539
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=24494735360 src_hash=9936599067038751539 remap_hash=9936599067038751539
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=26105348096 src_hash=13711967803088213895 dst_hash=13711967803088213895
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=26105348096 src_hash=13711967803088213895 remap_hash=13711967803088213895
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=27715960832 src_hash=7939262300067498453 dst_hash=7939262300067498453
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=27715960832 src_hash=7939262300067498453 remap_hash=7939262300067498453
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=17548967936 src_hash=17911837230613794017 dst_hash=17911837230613794017
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=17548967936 src_hash=17911837230613794017 remap_hash=17911837230613794017
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:145] VMM activation copy verify ptr=15938355200 src_hash=4012541684419105737 dst_hash=4012541684419105737
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:173] VMM activation remap verify ptr=15938355200 src_hash=4012541684419105737 remap_hash=4012541684419105737
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:197] Memory activation complete
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:87] Disabling VMM aliasing mode
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:97] Activating memory for 0 tensors
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [device_allocator/vmm_allocator.py:197] Memory activation complete
(EngineCore_DP0 pid=9577) INFO 02-03 05:03:12 [v1/worker/gpu_worker.py:496] Aliased Init: KV cache activation handled by VMM remapping.
```
请求能正常返回，没有!号墙
