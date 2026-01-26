## 现象复盘（来自文档“新问题”日志）
- `/activate_model` 请求返回成功（没有 500 / OOM Traceback）。
- 但激活后同样的 chat 请求开始输出连续的 `"!"`（token 0 墙）。

在 Qwen3 系列里，token id 0 常对应 `!`，连续输出 token 0 往往意味着：
- logits 退化成“全 0/全相等”（argmax 变成 0）；
- 或者模型的关键权重（尤其 embedding / lm_head）被拷贝成 0（或拷贝未完成导致读到未初始化/清零的显存）。

由于本次“4GB buffer”后仍出现 `!`，并且日志里没有出现 OOM 堆栈（activate_model 成功），更像是 **权重拷贝路径仍存在不可靠/未同步** 的问题，而不是 OOM 把状态打坏。

## 可疑点列表（按优先级）

### 1) `copy_memory` 使用 `cuMemcpy` 可能不是同步语义，导致“先 unmap 再拷贝完成”
- 当前激活流程（[vmm_allocator.py](file:///d:/Code/AI/vllm/vllm/device_allocator/vmm_allocator.py)）在每个 tensor 上执行：
  1) `copy_memory(temp_va, d_mem, size)`
  2) 立刻 `unmap_memory(temp_va, size)` + `free_address(temp_va, size)`
- 但 `python_copy_memory`（[cumem_allocator.cpp](file:///d:/Code/AI/vllm/csrc/cumem_allocator.cpp#L904-L919)）当前是：
  - `cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, size)`
- 如果 `cuMemcpy` 的实现是“提交到某个 copy stream/引擎后返回”（或对某些 VMM 映射类型非严格同步），那么 `temp_va` 被 unmap/free 后拷贝仍在进行，会产生“拷贝成功返回但内容为 0/未定义”的现象。

### 2) 以 `CUdeviceptr` 表示的“Host physical 映射 VA”作为拷贝 src，可能需要显式 stream 同步
即使 `cuMemcpy` 本身通常同步，但 Host-backed 的 VMM 映射（`CU_MEM_LOCATION_TYPE_HOST`）是比较特殊的路径；在一些驱动版本/平台上，同步性/可见性可能更敏感。

### 3) 缺少“拷贝后校验”，导致 silent corruption 难以定位
目前只要 `cuMemcpy` 没返回错误就继续执行；一旦出现 silent corruption，只能从最终生成 `!` 推断。

## 解决方案（修复 + 诊断增强）

### A. 让 `copy_memory` 变为“强同步”
1. 将 [cumem_allocator.cpp](file:///d:/Code/AI/vllm/csrc/cumem_allocator.cpp#L904-L919) 的 `python_copy_memory` 改为：
   - 使用 `cuMemcpyDtoDAsync`（或 `cuMemcpyAsync`）在显式 `CUstream` 上发起拷贝
   - 紧接着 `cuStreamSynchronize(stream)` 或 `cuCtxSynchronize()`

目标：保证返回到 Python 之前，`temp_va` 相关拷贝已经完成，从而 `unmap/free temp_va` 不会破坏 in-flight copy。

### B. 增加“拷贝后抽样校验”日志（默认关闭，通过 env 开启）
1. 在 C++ 扩展中增加一个只用于 debug 的方法（例如 `peek_bytes(ptr, nbytes)` 或 `checksum(ptr, nbytes)`）：
   - 通过 `cuMemcpyDtoH` 把 `ptr` 指向的前 N 字节拷贝到 host buffer
   - 计算一个小 hash（如 xor / crc32）并返回
2. 在 [vmm_allocator.py](file:///d:/Code/AI/vllm/vllm/device_allocator/vmm_allocator.py) `activate_memory` 中：
   - 在 copy 前后对 `d_mem` 与 `temp_va` 抽样计算 hash，打印：
     - `src_hash_before`, `dst_hash_after`, `equal?`
   - 若发现 `dst_hash_after` 为全 0 或不匹配，立即 `logger.error` 并中止 activation（抛异常），避免返回 200 但模型状态已坏。

### C. 文档修正
更新 [vLLM_Approximate_Hot_Swap_Design.md](file:///d:/Code/AI/vllm/docs/vLLM_Approximate_Hot_Swap_Design.md)：
- 将“4GB buffer 后仍为 !”保留在“新问题”，并记录本轮定位：copy 同步/校验问题。
- 纠正文档中对 `copy_memory` 的描述（目前写的是 `cudaMemcpy`）。

## 验证与测试计划
### 1) 最小化 GPU 回归测试（新增测试脚本/pytest，需 GPU + cumem available）
- 开启 aliasing，分配一个小 tensor（例如 16MB），写入非零 pattern（或随机数）
- 调用 `activate_memory`
- 读取 tensor 内容验证 hash 与激活前一致

### 2) 端到端验证
- Model B `--enable-aliased-init` 启动
- `/activate_model` 返回 200
- 同一 prompt 输出不再出现 `!` 墙
- Debug 模式下确认每个 copy 的 hash 校验通过

---
如果您确认该方案，我将按上述步骤提交代码修改（C++ 扩展 + Python 激活流程 + 文档 + 测试）。