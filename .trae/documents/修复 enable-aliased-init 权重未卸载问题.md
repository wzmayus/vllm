# 修复 `--enable-aliased-init` 权重未卸载到 CPU 的问题

## 问题分析
经过代码审查，发现在 `vllm/v1/worker/gpu_worker.py` 文件中，模型加载时的内存分配逻辑存在缺陷。

1.  **根本原因**：
    在 `load_model` 方法中，虽然调用了 `VMMAllocator.enable_aliasing()` 设置了 Host Mode 标志位，但是实际的内存分配（`self.model_runner.load_model`）是在 `_maybe_get_memory_pool_context` 上下文管理器中进行的。
    
    目前的 `_maybe_get_memory_pool_context` 实现**只检查了 `enable_sleep_mode`**：
    ```python
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            # ... 使用 CuMemAllocator ...
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()
    ```
    
    当用户仅启用 `--enable-aliased-init` 而未启用 `--enable-sleep-mode` 时，该函数返回 `nullcontext()`。这意味着 PyTorch 使用默认的 CUDA 分配器，而不是我们自定义的 `cumem_allocator`（它负责拦截分配请求并根据 Host Mode 标志位将其重定向到 Host Pinned Memory）。因此，权重被直接加载到了 GPU 显存，导致功能失效。

2.  **验证**：
    代码逻辑明确显示 `enable_aliased_init` 的情况被遗漏了，导致自定义分配器未被挂载。

## 修复方案

修改 `vllm/v1/worker/gpu_worker.py` 中的 `_maybe_get_memory_pool_context` 方法，使其在启用 `enable_aliased_init` 时也能正确返回 `VMMAllocator` 的内存池上下文。

### 修改后的代码逻辑：
```python
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # 优先处理 Aliased Init，因为它包含更高级的内存管理逻辑
        if self.vllm_config.model_config.enable_aliased_init:
             from vllm.device_allocator.vmm_allocator import VMMAllocator
             allocator = VMMAllocator.get_instance()
             if tag == "weights":
                 # 确保分配器状态干净，类似于 sleep mode 的检查
                 assert allocator.get_current_usage() == 0, (
                    "Aliased init mode can only be used for one instance per process."
                 )
             return allocator.use_memory_pool(tag=tag)
        elif self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()
```

## 执行步骤
1.  **编辑文件**：修改 `vllm/v1/worker/gpu_worker.py`。
2.  **验证**：虽然无法直接运行完整的 GPU 测试，但逻辑修复是确定的。
