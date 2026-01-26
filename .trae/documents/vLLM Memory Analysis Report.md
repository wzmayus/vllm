# Deferred KV Cache Allocation Plan

To satisfy the requirement of running Model B with minimal VRAM usage while still performing Profiling and CUDA Graph Capture at startup, we will implement a "Minimal KV Cache" strategy. This allows the model to initialize and warm up using a tiny fraction of the KV cache (just enough for the dummy run), and then expand to the full capacity upon activation.

## 1. Modify `vllm/v1/worker/gpu_worker.py`

### `initialize_from_config`
- Detect `enable_aliased_init` flag.
- If enabled:
    - Deepcopy the provided `kv_cache_config` and store it as `self._deferred_kv_cache_config`.
    - Create a modified `kv_cache_config` with `num_gpu_blocks` reduced to the minimum required for startup profiling (calculated based on `max_num_batched_tokens`).
    - Use this minimal config for the immediate initialization.
    - Log the deferral/minimization.

### `restore_full_kv_cache` (New Method)
- Check for `self._deferred_kv_cache_config`.
- If present:
    - Log "Restoring full KV cache".
    - Call `self.model_runner.initialize_kv_cache(self._deferred_kv_cache_config)` to re-allocate the KV cache tensors with the full size.
    - Clear `self._deferred_kv_cache_config`.

## 2. Modify `vllm/v1/engine/core.py`

### `_activate_worker`
- Update this function (which runs on the worker during `activate_model`) to call `worker.restore_full_kv_cache()` after the VMM memory activation is complete.

## Verification
- This ensures Model B consumes only ~2.5GB (Activation + Graph + Minimal KV) at startup instead of 45GB+.
- Model A can run safely with 0.9 utilization.
- Switching speed is preserved because Profiling and Graph Capture are already done; only the KV cache memory allocation (which is fast, just `malloc`) happens at switch time.
