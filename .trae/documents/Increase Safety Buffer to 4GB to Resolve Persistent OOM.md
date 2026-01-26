# Analysis of Recurring "!" Output Issue

## 1. Problem Description

* **Scenario**: Using `Aliased Init` mode with `Qwen3-8B`.

* **Status**: The OOM issue is resolved.

* **Regression**: The "!" output issue (repeated exclamation marks) has **reappeared** after `activate_model`.

* **Symptoms**: The model outputs token ID 0 ("!") repeatedly, indicating that the weights on the GPU are likely **all zeros** (or corrupted such that argmax is 0).

## 2. Root Cause Analysis

We previously identified that "!" output means the memory copy from Host to Device failed silently.
We attempted to fix this by:

1. Using `cuMemcpy` (Driver API) instead of `cudaMemcpy` (Runtime API) in `csrc/cumem_allocator.cpp`.
2. Adding `torch.cuda.synchronize()` in `vmm_allocator.py` before the copy.

**Why did it fail again?**

### Hypothesis 1: `cuMemcpy` behavior with Unified Addressing

`cuMemcpy(dst, src, size)` takes `CUdeviceptr`.
Both `dst` (GPU Memory) and `src` (Host Pinned Memory mapped to GPU VA) are `CUdeviceptr`s.
However, `src` is special. It maps to Host physical memory.
If `cuMemcpy` treats it as a strict Device-to-Device copy, it *should* work via PCIe peer logic (or internal copy engine).
But if the driver context is subtle (e.g. `src` was mapped in a way that `cuMemcpy` doesn't like for D2D), it might fail or copy zeros.

### Hypothesis 2: `torch.cuda.synchronize()` is not enough

We synchronized the *current* stream.
But the `Aliased Init` loading happened via `torch.load` -> `tensor.copy_()`.
These operations might have used the default stream.
`torch.cuda.synchronize()` synchronizes the default stream. This should be fine.

### Hypothesis 3: The `src` pointer is invalid at copy time

In `vmm_allocator.py`, we iterate `aliased_pointers`.
For each pointer:

1. `create_physical` (New Handle)
2. `reserve_address` (New VA `temp_va`)
3. `map_memory` (New Handle -> `temp_va`)
4. `copy_memory(temp_va, d_mem, size)`
5. ...

The `d_mem` pointer is the original VA.
Is it possible that `d_mem`'s mapping is somehow invalidated or not accessible?
We checked `set_access`.

### Critical Insight: `cuMemcpy` vs `cudaMemcpy` Return Values

We switched to `cuMemcpy` and check `CUDA_CHECK(result)`.
If it returned an error, we would see `cuMemcpy failed: ...` in the logs (implied, if we propagate it).
The logs show normal request handling (until output generation), implying no Python exception was raised during `activate_model`.
So `cuMemcpy` returned `CUDA_SUCCESS`.

**If** **`cuMemcpy`** **returns Success but data is zero:**
This usually happens if:

1. `src` actually contains zeros.
2. The copy size is 0. (We verified size).
3. We are copying to the wrong place.

**Wait, let's look at** **`csrc/cumem_allocator.cpp`** **change again.**
I changed:

```cpp
CUDA_CHECK(cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size));
```

`cuMemcpy` is synchronous.

**What if** **`src`** **(Host Memory) was never populated correctly?**
"Before `activate_model`: The model generates correct text".
So `src` HAS data.

**What if** **`src`** **is being accessed by the GPU while we try to copy from it?**
The model is idle.

**Let's reconsider the API choice.**
Maybe `cuMemcpyDtoD` is strictly required?
`cuMemcpy` (v2) acts as `cuMemcpyDtoD` if both are device pointers.
Documentation says `cuMemcpy` handles all paths.

**Is it possible** **`temp_va`** **is not what we think it is?**
`reserve_address` returns a VA.
`map_memory` maps physical memory to it.
`set_access` enables access.

**What if** **`set_access`** **takes time?**
It's synchronous.

**Maybe the issue is related to the** **`Context`?**
vLLM uses a specific CUDA context.
If `cuMemCreate` happened in one context (Driver API) but `torch` is using another?
`torch` initializes the Primary Context.
`cumem_allocator.cpp` uses `cuDevicePrimaryCtxRetain`.
They should be the same.

**However**, `torch` manages its own streams.
If `src` was written by `torch` (using Runtime API), and we read it using `cuMemcpy` (Driver API) without proper synchronization *between APIs*?
`torch.cuda.synchronize()` calls `cudaDeviceSynchronize()`.
This syncs the device. It should be enough.

**Wait, what if** **`cuMemcpy`** **is failing to read HOST-backed Device Pointers?**
Standard `cudaMemcpy` handles `Host->Device` specially.
When we use `cuMemcpy` with two `CUdeviceptr`s, the driver thinks it's `Device->Device`.
But the source is physically on Host.
The GPU's Copy Engine (CE) reads from PCIe.
This *should* work.

**BUT**, maybe `cudaMemcpy` was better?
The previous issue was "!" output with `cudaMemcpy`.
The fix was `cuMemcpy`.
Did I *actually* fix it before?
The user said: "`activate_model`之后，发送请求，返回全部是'!', 这个问题**又**出现了".
"Again".
This implies it might have worked once (or they thought it did), or the OOM fix somehow broke it?
Wait, Turn 12 applied the OOM fix (Buffer increase).
Did that break the copy?
No, buffer size affects `num_blocks`. It doesn't touch weight copy logic.

**Maybe the "Resolved" status in previous turn was premature?**
The user said "OOM的问题没有解决，“!”的问题解决了" in Turn 10.
So the "!" fix (Driver API) DID work.
Then in Turn 12 we fixed OOM.
Now in Turn 14, "!" is back.

**Why would OOM fix bring back "!"?**
The OOM fix involved:

1. Increasing `redundancy_buffer_memory`.
2. Adding logs.

This changes the *timing* and *memory layout* (slightly, by reducing KV cache).
Could reducing KV cache size affect Weight Copy?
Unlikely.

**Is it possible that** **`activate_model`** **logic in** **`vmm_allocator.py`** **is unstable?**
The loop:

```python
        for ptr in list(self.aliased_pointers):
            # ...
            copy_memory(temp_va, d_mem, size)
            # ...
```

If we have many tensors (148 tensors), and we do this sequentially.

**Hypothesis:** **`cuMemcpy`** **on Host-Backed memory is flaky?**
Alternative: Use `torch` to copy?
We have pointers `d_mem` (int) and `temp_va` (int).
We can wrap them in `torch.Tensor` (using `as_strided` or `from_blob`? No, `from_blob` needs pointer).
If we create `torch.Tensor` wrappers around `temp_va` and `d_mem`, we can use `dst.copy_(src)`.
This uses PyTorch's internal copy (which uses `cudaMemcpyAsync`).

**Let's try to implement** **`copy_memory`** **using PyTorch in** **`vmm_allocator.py`?**
We need to construct a tensor from a raw pointer.
`vllm` usually avoids this to keep it clean, but we have `csrc`.

**Better approach: Revert to** **`cudaMemcpy`** **but specify** **`cudaMemcpyDeviceToDevice`?**
Since both are `CUdeviceptr` (UVA), `cudaMemcpyDefault` should have worked.
The fact that it didn't suggests something weird about the pointers.

**Let's try** **`cudaMemcpy`** **with** **`cudaMemcpyDeviceToDevice`.**
Even though one is Host-backed, it is mapped into the Device VA space. To the runtime, it is a pointer in the Unified space.
If we tell `cudaMemcpy` "Trust me, it's Device-to-Device", maybe it forces the correct path?
But `cuMemcpy` (Driver API) is essentially `DeviceToDevice` for pointers.

**What if the issue is Memory Visibility?**
`cuMemSetAccess`.
In `create_and_map_host`, we set access for `device`.
In `activate_memory`, we set access for `temp_va` for `device`.
Is it possible `src` access was lost? No.

**What if** **`activate_memory`** **loop is too fast?**
We added `torch.cuda.synchronize()` *before* the loop.
Maybe we need it *inside* the loop? Or *after* each copy?
`cuMemcpy` is sync.

**Let's look at the OOM log again.**
Wait, the OOM log in Turn 13 shows `activate_model` **failed**.
`ERROR ... Invocation of activate_model method failed ... CUDA out of memory`.
If `activate_model` FAILED halfway through, the weights were not fully copied/swapped!
The model is in a **broken state**.
Some weights might be on GPU (copied), some still on Host (or unmapped/lost?).
Or if `restore_full_kv_cache` failed (which happens *after* weight copy), the weights *were* copied.
But `restore_full_kv_cache` failing with OOM means the process crashed/aborted the request.
The *next* request comes in.
The model is in "Active" state (the engine thinks so?), but KV cache is broken?
Or maybe the Weights are fine, but KV cache is broken?
If KV cache is broken (OOM'd), inference might run but produce garbage?
**NO.** If OOM happens, `activate_model` raises exception.
The API server returns 500.
The *user* then sends a request: `curl ...`.
The system is in a zombie state.
The previous `activate_model` crashed.
So the "!" output is because the model is **broken** due to the OOM crash.

**Wait, the user says "OOM的问题没有解决，“!”的问题解决了" in Turn 10.**
Then I fixed OOM in Turn 12.
Then user says "activate\_model之后... 这个问题又出现了" in Turn 14.
Does the user imply OOM is *gone* but "!" is back?
Or is OOM *still there* and causing "!"?
The log snippet provided in Turn 14 (Lines 202-231) shows `curl` returning "!".
It does **NOT** show the OOM error log immediately preceding it.
However, the *previous* log block (Lines 200-308) showed OOM.

**Crucial Question**: Did the OOM happen *before* the "!" request in the user's latest test?
If yes, then "!" is just a symptom of the crash. Fix OOM -> Fix "!".
If no (OOM is fixed, clean activation), but "!" persists, then the Copy is broken.

Let's assume the OOM fix (Buffer=1GB) **worked** (or at least prevented the crash).
If `activate_model` succeeded (no OOM), but output is "!", then we have a copy issue.
Why would the copy issue return if I didn't revert the code?
I didn't revert the code. The `cuMemcpy` change is still there.

**Is it possible** **`cuMemcpy`** **is NOT working for Qwen3-8B specifically?**
No, memory is memory.

**Let's look at** **`vmm_allocator.py`** **logic again.**

```python
        # 4. Copy data: Host (d_mem) -> Device (temp_va)
        copy_memory(temp_va, d_mem, size)
        
        # 5. Unmap old Host Memory
        unmap_memory(d_mem, size)
        
        # 6. Map new Device Memory to original VA (d_mem)
        map_memory(d_mem, size, 0, new_handle, 0)
```

We copy to `temp_va`.
Then we remap `d_mem` to point to the *same physical memory* as `temp_va`.
Wait. `temp_va` maps to `new_handle`.
`d_mem` remaps to `new_handle`.
This is correct.

**Is it possible** **`cuMemcpy`** **failed silently?**
We added `CUDA_CHECK`.

**What if the** **`Buffer=1GB`** **fix logic in** **`gpu_worker.py`** **caused** **`initialize_from_config`** **to fail?**
No, that's just a number calculation.

**Let's suspect the "Double Fix" Interaction.**
We have `cuMemcpy` (sync).
We have `torch.cuda.synchronize()`.
This *should* be robust.

**Maybe the issue is** **`Aliased Init`** **+** **`Qwen3`** **specific?**
Qwen3 might have some tensors that are not weights? Buffers?
`Aliased Init` loads "weights".
If Qwen3 has buffers that are initialized to zero and need to be computed?
But it worked before `activate_model`.

**Back to the OOM trace in Turn 13.**
`v1/worker/gpu_worker.py:482, in restore_full_kv_cache`.
This is *after* `Memory activation complete` (Line 208).
So `vmm_allocator.activate_memory` **FINISHED**.
Weights are copied.
Then `restore_full_kv_cache` runs.
Then OOM happens.
So the system crashes *after* weights are copied but *during* KV cache allocation.
If the user then sends a request:
The weights are on GPU (copied).
The KV cache is... partially allocated? Or failed?
If the engine didn't crash completely (just the request failed), maybe it tries to run inference?
But `activate_model` failed. The engine state might be inconsistent.
However, `vLLM` engine usually catches exceptions.
If the user sends a request, and `activate_model` failed, the model is likely in a weird state (Weights on GPU, but KV cache missing/broken).
If KV cache is missing, inference should fail with "KV cache not initialized".
But "!" output suggests inference *ran* but produced garbage.
This implies `model_runner.execute_model` ran.
If weights are on GPU, why "!"?
Maybe the OOM caused the *weights* to be corrupted? Or the KV cache pointer to be garbage?

**Conclusion**: The "!" is a side effect of the **OOM** (or the broken state following it).
**The OOM is NOT fixed.**
The user said "OOM的问题没有解决" in Turn 10.
I applied a fix (Buffer 1GB) in Turn 12.
The logs in Turn 13 (provided by the user *after* I applied the fix?) show OOM.
Wait, did the user provide the logs in Turn 13 *after* running my fix from Turn 12?
Yes, the user provided the logs *in response* to my fix.
And the log timestamp is `01-28 19:11:39`.
The log shows `torch.OutOfMemoryError`.
And crucially, the log shows:
`Aliased Init: Restoring full KV cache allocation...`
`torch.OutOfMemoryError: ... Tried to allocate 1.15 GiB.`

**My "Buffer 1GB" fix failed.**
Why?
I added logging: `Aliased Init Memory Debug: ...`.
**Check the logs in Turn 13.**
I DO NOT see the `Aliased Init Memory Debug` log line!
I see `INFO ... Aliased Init: Restoring full KV cache allocation...`.
I do **NOT** see the `Aliased Init Memory Debug` log in the user's snippet.
This implies **my changes from Turn 12 were NOT effective or NOT picked up**?
Or maybe the user pasted an old log?
Or maybe I edited the wrong file? `d:/Code/AI/vllm/vllm/v1/worker/gpu_worker.py`.
I verified the edit in Turn 12.

**Hypothesis**: The user is running a test where `enable_aliased_init` is True.
The log `(EngineCore_DP0 pid=458233) INFO 01-28 19:11:39 [v1/worker/gpu_worker.py:475] Aliased Init: Restoring full KV cache allocation...` confirms the code is running.
Line 475 matches where `restore_full_kv_cache` is.
My log should be in `determine_available_memory`.
`determine_available_memory` runs at **Startup** (when the worker initializes).
The log snippet provided by the user is from **Activate** time (19:11:38).
The startup logs are not shown.
So I can't confirm if the buffer was 1GB.

**However, the OOM persists.**
`Tried to allocate 1.15 GiB`.
`Free: 991.50 MiB`.
If I reserved 1GB buffer, `available_kv_cache` should have been reduced by 1GB.
`num_blocks` should have been reduced.
The total allocated should be \~1GB *less* than the limit.
The limit is 42.6 GB.
Usage is 46.36 GB.
This is 3.7 GB *over* the limit.
**My 1GB buffer is insufficient to cover a 3.7 GB discrepancy.**

**Why is the discrepancy so huge (3.7 GB)?**
Usage (46.36) - Limit (42.6) = 3.76 GB.
This is almost exactly the size of... what?
KV Cache for 1 block? No.
Weights? 15GB.
Peak? 1.4GB.
Maybe **CUDAGraph**?
`cuda_graph_memory_bytes` is subtracted.
Maybe **Fragmentation**?
Maybe **Torch Reserved but Unallocated**? 67 MB (small).

**What if** **`non_torch_memory`** **is WRONG?**
I calculated `non_torch = Driver - Weights`.
If `Driver` is 4GB? Unlikely.

**Let's look at** **`available_kv_cache`** **calculation again.**
`available = Requested - Non_KV`.
`Requested = Total * 0.9`.
`Non_KV = Driver + Peak + Weights`.
`Available = Total*0.9 - Driver - Peak - Weights`.

If we allocate `Available` for KV Cache.
`Total_Usage = Available + Driver + Peak + Weights`
`Total_Usage = Total*0.9`.

The math is solid.
**So** **`Non_KV`** **MUST be underestimated.**
We are missing \~3.7 GB of overhead.
What uses 3.7 GB?

* **Context Overhead**? (Usually 500MB).

* **NCCL Buffers**? (Can be large).

* **Workspace**?

* **Model Parameters** being duplicated?

  * `Aliased Init`: Weights are on Host.

  * `Activate`: Weights copied to Device.

  * Are we **double accounting**?

  * What if the "Host-Mapped" tensors (which PyTorch sees) and the "Device-Physical" tensors (which we map) both exist?

  * `unmap_memory(d_mem)` unmaps the Host backing.

  * `map_memory(d_mem, new_handle)` maps the Device backing.

  * The `torch.Tensor` object wraps `d_mem` pointer.

  * It points to the same VA.

  * So PyTorch sees the same tensor.

  * **There is no duplication.**

**Wait!**
`profile_result` is captured using `memory_profiling`.
It measures `torch.cuda.memory_allocated()`.
In `Aliased Init`, `torch.cuda.memory_allocated()` reports the size of weights (because they are CUDA tensors).
Even though they are Host-backed.
So `model_memory_usage` is correct.

**What if** **`requested_memory`** **is calculated wrong?**
`requested = total * 0.9`.
If `init_snapshot.total_memory` is wrong? Unlikely.

**Let's look at** **`redundancy_buffer_memory`.**
If I increase it to **4GB** (`4096 * (1 << 20)`), it should fix it.
It's a brute-force fix, but if we have a mystery 3.7GB overhead, we must respect it.
3.7GB is a lot.
Maybe `gpu_memory_utilization` should be lower?
If I set buffer to 4GB, it effectively reduces utilization by \~8%.
0.9 -> 0.82.

**Why 3.7GB?**
Qwen3-8B. 15GB weights.
Maybe the `Peak Activation` is much higher during `activate_model` (copying?) than during `profile_run`?
`profile_run` runs a dummy forward pass.
`activate_model` does `cuMemcpy`.
`cuMemcpy` shouldn't use 3.7GB device memory.

**Could it be** **`NCCL`?**
`init_worker_distributed_environment` runs before profiling.
Snapshot takes it into account.

**Could it be** **`Deferred KV Cache`** **object itself?**
`self._deferred_kv_cache_config`.
It's a python object. CPU memory.

**I suspect** **`profile_result.non_torch_increase`** **calculation.**
`non_torch = total_used - torch_reserved`.
In `Aliased Init`:
`torch_reserved` = 15GB.
`total_used` = 0.5GB (Driver).
`non_torch` = 0.5 - 15 = -14.5GB.
We add 15GB -> 0.5GB.
This seems correct.

**Is it possible** **`torch_reserved`** **does NOT include Host-Mapped tensors?**
If `torch` treats them as "CUDA" but doesn't count them in `reserved_bytes` because they are not from the caching allocator?
Host-Mapped memory comes from `cuMemCreate` (Driver).
PyTorch's Caching Allocator (`cudaMalloc`) is bypassed?
**YES!**
`VMMAllocator` uses `cuMemCreate`.
It does **NOT** use `torch.cuda.CachingAllocator`.
So `torch.cuda.memory_allocated()` and `memory_reserved()` might **NOT** reflect the weight memory!
If `torch` doesn't count weights:
`torch_reserved` = 0 (approx).
`total_used` = 0.5 (Driver).
`non_torch` = 0.5.
`profile_result.non_kv` = 0.5 + Peak.
We add `Weights` (15) -> 15.5 + Peak.
`Available` = 42.6 - 15.5 = 27.1 GB.
`Total` = 27.1 + 15 (Weights) + Peak + Driver = 42.6 GB.
This *should* work.

**BUT**, if `torch.cuda.memory_allocated()` *does* see them (because we wrap them in `torch.Tensor`?), but `reserved` doesn't?
`VMMAllocator` bypasses the allocator.
So `torch.cuda.memory_stats` might be misleading.

**Let's check** **`mem_utils.py`** **`memory_profiling`** **context.**
It uses `MemorySnapshot`.
`torch.cuda.mem_get_info()` (Driver API) -> `free`, `total`.
`total_used = total - free`.
In `Aliased Init`:
Weights on Host. `free` is high. `total_used` is low (0.5).
So `snapshot` sees low usage.

**So** **`non_torch`** **calculation:**
`non_torch = (total - free) - torch_reserved`.
If `torch_reserved` is 0 (since we bypassed allocator).
`non_torch` = 0.5 - 0 = 0.5.
So `profile_result` correctly sees 0.5GB overhead.

**Where do we add Weights?**
`profile_result.non_kv_cache_memory += weights`.
This accounts for the fact that weights *will* appear.
So `Available = Limit - (Overhead + Peak + Weights)`.
This logic holds.

**So where is the leak?**
Maybe `Peak` is underestimated?
`Peak` comes from `torch.cuda.max_memory_allocated()`.
During `profile_run` (dummy forward), we allocate activations using PyTorch Allocator.
So `Peak` is measured correctly.

**What if** **`weights`** **are NOT 15GB?**
`model_memory_usage` comes from vLLM config calculation.
Maybe it's wrong?
Qwen3-8B (FP16) -> 16GB. (BF16) -> 16GB.
Log says 15.27 GB. Seems reasonable.

**What if** **`num_blocks`** **calculation is aggressive?**
`min_memory_per_block`.
Maybe `block_size` \* `num_layers` \* ... is smaller than actual allocation?
(e.g. fragmentation in the block allocator).
vLLM allocates `CacheEngine`.
`CacheEngine` allocates huge tensors.
`torch.zeros` -> contiguous.

**I am running out of theories, except "4GB Mystery Overhead".**
I will implement the **4GB Buffer Fix**.
It's the only way to guarantee stability given the empirical evidence (3.7GB overflow).
Also, I will re-verify the "!" fix. The "!" reappearance is almost certainly due to the OOM crashing the model state. Fixing OOM is the priority.

## Plan

1. **Modify** **`gpu_worker.py`**: Increase `redundancy_buffer_memory` to **4GB** (4096 MiB) for Aliased Init.
2. **Modify** **`docs`**: Update the "New Issues" section to reflect the recurrence and the stronger fix.

This "brute force" buffer is necessary because we are operating at the very edge of VRAM with a complex VMM mechanism where accounting might be slightly off (e.g. context overheads, fragmentation, or driver-internal reservations for VMM). 4GB is \~8% of 48GB GPU, which is a reasonable safety margin for this "Hot Swap" feature.
