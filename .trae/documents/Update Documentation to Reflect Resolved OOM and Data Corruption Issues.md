# Analysis of OOM Error in `activate_model` (Post-Fix)

## 1. Problem Description
Despite the previous fix (deducting `model_weights_size` from `available_kv_cache`), the user reports that OOM still occurs during `activate_model`.
Log: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacity of 47.37 GiB of which 991.50 MiB is free.`
This happens in `_allocate_kv_cache_tensors` -> `torch.zeros`.

## 2. Deep Dive Analysis
- **Total Memory**: 47.37 GiB.
- **Used Memory**: 46.36 GiB.
- **Requested**: 1.15 GiB (for a KV cache layer/chunk).
- **Free**: 991 MiB (< 1.15 GiB).
- **Previous Calculation Logic**: `Available = Total * 0.9 - (Peak + Weights + Graph + Buffer)`.
- **Buffer**: 150 MiB (Default).

Why did we calculate that we had enough space, but then ran out?
1.  **Fragmentation**: PyTorch allocator might have fragmented memory such that while *total free* might be X, contiguous free is < X? Unlikely to be the sole cause for 1GB blocks if using `expandable_segments` (though log suggests turning it on).
2.  **Miscalculation**:
    - `non_torch_memory` might be fluctuating.
    - `Peak Activation` during `activate_model` might be higher than during profiling.
    - **Buffer Too Small**: 150 MiB buffer is extremely tight when we are pushing memory usage to ~46/47 GB (98%!).
    - Wait, `gpu_memory_utilization` is usually 0.9 (90%).
    - 47.37 * 0.9 = 42.63 GiB.
    - If we respect 0.9 limit, used memory should be ~42.6 GB.
    - But log says **46.36 GiB in use**.
    - This means we are using **97.8%** of GPU memory.
    - **Why?**
    - **Hypothesis**: `determine_available_memory` calculated available space based on 0.9 utilization limit.
      `Available_KV = (Total * 0.9) - Fixed_Overhead`.
      So `KV + Fixed_Overhead = Total * 0.9`.
      BUT, maybe `Fixed_Overhead` (Weights + Peak) was **underestimated**?
      Or maybe there is a leak?
      
      Wait, if `non_torch` was negative (-15GB) and we added `weights` (15GB), we got `non_torch` ≈ 0.
      But maybe `non_torch` should be *positive* (Driver overhead)?
      Usually `non_torch` is ~300-500MB.
      If our correction logic resulted in `non_torch` being 0, we underestimated overhead by ~500MB.
      
      Also, `redundancy_buffer_memory` (150MB) is subtracted from *limit*.
      
      Let's look at the numbers in the OOM log again.
      `46.36 GiB memory in use`.
      If limit was 0.9 (42.6 GB), why is usage 46.3 GB?
      This implies `KV Cache` allocated was HUGE.
      
      If `KV Cache` size was determined by `Total * 0.9`, it should stick to 42.6 GB total.
      The only way to exceed 42.6 GB is if:
      1. `gpu_memory_utilization` > 0.9? (Default is 0.9).
      2. `available_kv_cache` was calculated WRONG (too big).
      
      Let's check the fix I applied in Turn 7:
      ```python
      profile_result.non_kv_cache_memory += model_weights_size
      ```
      `non_kv_cache_memory` is subtracted from `requested_memory` (Total * 0.9).
      If `non_kv_cache_memory` is too small, `available` is too big.
      
      Original (Buggy): `non_kv` ≈ `Peak`. (Because `non_torch` ≈ `-Weights`).
      Fix: `non_kv` = `Peak + Weights`.
      
      Is `non_torch` accurately `-Weights`?
      `non_torch` = `Total_Used - Torch_Reserved`.
      If `Total_Used` = `Small` (Weights on CPU).
      `Torch_Reserved` = `Weights` (PyTorch thinks it has them).
      `non_torch` = `Small - Weights`.
      Correct.
      
      So `non_kv` (fixed) = `(Small - Weights) + Peak + Weights` = `Small + Peak`.
      `Small` is real driver overhead (~500MB).
      So `non_kv` = `Driver + Peak`.
      
      Wait!
      We want `non_kv` to be `Driver + Peak + Weights`.
      Because when active, we have `Driver + Peak + Weights` occupying memory that KV cannot use.
      
      My fix: `non_kv += Weights`.
      Result: `non_kv` = `(Driver - Weights) + Peak + Weights` = `Driver + Peak`.
      **ERROR FOUND!**
      I added `Weights` *once*.
      This merely canceled out the negative `non_torch`.
      It restored `non_kv` to what it *would be* if weights were on GPU but *ignored* in the subtraction?
      
      Let's trace:
      We want `Available_KV = Limit - (Driver + Peak + Weights)`.
      
      Current `profile_result.non_kv_cache_memory` (from profiler) = `non_torch + peak + weights`.
      Where `non_torch` ≈ `Driver - Weights`.
      So `profile_result` (raw) = `(Driver - Weights) + Peak + Weights` = `Driver + Peak`.
      
      So raw `profile_result` *already* effectively ignores weights in the sum?
      No, `weights` term cancels the `-weights` in `non_torch`.
      So raw `profile_result` says "Non KV usage is Driver + Peak".
      
      If we subtract *that* from Limit:
      `Available = Limit - (Driver + Peak)`.
      Then `Total_Usage = Available + Driver + Peak + Weights` (Since weights come back).
      `Total_Usage = Limit + Weights`.
      **BOOM.** OOM.
      
      We need `Available = Limit - (Driver + Peak + Weights)`.
      So we need `non_kv_cache_memory` to be `Driver + Peak + Weights`.
      
      Currently raw `profile_result` is `Driver + Peak`.
      So we need to add `Weights` to it.
      
      My previous fix:
      ```python
      profile_result.non_kv_cache_memory += model_weights_size
      ```
      This adds `Weights` to `Driver + Peak`.
      Result: `Driver + Peak + Weights`.
      
      **Wait, my logic above seems correct.**
      Let's re-verify `non_torch` calculation.
      `non_torch = total_used - torch_reserved`.
      If `enable_aliased_init`:
      `torch_reserved` includes weights (virtual).
      `total_used` does NOT include weights (physical).
      So `non_torch` is indeed `Driver - Weights`.
      
      `profile_result.non_kv_cache_memory` (in `mem_utils.py`) = `non_torch + peak + weights`.
      = `(Driver - Weights) + Peak + Weights` = `Driver + Peak`.
      
      So raw result is `Driver + Peak`.
      We need `Driver + Peak + Weights`.
      So we add `Weights`.
      
      So my previous fix *should* be correct?
      `Driver + Peak + Weights` -> `Available = Limit - (Driver + Peak + Weights)`.
      `Total = Available + Driver + Peak + Weights = Limit`.
      
      **Why OOM then?**
      Maybe `Limit` (0.9 * Total) is 42.6 GB.
      But Usage is 46.36 GB.
      This means `Available` was calculated to be ~29 GB.
      `Driver + Peak + Weights` ≈ 0.5 + 1.4 + 15.3 = 17.2 GB.
      `Available` = 42.6 - 17.2 = 25.4 GB.
      `Total` = 25.4 + 17.2 = 42.6 GB.
      
      If Usage is 46.36, it means we exceeded Limit by ~4 GB.
      Where did 4 GB come from?
      
      Maybe `non_torch` is NOT `Driver - Weights`?
      Maybe `torch_reserved` does *not* include weights in Aliased Init?
      In Aliased Init, we use `VMMAllocator`.
      Does `torch` know about these weights?
      Yes, they are `torch.Tensor`s.
      But their storage is mapped to Host.
      Does `torch.cuda.memory_reserved()` count Host-Mapped CUDA tensors?
      If yes, then my logic holds.
      If no (Torch doesn't count them as CUDA reserved), then `torch_reserved` is small.
      Then `non_torch` = `Driver - Small` = `Driver`.
      Then raw `profile_result` = `Driver + Peak + Weights`.
      Then we added `Weights` *again* -> `Driver + Peak + 2*Weights`.
      Then `Available` would be tiny. Usage would be low.
      But we have OOM, so Usage is high.
      
      So `torch_reserved` MUST be counting weights.
      
      **What if `peak_activation` is higher?**
      Maybe 4GB higher? Unlikely.
      
      **What if `block_size` alignment or fragmentation?**
      
      **What if `init_snapshot.free_memory` is used?**
      `kv_cache_memory_bytes_to_gpu_limit = self.init_snapshot.free_memory - non_kv ...`
      `init_snapshot.free_memory` is taken at very beginning.
      It should be ~Total.
      
      **Hypothesis**: The `profile_result` comes from `memory_profiling`.
      Inside `memory_profiling`, it runs the model.
      If `enable_aliased_init` is on, the model runs on GPU (accessing Host weights).
      
      **Let's assume there is a discrepancy.**
      The safest way is to **force** a larger buffer.
      And add logging to see exactly what the numbers are.
      
      User request: "Add logs to prove KV cache space is Total * 0.9 - (Peak + Weights) - Weights".
      Wait, "minus weights" twice?
      No, user means "prove it is Limit - (Peak + Weights)".
      
      And "Increase redundancy buffer" is a solid fix for the 4GB discrepancy (or at least 1GB buffer helps).
      
      Actually, 4GB is huge.
      Is it possible `non_torch` calculation is weird?
      
      **Wait, I see `gpu_worker.py`:**
      ```python
      model_weights_size = int(self.model_runner.model_memory_usage)
      ```
      Is `model_memory_usage` correct?
      It's calculated by vLLM based on config.
      
      **Alternative explanation**:
      When we restore KV cache, we allocate `num_blocks`.
      Each block needs `block_size * num_layers * hidden * ...`.
      Is it possible that `num_blocks` * `block_size` is calculated correctly, but overhead per block (pointers?) is missing?
      
      **Let's stick to the Plan**:
      1. Increase `redundancy_buffer_memory` to 1GB (Turn 11 plan).
      2. Add DETAILED logs showing the calculation (Turn 11 plan).
         `Total`, `Weights`, `Peak`, `NonTorch`, `Graph`, `Buffer`.
         This will allow us (and the user) to debug if the math is wrong.
      
      I will execute the plan from Turn 11.

## 3. Plan Execution
1.  **Modify `gpu_worker.py`**:
    - Increase buffer to 1GB for Aliased Init.
    - Add the debug log.

2.  **Update Documentation**:
    - Update `vLLM_Approximate_Hot_Swap_Design.md` to reflect that we tuned the buffer and added logs.

This addresses the user's explicit request for "More logs to prove..." and the implicit need to fix OOM.
