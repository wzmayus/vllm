# Analysis of 1GB Buffer Impact

## 1. User Question
"Increase buffer to 1GB for Aliased Init. 会导致aliased启动时占用的显存增大850M吗"
(Will increasing the buffer to 1GB for Aliased Init cause the GPU memory usage at startup to increase by 850MB?)

## 2. Analysis
The `redundancy_buffer_memory` in `determine_available_memory` is used to **subtract** from the available memory limit when calculating `available_kv_cache_memory_bytes`.

Formula:
`available_kv_cache = (Total * 0.9) - (Peak + Weights + Graph + Overhead) - Buffer`

- **Startup Phase (Aliased Init)**:
  - We calculate `available_kv_cache` using the *increased* buffer (1GB instead of 150MB).
  - This results in a **smaller** `available_kv_cache` value (smaller by ~850MB).
  - We log this value as "Calculated potential (deferred) KV cache memory".
  - **However**, at startup, we do NOT allocate this full amount.
  - In `initialize_from_config`, we override `num_blocks` to `min_blocks` (160 blocks, ~0.35 GB).
  - So, the **actual allocated memory at startup** is determined by `min_blocks`, which is tiny and independent of the buffer size.
  - Therefore, increasing the buffer does **NOT** increase memory usage at startup.

- **Activation Phase**:
  - We restore the full KV cache configuration.
  - The `num_blocks` for the full cache was derived from the `available_kv_cache` calculated at startup.
  - Since `available_kv_cache` is now **smaller** (by 850MB), the restored `num_blocks` will be fewer.
  - This means the **total memory allocated after activation** will be **850MB LESS** than before.
  - This is exactly what we want: to leave more safety margin (empty space) on the GPU to prevent OOM.

## 3. Conclusion
- **Startup Usage**: No change. (Controlled by `min_blocks`).
- **Activation Usage**: **Decreases** by ~850MB. (Controlled by `num_blocks` which is reduced by the larger buffer).
- **Answer**: No, it will not increase startup memory usage. Instead, it will **reduce** the maximum memory usage after activation, preventing the OOM error.

## 4. Plan
1.  **Explain this logic to the user**.
2.  **Proceed with the code change** (increasing buffer and adding logs) as planned.

The user's concern ("increase usage") is the opposite of the actual effect ("decrease capacity/limit"). I will clarify this.
