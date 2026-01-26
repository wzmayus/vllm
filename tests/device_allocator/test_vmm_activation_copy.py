# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.device_allocator.cumem import cumem_available
from vllm.device_allocator.vmm_allocator import VMMAllocator

try:
    from vllm.cumem_allocator import sample_hash
except ModuleNotFoundError:
    sample_hash = None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not cumem_available or sample_hash is None,
    reason="CUDA/cumem_allocator is not available.",
)
def test_vmm_activate_memory_preserves_contents():
    allocator = VMMAllocator.get_instance()
    allocator.enable_aliasing()
    tensor = None
    device_id = torch.cuda.current_device()
    try:
        with allocator.use_memory_pool(tag="weights"):
            tensor = torch.empty((1024 * 1024,), device="cuda", dtype=torch.uint8)
            tensor.random_(0, 256)
            torch.cuda.synchronize()
            before = int(sample_hash(tensor.data_ptr(), 256))

        allocator.disable_aliasing()
        allocator.activate_memory(device_id=device_id)
        torch.cuda.synchronize()
        after = int(sample_hash(tensor.data_ptr(), 256))
        assert before == after
    finally:
        allocator.disable_aliasing()
        del tensor
        torch.cuda.empty_cache()

