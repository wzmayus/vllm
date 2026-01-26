
import os
import torch
from vllm.device_allocator.cumem import CuMemAllocator, cumem_available, HandleType
from vllm.logger import init_logger

try:
    from vllm.cumem_allocator import (
        create_physical,
        create_host_physical,
        map_memory,
        unmap_memory,
        set_access,
        release_physical,
        set_host_mode,
        set_sparse_limit,
        update_handle,
        reserve_address,
        free_address,
        copy_memory,
        sample_hash,
        make_tensor_from_ptr,
    )
except ImportError:
    create_physical = None
    create_host_physical = None
    map_memory = None
    unmap_memory = None
    set_access = None
    release_physical = None
    set_host_mode = None
    set_sparse_limit = None
    update_handle = None
    reserve_address = None
    free_address = None
    copy_memory = None
    sample_hash = None
    make_tensor_from_ptr = None

logger = init_logger(__name__)

class VMMAllocator(CuMemAllocator):
    instance: "VMMAllocator" = None

    @staticmethod
    def get_instance() -> "VMMAllocator":
        assert cumem_available, "cumem allocator is not available"
        if VMMAllocator.instance is None:
            VMMAllocator.instance = VMMAllocator()
        return VMMAllocator.instance

    def __init__(self):
        super().__init__()
        self.aliased_pointers = set()
        self.aliasing_enabled = False
        self.sparse_limit = 0
        self.sparse_tensors = {}  # ptr -> backed_size

    def set_sparse_allocation_limit(self, limit: int):
        self.sparse_limit = limit
        if set_sparse_limit:
            set_sparse_limit(limit)

    def allocate_tensor(self, size: int, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Allocate a tensor using VMM memory (Host-Backed if aliasing is enabled).
        Returns a torch.Tensor that wraps the raw pointer.
        The tensor will have a custom deleter that calls free() correctly.
        """
        # Allocate raw memory (returns int pointer)
        # Note: self.malloc is inherited from CuMemAllocator.
        # It calls my_malloc -> _python_malloc_callback -> tracks pointer.
        ptr = self.malloc(size)
        
        # Create Tensor wrapper
        if make_tensor_from_ptr:
            return make_tensor_from_ptr(ptr, shape, dtype, device.index)
        else:
            raise RuntimeError("make_tensor_from_ptr not available")

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        super()._python_malloc_callback(allocation_handle)
        if self.aliasing_enabled:
            py_d_mem = allocation_handle[2]
            self.aliased_pointers.add(py_d_mem)
            if self.sparse_limit > 0:
                self.sparse_tensors[py_d_mem] = self.sparse_limit
                logger.debug("Sparse allocation for %s: backed %d bytes", py_d_mem, self.sparse_limit)
            logger.debug("Mapped address %s to Host Memory (Aliasing Mode)", py_d_mem)

    def _python_free_callback(self, ptr: int) -> HandleType:
        if ptr in self.aliased_pointers:
            self.aliased_pointers.remove(ptr)
        if ptr in self.sparse_tensors:
            del self.sparse_tensors[ptr]
        return super()._python_free_callback(ptr)

    def enable_aliasing(self):
        # We call it "Aliasing" but it uses Host Mapping
        logger.info("Enabling VMM aliasing (Host Mapping) mode")
        if set_host_mode:
            set_host_mode(True)
        self.aliasing_enabled = True
        
    def disable_aliasing(self):
        logger.info("Disabling VMM aliasing mode")
        if set_host_mode:
            set_host_mode(False)
        self.aliasing_enabled = False

    def activate_memory(self, device_id: int):
        """
        Replace Host-Backed memory with Real Device memory.
        Copies data from Host to Device.
        """
        logger.info(f"Activating memory for {len(self.aliased_pointers)} tensors")
        
        # Ensure any pending operations on the host-mapped memory are completed
        import torch
        torch.cuda.synchronize()
        
        verify = os.environ.get("VLLM_VMM_ACTIVATE_VERIFY") == "1" and sample_hash is not None
        to_remove = []
        
        # We process each pointer.
        for ptr in list(self.aliased_pointers):
            data = self.pointer_to_data.get(ptr)
            if not data:
                continue
                
            handle_tuple = data.handle
            # handle_tuple: (device, size, d_mem, p_memHandle)
            size = handle_tuple[1]
            d_mem = handle_tuple[2]
            p_memHandle_addr = handle_tuple[3]
            
            try:
                # 1. Allocate new physical memory (Device)
                new_handle = create_physical(size, device_id)
                
                # 2. Reserve temporary VA to map the new Device Memory (for copying)
                # alignment? usually 2MB or 0
                temp_va = reserve_address(size, 0, 0, 0)
                
                # 3. Map new Device Memory to temp_va
                map_memory(temp_va, size, 0, new_handle, 0)
                set_access(temp_va, size, device_id)
                
                src_hash = None
                if verify:
                    src_hash = int(sample_hash(d_mem, 256))

                # 4. Copy data: Host (d_mem) -> Device (temp_va)
                # d_mem is currently mapped to Host. temp_va is mapped to Device.
                copy_size = size
                if ptr in self.sparse_tensors:
                    copy_size = self.sparse_tensors[ptr]
                    logger.debug("Copying sparse tensor %s: %d bytes (full %d)", ptr, copy_size, size)
                
                copy_memory(temp_va, d_mem, copy_size)

                if verify:
                    dst_hash = int(sample_hash(temp_va, 256))
                    logger.info(
                        "VMM activation copy verify ptr=%s src_hash=%s dst_hash=%s",
                        ptr,
                        src_hash,
                        dst_hash,
                    )
                    if src_hash != dst_hash:
                        raise RuntimeError(
                            f"Activation copy verification failed for ptr {ptr}: "
                            f"src_hash={src_hash}, dst_hash={dst_hash}"
                        )
                
                # 5. Unmap and Free temp_va
                unmap_memory(temp_va, size)
                free_address(temp_va, size)
                
                # 6. Unmap Host Memory from d_mem
                unmap_size = size
                if ptr in self.sparse_tensors:
                    unmap_size = self.sparse_tensors[ptr]
                unmap_memory(d_mem, unmap_size)
                
                # 7. Map new Device Memory to d_mem
                map_memory(d_mem, size, 0, new_handle, 0)
                set_access(d_mem, size, device_id)

                if verify:
                    remap_hash = int(sample_hash(d_mem, 256))
                    logger.info(
                        "VMM activation remap verify ptr=%s src_hash=%s remap_hash=%s",
                        ptr,
                        src_hash,
                        remap_hash,
                    )
                    if src_hash != remap_hash:
                        raise RuntimeError(
                            f"Activation remap verification failed for ptr {ptr}: "
                            f"src_hash={src_hash}, remap_hash={remap_hash}"
                        )
                
                # 8. Update Handle in C++ and release old Host Handle
                old_handle = update_handle(p_memHandle_addr, new_handle)
                release_physical(old_handle)
                
                to_remove.append(ptr)
            except Exception as e:
                logger.error(f"Failed to activate memory for ptr {ptr}: {e}")
                # Don't remove from set if failed?

        for ptr in to_remove:
            self.aliased_pointers.remove(ptr)
            
        logger.info("Memory activation complete")
