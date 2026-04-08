"""
This module introduces some fixes for running the colbert package on Windows or MacOS.
"""
import os
import torch
import torch.nn as nn
import colbert.infra.launcher

# Keep a reference to the original functions
original_setup_new_process = colbert.infra.launcher.setup_new_process
original_DDP = nn.parallel.DistributedDataParallel


def patched_DDP(module, *args, **kwargs):
    """
        A patched version of DistributedDataParallel that sets device_ids and output_device to None
        when the model is running on CPU, to avoid a ValueError from PyTorch.
    """
    is_cpu = all(p.device.type == 'cpu' for p in module.parameters())
    if is_cpu:
        kwargs['device_ids'] = None
        kwargs['output_device'] = None
    
    return original_DDP(module, *args, **kwargs)


def patched_setup_new_process(*args, **kwargs):
    """
        A patched version of colbert.infra.launcher.setup_new_process that:
        1. Sets default environment variables for torch.distributed.
        2. Initializes the process group.
        3. Patches DistributedDataParallel to handle CPU-only training correctly.
        4. Prevents the original setup function from re-initializing the process group.
    """
    # 1. Set environment variables for a single-process run if not set.
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # 2. Initialize the process group.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='gloo', init_method='env://')

    # 3. Patch DistributedDataParallel for CPU training.
    nn.parallel.DistributedDataParallel = patched_DDP

    # 4. Temporarily replace colbert's distributed init to avoid re-initialization.
    import colbert.utils.distributed
    original_colbert_dist_init = colbert.utils.distributed.init

    def mocked_dist_init(*_args, **_kwargs):
        # The original returns (world_size, distributed_module)
        return torch.distributed.get_world_size(), colbert.utils.distributed

    colbert.utils.distributed.init = mocked_dist_init

    # Call the original setup function.
    result = original_setup_new_process(*args, **kwargs)

    # Restore original function
    colbert.utils.distributed.init = original_colbert_dist_init

    return result


# Apply the monkey-patch
colbert.infra.launcher.setup_new_process = patched_setup_new_process
