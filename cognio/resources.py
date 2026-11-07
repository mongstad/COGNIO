"""Configuration-sized allocation safeguards, checked before model construction."""

import math
import psutil
import torch

MIB = 1024**2


def check_budget(config, device, budget_mib=None):
    host_available = psutil.virtual_memory().available
    host_limit = int(host_available * .5)
    limit = host_limit
    gpu_free = None
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        gpu_free = free
        limit = min(limit, int(free * .6))
    if budget_mib is not None:
        if not math.isfinite(budget_mib) or budget_mib <= 0:
            raise ValueError("memory budget must be positive finite MiB")
        limit = min(limit, int(budget_mib * MIB))
    if config.working_bytes > limit:
        raise ValueError(f"{config.channels}×{config.grid} needs an estimated {config.working_bytes/MIB:.0f} MiB "
                         f"working allowance, above the available budget {limit/MIB:.0f} MiB. "
                         "Reduce channels/grid, close other consumers, or raise an explicitly smaller --memory-budget-mib.")
    return dict(estimated_working_mib=config.working_bytes/MIB,
                estimated_host_total_mib=(psutil.Process().memory_info().rss + config.working_bytes)/MIB, budget_mib=limit/MIB,
                host_available_mib=host_available/MIB, host_budget_mib=host_limit/MIB,
                gpu_free_mib=gpu_free/MIB if gpu_free is not None else None,
                gpu_budget_mib=.6*gpu_free/MIB if gpu_free is not None else None)
