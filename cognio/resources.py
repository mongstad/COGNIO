"""Configuration-sized allocation safeguards, checked before model construction."""

import math
import psutil
import torch

MIB = 1024**2


def check_budget(config, device, budget_mib=None):
    host_available = psutil.virtual_memory().available
    limit = int(host_available * .5)
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        limit = min(limit, int(free * .6))
    if budget_mib is not None:
        if not math.isfinite(budget_mib) or budget_mib <= 0:
            raise ValueError("memory budget must be positive finite MiB")
        limit = min(limit, int(budget_mib * MIB))
    if config.working_bytes > limit:
        raise ValueError(f"{config.channels}×{config.grid} needs an estimated {config.working_bytes/MIB:.0f} MiB "
                         f"working allowance, above the available budget {limit/MIB:.0f} MiB. "
                         "Reduce channels/grid, close other consumers, or raise an explicitly smaller --memory-budget-mib.")
    return dict(estimated_working_mib=config.working_bytes/MIB, budget_mib=limit/MIB)
