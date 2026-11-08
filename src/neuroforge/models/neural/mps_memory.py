"""Bound MPS allocator cache without disabling its safety limits."""
import torch


def housekeeping(device,label,budget_gib=8.):
    if device!='mps':return None
    torch.mps.synchronize()
    before=torch.mps.driver_allocated_memory()/1024**3
    torch.mps.empty_cache()
    after=torch.mps.driver_allocated_memory()/1024**3
    live=torch.mps.current_allocated_memory()/1024**3
    print(f'[MPS memory {label}] tensors={live:.2f} GiB driver={before:.2f}->{after:.2f} GiB',flush=True)
    if after>budget_gib:raise RuntimeError(f'MPS driver allocation remains {after:.2f} GiB after cleanup; stopping at configured {budget_gib:.2f} GiB budget')
    return {'label':label,'tensor_gib':live,'driver_before_gib':before,'driver_after_gib':after}
