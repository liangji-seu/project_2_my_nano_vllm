"""Pure helpers for GPU memory budgeting; importing this module does not touch CUDA."""


def calculate_available_kv_cache_memory(
    initial_free_memory: int,
    utilization: float,
    reserved_baseline: int,
    reserved_before_profile: int,
    activation_peak: int,
    free_after_profile: int,
) -> int:
    """Compute KV bytes from an initial-free budget and current physical headroom."""
    worker_torch_growth = max(0, reserved_before_profile - reserved_baseline)
    policy_budget = int(initial_free_memory * utilization)
    policy_available = policy_budget - worker_torch_growth - activation_peak
    physical_available = free_after_profile - activation_peak
    return min(policy_available, physical_available)
