from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_vllm.worker.gpu_memory import calculate_available_kv_cache_memory


def budget(initial_free, utilization, reserved_baseline, reserved_before_profile, activation, free_after_profile):
    return calculate_available_kv_cache_memory(
        initial_free_memory=initial_free,
        utilization=utilization,
        reserved_baseline=reserved_baseline,
        reserved_before_profile=reserved_before_profile,
        activation_peak=activation,
        free_after_profile=free_after_profile,
    )


def test_normal_budget_subtracts_worker_growth_and_activation():
    assert budget(10_000, 0.8, 1_000, 3_000, 1_000, 8_000) == 5_000


def test_existing_other_process_occupancy_is_outside_initial_free_budget():
    # Initial free memory already excludes other processes; do not subtract them again.
    assert budget(12_000, 0.5, 2_000, 3_000, 1_000, 10_000) == 4_000


def test_external_growth_caps_budget_and_preserves_activation_headroom():
    assert budget(20_000, 0.8, 1_000, 3_000, 1_000, 2_500) == 1_500


def test_negative_policy_budget_is_preserved_for_worker_error_path():
    assert budget(8_000, 0.5, 1_000, 4_000, 2_000, 10_000) == -1_000


def test_negative_physical_headroom_is_preserved_for_worker_error_path():
    assert budget(20_000, 0.8, 1_000, 2_000, 1_000, 500) == -500
