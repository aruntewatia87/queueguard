"""Tests for the public API surface and end-to-end determinism."""

from __future__ import annotations

import pytest

import queueguard
from queueguard import (
    BurstyArrivals,
    PoissonArrivals,
    RequestSpec,
    ServiceTimeModel,
    SimulationConfig,
    TraceReplay,
    simulate,
)


def test_version_is_exposed() -> None:
    assert queueguard.__version__ == "0.1.0"


def test_all_is_sorted_the_way_ruff_requires() -> None:
    """RUF022 ordering: uppercase names, then ``__version__``, then lowercase."""
    names = queueguard.__all__
    assert names == sorted(names)
    upper = [n for n in names if n.isupper() or (n[:1].isupper() and n.isidentifier())]
    assert names[: len(upper)] == upper
    assert "__version__" in names
    version_index = names.index("__version__")
    assert all(not n.islower() for n in names[:version_index])
    assert all(n.islower() for n in names[version_index + 1 :])


def test_every_exported_name_exists() -> None:
    for name in queueguard.__all__:
        assert hasattr(queueguard, name), f"{name} is exported but missing"


def test_the_public_api_covers_the_documented_surface() -> None:
    expected = {
        "AdmissionController",
        "AdmissionDecision",
        "AdmissionPolicy",
        "AdmissionResult",
        "AdmittedRequest",
        "BurstyArrivals",
        "CapacityPlan",
        "CapacityPlanner",
        "LatencyStats",
        "PoissonArrivals",
        "QueuePolicy",
        "QueuedRequest",
        "RequestOutcome",
        "RequestSpec",
        "Server",
        "ServiceTimeModel",
        "SimulationConfig",
        "SimulationResult",
        "SpikeAnalysis",
        "TimeToScale",
        "TraceReplay",
        "Workload",
        "__version__",
        "default_request_factory",
        "order_by_policy",
        "simulate",
    }
    assert expected == set(queueguard.__all__)


def test_a_full_run_is_deterministic() -> None:
    """Same seed, same inputs, byte-identical results."""

    def run() -> tuple[float, float, float, int, int]:
        config = SimulationConfig(
            n_parallel=4,
            duration_ms=30_000,
            workload=PoissonArrivals(rate=15.0, seed=99),
            service_model=ServiceTimeModel(jitter_sigma=0.25, seed=7),
            batching_mode="continuous",
            max_batch_size=8,
            seed=99,
        )
        result = simulate(config)
        return (
            result.throughput_rps,
            result.ttft_ms.p99,
            result.end_to_end_ms.p99,
            result.queue_depth_max,
            result.completed,
        )

    assert run() == run()


def test_a_different_seed_changes_the_jitter() -> None:
    def run(seed: int) -> float:
        config = SimulationConfig(
            n_parallel=4,
            duration_ms=30_000,
            workload=PoissonArrivals(rate=15.0, seed=99),
            service_model=ServiceTimeModel(jitter_sigma=0.25, seed=3),
            batching_mode="continuous",
            max_batch_size=8,
            seed=seed,
        )
        return simulate(config).ttft_ms.p99

    assert run(1) != run(2)


def test_the_service_model_seed_drives_standalone_draws() -> None:
    """The model has its own seed for use outside the simulator."""
    request = RequestSpec(input_tokens=300, requested_output_tokens=100)
    first = ServiceTimeModel(jitter_sigma=0.3, seed=11).service_time_ms(request)
    second = ServiceTimeModel(jitter_sigma=0.3, seed=11).service_time_ms(request)
    other = ServiceTimeModel(jitter_sigma=0.3, seed=12).service_time_ms(request)
    assert first == second
    assert first != other


def test_a_trace_replay_round_trips() -> None:
    trace = [
        (0.0, RequestSpec(input_tokens=100, requested_output_tokens=100)),
        (10.0, RequestSpec(input_tokens=200, requested_output_tokens=200)),
    ]
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=30_000,
        workload=TraceReplay(trace),
        service_model=ServiceTimeModel(jitter_sigma=0.0),
        seed=1,
    )
    result = simulate(config)
    assert result.arrived == 2
    assert result.completed == 2


def test_bursty_workload_runs_end_to_end() -> None:
    config = SimulationConfig(
        n_parallel=3,
        duration_ms=20_000,
        workload=BurstyArrivals(rate=10.0, burst_size=5, duty_cycle=0.5, seed=3),
        service_model=ServiceTimeModel(),
        batching_mode="continuous",
        max_batch_size=4,
        seed=3,
    )
    result = simulate(config)
    assert result.arrived > 0
    assert 0.0 <= result.utilization <= 1.0


def test_the_readme_api_tour_snippets_run() -> None:
    """The README's API tour must stay copy-pasteable."""
    service = ServiceTimeModel(
        base_ms=20.0,
        per_input_token_ms=0.6,
        per_output_token_ms=6.0,
        jitter_sigma=0.2,
    )

    result = simulate(
        SimulationConfig(
            n_parallel=4,
            duration_ms=60_000,
            workload=BurstyArrivals(rate=8.0, burst_size=12, duty_cycle=0.5, seed=11),
            service_model=service,
            batching_mode="continuous",
            max_batch_size=8,
            seed=11,
        )
    )
    assert result.throughput_rps == pytest.approx(6.533333333333333)
    assert result.ttft_ms.p99 == pytest.approx(7739.89725550218)
    assert result.utilization == pytest.approx(0.9984375)
    assert result.queue_depth_max == 63

    trace = [(0.0, RequestSpec(input_tokens=512, requested_output_tokens=128))]
    assert TraceReplay(trace, scale=2.0).generate(100.0)[0][1].input_tokens == 1024
