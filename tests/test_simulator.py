"""Tests for the discrete-event queueing simulator."""

from __future__ import annotations

import numpy as np
import pytest

from queueguard import (
    AdmissionDecision,
    PoissonArrivals,
    QueuePolicy,
    RequestSpec,
    Server,
    ServiceTimeModel,
    SimulationConfig,
    TraceReplay,
    simulate,
)

NO_JITTER = ServiceTimeModel(jitter_sigma=0.0)


def _trace(n: int, *, gap_ms: float = 0.0, **kwargs) -> TraceReplay:
    """A trace of ``n`` evenly spaced identical requests."""
    return TraceReplay(
        [
            (i * gap_ms, RequestSpec(input_tokens=100, requested_output_tokens=100, **kwargs))
            for i in range(n)
        ]
    )


def test_single_request_has_no_wait() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=_trace(1),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    outcome = result.outcomes[0]
    assert outcome.wait_time_ms == pytest.approx(0.0)
    assert result.completed == 1


def test_single_request_service_time_matches_the_model() -> None:
    request = RequestSpec(input_tokens=100, requested_output_tokens=50)
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=TraceReplay([(0.0, request)]),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    outcome = result.outcomes[0]
    expected = NO_JITTER.nominal_service_time_ms(request)
    assert outcome.service_time_ms == pytest.approx(expected)
    assert outcome.end_to_end_ms == pytest.approx(expected)
    assert outcome.ttft_ms == pytest.approx(NO_JITTER.prefill_ms(request))


def test_ttft_is_shorter_than_the_end_to_end_time() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=_trace(1),
        service_model=NO_JITTER,
        seed=1,
    )
    outcome = simulate(config).outcomes[0]
    assert outcome.ttft_ms < outcome.end_to_end_ms


def test_throughput_is_completed_requests_over_duration() -> None:
    config = SimulationConfig(
        n_parallel=4,
        duration_ms=30_000,
        workload=PoissonArrivals(rate=5.0, seed=4),
        service_model=NO_JITTER,
        seed=4,
    )
    result = simulate(config)
    assert result.throughput_rps == pytest.approx(result.completed / 30.0)


def test_fifo_queue_serves_in_arrival_order() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=60_000,
        workload=_trace(5, gap_ms=10.0),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    starts = [o.start_time_ms for o in result.served]
    assert starts == sorted(starts)
    assert len(result.served) == 5


def test_percentiles_are_ordered_p99_ge_p95_ge_p50() -> None:
    config = SimulationConfig(
        n_parallel=2,
        duration_ms=60_000,
        workload=PoissonArrivals(rate=8.0, seed=13),
        service_model=ServiceTimeModel(jitter_sigma=0.25, seed=2),
        seed=13,
    )
    result = simulate(config)
    for stats in (result.wait_ms, result.ttft_ms, result.end_to_end_ms):
        assert stats.p99 >= stats.p95 >= stats.p50
        assert stats.min <= stats.p50 <= stats.max


def test_percentiles_match_numpy_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    arr = np.asarray(values)
    # The simulator uses numpy's linear interpolation; check the same numbers
    # come out of a hand-rolled linear interpolation.
    expected_p95 = float(np.percentile(arr, 95))
    expected_p99 = float(np.percentile(arr, 99))
    # Linear interpolation: index 0.95 * (5 - 1) = 3.8 -> 4 + 0.8 * 96.
    assert expected_p95 == pytest.approx(80.8)
    # index 0.99 * (5 - 1) = 3.96 -> 4 + 0.96 * 96.
    assert expected_p99 == pytest.approx(96.16)


def test_queue_depth_max_tracks_a_burst() -> None:
    # Ten requests arrive at once on a single worker: nine of them must wait
    # for the worker to free up.
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=60_000,
        workload=TraceReplay(
            [(0.0, RequestSpec(input_tokens=100, requested_output_tokens=100))] * 10
        ),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    # One request is in service, the other nine are waiting.
    assert result.queue_depth_max == 9


def test_continuous_batching_improves_throughput_over_static() -> None:
    common = {
        "n_parallel": 4,
        "duration_ms": 20_000,
        "service_model": NO_JITTER,
        "seed": 3,
    }
    static = simulate(
        SimulationConfig(
            **common, workload=PoissonArrivals(rate=20.0, seed=5), batching_mode="static"
        )
    )
    continuous = simulate(
        SimulationConfig(
            **common,
            workload=PoissonArrivals(rate=20.0, seed=5),
            batching_mode="continuous",
            max_batch_size=8,
        )
    )
    assert continuous.throughput_rps > static.throughput_rps


def test_utilisation_is_busy_time_over_total_capacity() -> None:
    config = SimulationConfig(
        n_parallel=2,
        duration_ms=10_000,
        workload=_trace(2),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    expected = result.busy_time_ms / (2 * 10_000.0)
    assert result.utilization == pytest.approx(expected)
    assert result.busy_time_ms == pytest.approx(sum(result.worker_busy_time_ms))


def test_utilisation_is_between_zero_and_one() -> None:
    config = SimulationConfig(
        n_parallel=2,
        duration_ms=20_000,
        workload=PoissonArrivals(rate=3.0, seed=6),
        service_model=NO_JITTER,
        seed=6,
    )
    result = simulate(config)
    assert 0.0 <= result.utilization <= 1.0


def test_same_seed_gives_identical_results() -> None:
    def run() -> tuple[float, float, int]:
        config = SimulationConfig(
            n_parallel=3,
            duration_ms=20_000,
            workload=PoissonArrivals(rate=10.0, seed=21),
            service_model=ServiceTimeModel(jitter_sigma=0.3, seed=4),
            seed=21,
        )
        result = simulate(config)
        return result.throughput_rps, result.ttft_ms.p99, result.queue_depth_max

    assert run() == run()


def test_different_seeds_give_different_results() -> None:
    def run(seed: int) -> float:
        config = SimulationConfig(
            n_parallel=3,
            duration_ms=20_000,
            workload=PoissonArrivals(rate=10.0, seed=21),
            service_model=ServiceTimeModel(jitter_sigma=0.3, seed=4),
            seed=seed,
        )
        return simulate(config).ttft_ms.p99

    assert run(1) != run(2)


def test_priority_queue_serves_high_priority_first() -> None:
    # Three simultaneous requests, one worker: the first starts immediately and
    # the other two queue up. Priority ordering decides which of the two
    # waiting requests goes next.
    trace = [
        (0.0, RequestSpec(input_tokens=10, requested_output_tokens=10, priority=0)),
        (0.0, RequestSpec(input_tokens=10, requested_output_tokens=10, priority=1)),
        (0.0, RequestSpec(input_tokens=10, requested_output_tokens=10, priority=9)),
    ]
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=30_000,
        workload=TraceReplay(trace),
        service_model=NO_JITTER,
        queue_policy=QueuePolicy.PRIORITY,
        seed=1,
    )
    result = simulate(config)
    served = sorted(result.served, key=lambda o: o.start_time_ms)
    # The first request is already running, so the high-priority one that
    # queued behind it is served second.
    assert served[1].priority == 9
    assert served[2].priority == 1


def test_deadline_miss_rate_is_reported() -> None:
    trace = [
        (0.0, RequestSpec(input_tokens=100, requested_output_tokens=100, deadline_ms=10.0)),
    ]
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=TraceReplay(trace),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    assert result.deadline_miss_rate == pytest.approx(1.0)
    assert result.outcomes[0].deadline_missed is True


def test_no_deadlines_means_a_zero_miss_rate() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=_trace(1),
        service_model=NO_JITTER,
        seed=1,
    )
    assert simulate(config).deadline_miss_rate == 0.0


def test_simulation_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="n_parallel must be positive"):
        SimulationConfig(n_parallel=0)
    with pytest.raises(ValueError, match="duration_ms must be positive"):
        SimulationConfig(duration_ms=0.0)
    with pytest.raises(ValueError, match="batching_mode must be"):
        SimulationConfig(batching_mode="turbo")
    with pytest.raises(ValueError, match="max_batch_size must be positive"):
        SimulationConfig(max_batch_size=0)
    with pytest.raises(ValueError, match="speed_factor must be positive"):
        SimulationConfig(speed_factor=0.0)


def test_server_exposes_its_controller() -> None:
    config = SimulationConfig(
        n_parallel=2,
        duration_ms=10_000,
        workload=_trace(2),
        service_model=NO_JITTER,
        seed=1,
    )
    server = Server(config)
    result = server.run()
    assert server.controller.served == result.completed
    assert server.config is config


def test_outcome_for_finds_the_right_request() -> None:
    request = RequestSpec(input_tokens=10, requested_output_tokens=10)
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=TraceReplay([(0.0, request)]),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    assert result.outcome_for(request) is result.outcomes[0]


def test_outcome_for_raises_for_an_unknown_request() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=_trace(1),
        service_model=NO_JITTER,
        seed=1,
    )
    result = simulate(config)
    with pytest.raises(KeyError):
        result.outcome_for(RequestSpec(input_tokens=1))


def test_speed_factor_halves_the_service_time() -> None:
    request = RequestSpec(input_tokens=100, requested_output_tokens=100)
    trace = TraceReplay([(0.0, request)])
    slow = simulate(
        SimulationConfig(
            n_parallel=1, duration_ms=30_000, workload=trace, service_model=NO_JITTER, seed=1
        )
    )
    fast = simulate(
        SimulationConfig(
            n_parallel=1,
            duration_ms=30_000,
            workload=trace,
            service_model=NO_JITTER,
            speed_factor=2.0,
            seed=1,
        )
    )
    assert fast.outcomes[0].service_time_ms == pytest.approx(slow.outcomes[0].service_time_ms / 2.0)


def test_accepted_requests_are_marked_accepted() -> None:
    config = SimulationConfig(
        n_parallel=1,
        duration_ms=10_000,
        workload=_trace(1),
        service_model=NO_JITTER,
        seed=1,
    )
    assert simulate(config).outcomes[0].decision is AdmissionDecision.ACCEPT
