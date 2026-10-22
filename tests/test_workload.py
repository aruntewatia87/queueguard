"""Tests for the workload and service-time models."""

from __future__ import annotations

import numpy as np
import pytest

from queueguard import (
    BurstyArrivals,
    PoissonArrivals,
    RequestSpec,
    ServiceTimeModel,
    TraceReplay,
)


def test_request_spec_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        RequestSpec(input_tokens=-1)
    with pytest.raises(ValueError, match="requested_output_tokens must be non-negative"):
        RequestSpec(requested_output_tokens=-5)


def test_request_spec_rejects_negative_priority() -> None:
    with pytest.raises(ValueError, match="priority must be non-negative"):
        RequestSpec(priority=-1)


def test_request_spec_rejects_non_positive_deadline() -> None:
    with pytest.raises(ValueError, match="deadline_ms must be positive"):
        RequestSpec(deadline_ms=0)


def test_request_spec_total_tokens_is_the_sum() -> None:
    request = RequestSpec(input_tokens=100, requested_output_tokens=50)
    assert request.total_tokens == 150


def test_poisson_interarrivals_have_the_right_mean() -> None:
    workload = PoissonArrivals(rate=20.0, seed=42)
    arrivals = workload.generate(60_000.0)
    gaps = np.diff([0.0] + [t for t, _ in arrivals])
    # 20 rps means a 50ms mean gap; a generous band keeps the test non-flaky.
    assert 40.0 < gaps.mean() < 60.0


def test_poisson_arrival_count_is_in_a_sane_band() -> None:
    workload = PoissonArrivals(rate=20.0, seed=7)
    arrivals = workload.generate(60_000.0)
    # 20 rps for 60s is 1200 arrivals; Poisson spread is about 35.
    assert 1080 < len(arrivals) < 1320


def test_poisson_arrivals_are_sorted_in_time() -> None:
    arrivals = PoissonArrivals(rate=50.0, seed=3).generate(10_000.0)
    times = [t for t, _ in arrivals]
    assert times == sorted(times)


def test_poisson_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="rate must be positive"):
        PoissonArrivals(rate=0.0)


def test_poisson_generates_nothing_for_a_zero_duration() -> None:
    assert PoissonArrivals(rate=10.0, seed=1).generate(0.0) == []


def test_bursty_arrivals_produce_bursts() -> None:
    workload = BurstyArrivals(rate=20.0, burst_size=10, duty_cycle=0.5, seed=1)
    arrivals = workload.generate(20_000.0)
    times = [t for t, _ in arrivals]
    # With 10 requests per burst spread over the active half of a 500ms
    # period, the gaps inside a burst are small and the gaps between bursts
    # are large.
    gaps = np.diff(times)
    assert gaps.min() < 40.0
    assert gaps.max() > 200.0


def test_bursty_duty_cycle_controls_the_idle_fraction() -> None:
    busy = BurstyArrivals(rate=20.0, burst_size=10, duty_cycle=0.9, seed=1).generate(20_000.0)
    idle = BurstyArrivals(rate=20.0, burst_size=10, duty_cycle=0.2, seed=1).generate(20_000.0)
    # The average rate is fixed by `rate`, so the counts match; the duty cycle
    # changes the *shape*: a higher duty cycle means a shorter quiet gap.
    assert len(busy) == len(idle)
    busy_gap = np.diff([t for t, _ in busy]).max()
    idle_gap = np.diff([t for t, _ in idle]).max()
    assert busy_gap < idle_gap


def test_bursty_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="burst_size must be positive"):
        BurstyArrivals(rate=1.0, burst_size=0)
    with pytest.raises(ValueError, match="duty_cycle must be in"):
        BurstyArrivals(rate=1.0, duty_cycle=0.0)


def test_trace_replay_returns_the_recorded_trace() -> None:
    trace = [(0.0, RequestSpec(input_tokens=10)), (5.0, RequestSpec(input_tokens=20))]
    replayed = TraceReplay(trace).generate(100.0)
    assert [t for t, _ in replayed] == [0.0, 5.0]


def test_trace_replay_clips_to_the_window() -> None:
    trace = [(0.0, RequestSpec()), (500.0, RequestSpec())]
    replayed = TraceReplay(trace).generate(100.0)
    assert len(replayed) == 1


def test_trace_replay_scale_resizes_requests() -> None:
    trace = [(0.0, RequestSpec(input_tokens=100, requested_output_tokens=50))]
    replayed = TraceReplay(trace, scale=2.0).generate(100.0)
    _, request = replayed[0]
    assert request.input_tokens == 200
    assert request.requested_output_tokens == 100


def test_service_time_grows_with_token_counts() -> None:
    model = ServiceTimeModel(jitter_sigma=0.0)
    small = model.nominal_service_time_ms(RequestSpec(input_tokens=100, requested_output_tokens=50))
    bigger_prompt = model.nominal_service_time_ms(
        RequestSpec(input_tokens=200, requested_output_tokens=50)
    )
    bigger_reply = model.nominal_service_time_ms(
        RequestSpec(input_tokens=100, requested_output_tokens=100)
    )
    assert bigger_prompt > small
    assert bigger_reply > small


def test_service_time_is_monotonic_in_output_tokens() -> None:
    model = ServiceTimeModel(jitter_sigma=0.0)
    previous = -1.0
    for tokens in range(0, 500, 25):
        current = model.nominal_service_time_ms(RequestSpec(requested_output_tokens=tokens))
        assert current > previous
        previous = current


def test_ttft_is_a_fraction_of_the_total_service_time() -> None:
    model = ServiceTimeModel(jitter_sigma=0.0)
    request = RequestSpec(input_tokens=400, requested_output_tokens=200)
    total = model.nominal_service_time_ms(request)
    assert model.prefill_ms(request) < total
    assert 0.0 < model.ttft_fraction(request) < 1.0
    assert model.prefill_ms(request) + model.decode_ms(request) == pytest.approx(total)


def test_jitter_is_deterministic_under_a_seed() -> None:
    model = ServiceTimeModel(jitter_sigma=0.3, seed=99)
    request = RequestSpec(input_tokens=300, requested_output_tokens=100)
    first = model.service_time_ms(request)
    second = model.service_time_ms(request)
    assert first == second


def test_jitter_changes_the_service_time() -> None:
    model = ServiceTimeModel(jitter_sigma=0.3, seed=5)
    request = RequestSpec(input_tokens=300, requested_output_tokens=100)
    nominal = model.nominal_service_time_ms(request)
    assert model.service_time_ms(request) != nominal


def test_zero_jitter_gives_the_nominal_time() -> None:
    model = ServiceTimeModel(jitter_sigma=0.0)
    request = RequestSpec(input_tokens=300, requested_output_tokens=100)
    assert model.service_time_ms(request) == model.nominal_service_time_ms(request)


def test_service_model_rejects_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="base_ms must be non-negative"):
        ServiceTimeModel(base_ms=-1.0)
    with pytest.raises(ValueError, match="per_output_token_ms must be non-negative"):
        ServiceTimeModel(per_output_token_ms=-1.0)
    with pytest.raises(ValueError, match="jitter_sigma must be non-negative"):
        ServiceTimeModel(jitter_sigma=-0.1)


def test_split_preserves_the_ttft_ratio() -> None:
    model = ServiceTimeModel(jitter_sigma=0.4, seed=2)
    request = RequestSpec(input_tokens=800, requested_output_tokens=100)
    prefill, decode = model.split(request)
    assert prefill + decode == pytest.approx(model.service_time_ms(request))
