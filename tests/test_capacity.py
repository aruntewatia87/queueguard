"""Tests for the capacity planner, spike analysis and autoscaling advice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from queueguard import CapacityPlanner, RequestSpec, ServiceTimeModel

# A deliberately simple service model so the expected worker counts are stable.
SIMPLE_MODEL = ServiceTimeModel(
    base_ms=10.0, per_input_token_ms=1.0, per_output_token_ms=1.0, jitter_sigma=0.0
)


def _fixed_request(rng: np.random.Generator) -> RequestSpec:
    """Every request is the same size, so the queueing maths is exact."""
    return RequestSpec(input_tokens=200, requested_output_tokens=200)


def _planner(**kwargs) -> CapacityPlanner:
    defaults = {
        "target_throughput_rps": 20.0,
        "ttft_slo_ms": 5000.0,
        "duration_ms": 30_000.0,
        "service_model": SIMPLE_MODEL,
        "batching_mode": "static",
        "max_workers": 128,
        "seed": 17,
        "cost_per_worker_hour": 2.0,
        "request_factory": _fixed_request,
    }
    defaults.update(kwargs)
    return CapacityPlanner(**defaults)


def test_planner_finds_the_minimum_workers_meeting_the_slo() -> None:
    planner = _planner()
    plan = planner.plan()
    assert plan.required_workers > 0
    assert plan.p99_ttft_ms <= planner.ttft_slo_ms
    assert plan.achieved_throughput_rps >= planner.target_throughput_rps * 0.9


def test_one_fewer_worker_fails_the_slo() -> None:
    """The crucial assertion: the plan is *minimal*, not merely sufficient."""
    planner = _planner()
    plan = planner.plan()
    if plan.required_workers == 1:
        pytest.skip("one worker is enough; there is nothing fewer to check")
    weaker = planner._simulate(plan.required_workers - 1, planner.target_throughput_rps)
    assert not planner._meets_slo(weaker), (
        f"{plan.required_workers - 1} workers also meet the SLO, so the plan is not minimal"
    )


def test_the_plan_is_exactly_the_minimum() -> None:
    """Every worker count below the plan fails, not just the one below it."""
    planner = _planner()
    plan = planner.plan()
    for workers in range(1, plan.required_workers):
        trial = planner._simulate(workers, planner.target_throughput_rps)
        assert not planner._meets_slo(trial), f"{workers} workers also meet the SLO"


def test_plan_reports_the_target_throughput() -> None:
    planner = _planner(target_throughput_rps=17.5)
    plan = planner.plan()
    assert plan.target_throughput_rps == pytest.approx(17.5)
    # The achieved throughput is within the Poisson band of the target.
    assert plan.achieved_throughput_rps >= plan.target_throughput_rps * 0.9


def test_more_workers_never_make_the_tail_worse() -> None:
    planner = _planner()
    plan = planner.plan()
    small = planner._simulate(plan.required_workers, planner.target_throughput_rps)
    large = planner._simulate(plan.required_workers * 2, planner.target_throughput_rps)
    assert large.ttft_ms.p99 <= small.ttft_ms.p99 + 1e-6


def test_doubling_the_target_doubles_the_fleet() -> None:
    """Little's Law: twice the load needs roughly twice the workers."""
    small = _planner(target_throughput_rps=10.0).plan()
    large = _planner(target_throughput_rps=20.0).plan()
    assert large.required_workers == pytest.approx(2 * small.required_workers, rel=0.35)


def test_cost_per_million_requests_is_arithmetically_correct() -> None:
    planner = _planner(cost_per_worker_hour=2.0)
    plan = planner.plan()
    result = plan.simulation
    hours = 1_000_000.0 / (result.throughput_rps * 3600.0)
    expected = result.n_parallel * hours * 2.0
    assert plan.cost_per_million_requests == pytest.approx(expected)


def test_cost_scales_with_the_price_of_a_worker() -> None:
    cheap = _planner(cost_per_worker_hour=1.0).plan()
    expensive = _planner(cost_per_worker_hour=2.0).plan()
    assert expensive.cost_per_million_requests == pytest.approx(
        2.0 * cheap.cost_per_million_requests
    )


def test_headroom_is_the_unused_capacity() -> None:
    plan = _planner().plan()
    assert plan.headroom_pct == pytest.approx(max(0.0, (1.0 - plan.utilization) * 100.0))


def test_scaling_recommendation_mentions_the_worker_count() -> None:
    plan = _planner().plan()
    assert str(plan.required_workers) in plan.scaling_recommendation
    assert "scale" in plan.scaling_recommendation


def test_planner_raises_when_no_fleet_meets_the_slo() -> None:
    planner = _planner(target_throughput_rps=5000.0, max_workers=4)
    with pytest.raises(RuntimeError, match="no fleet up to 4 workers meets the SLO"):
        planner.plan()


def test_planner_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="target_throughput_rps must be positive"):
        CapacityPlanner(target_throughput_rps=0.0)
    with pytest.raises(ValueError, match="ttft_slo_ms must be positive"):
        CapacityPlanner(ttft_slo_ms=0.0)
    with pytest.raises(ValueError, match="max_workers must be positive"):
        CapacityPlanner(max_workers=0)
    with pytest.raises(ValueError, match="cost_per_worker_hour must be non-negative"):
        CapacityPlanner(cost_per_worker_hour=-1.0)


def test_spike_analysis_identifies_the_breaking_point() -> None:
    planner = _planner()
    plan = planner.plan()
    spike = planner.analyze_spike(plan, 3.0)
    assert spike.multiplier == 3.0
    assert spike.target_throughput_rps == pytest.approx(plan.target_throughput_rps * 3.0)
    # A 3x spike always saturates a fleet sized for 1x.
    assert spike.slo_broken is True
    assert spike.workers_needed >= plan.required_workers
    assert spike.breaking_multiplier < 3.0
    assert "breaks" in spike.recommendation


def test_spike_analysis_reports_a_healthy_fleet() -> None:
    planner = _planner(target_throughput_rps=5.0)
    plan = planner.plan()
    spike = planner.analyze_spike(plan, 1.05)
    assert spike.slo_broken is False
    assert spike.workers_needed == plan.required_workers
    assert "absorbed" in spike.recommendation


def test_spike_analysis_rejects_a_non_positive_multiplier() -> None:
    planner = _planner()
    plan = planner.plan()
    with pytest.raises(ValueError, match="multiplier must be positive"):
        planner.analyze_spike(plan, 0.0)


def test_autoscale_recommendation_uses_the_standard_formula() -> None:
    planner = _planner()
    plan = planner.plan()
    replicas, tts, explanation = planner.autoscale_recommendation(plan, target_utilization=0.5)
    expected = math.ceil(plan.required_workers * plan.utilization / 0.5)
    assert replicas == max(plan.required_workers, expected)
    assert tts.total_ms == pytest.approx(210_000.0)
    assert "replicas" in explanation


def test_autoscale_never_recommends_fewer_replicas() -> None:
    planner = _planner()
    plan = planner.plan()
    replicas, _tts, _explanation = planner.autoscale_recommendation(plan, target_utilization=0.99)
    assert replicas >= plan.required_workers


def test_autoscale_flags_the_gpu_warmup_caveat() -> None:
    planner = _planner()
    plan = planner.plan()
    _replicas, tts, explanation = planner.autoscale_recommendation(
        plan, model_load_ms=180_000.0, scheduler_latency_ms=30_000.0, spike_window_ms=60_000.0
    )
    # The model takes longer to load than the spike lasts, so scaling is useless.
    assert tts.useful is False
    assert "cannot save this spike" in explanation
    assert "model load" in explanation


def test_autoscale_accepts_a_spike_longer_than_the_model_load() -> None:
    planner = _planner()
    plan = planner.plan()
    _replicas, tts, explanation = planner.autoscale_recommendation(
        plan, model_load_ms=180_000.0, scheduler_latency_ms=30_000.0, spike_window_ms=600_000.0
    )
    assert tts.useful is True
    assert "pre-warm" in explanation


def test_autoscale_rejects_an_invalid_target_utilization() -> None:
    planner = _planner()
    plan = planner.plan()
    with pytest.raises(ValueError, match="target_utilization must be in"):
        planner.autoscale_recommendation(plan, target_utilization=0.0)
    with pytest.raises(ValueError, match="target_utilization must be in"):
        planner.autoscale_recommendation(plan, target_utilization=1.5)


def test_time_to_scale_string_is_readable() -> None:
    planner = _planner()
    plan = planner.plan()
    _replicas, tts, _explanation = planner.autoscale_recommendation(plan)
    assert "210s" in str(tts)


def test_the_planner_is_deterministic() -> None:
    first = _planner().plan()
    second = _planner().plan()
    assert first.required_workers == second.required_workers
    assert first.p99_ttft_ms == second.p99_ttft_ms
    assert first.cost_per_million_requests == second.cost_per_million_requests
