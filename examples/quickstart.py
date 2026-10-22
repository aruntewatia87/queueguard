#!/usr/bin/env python3
"""End-to-end demo: bursty traffic, shedding, capacity planning and a spike."""

from __future__ import annotations

from queueguard import (
    AdmissionController,
    AdmissionPolicy,
    BurstyArrivals,
    CapacityPlanner,
    PoissonArrivals,
    QueuePolicy,
    RequestSpec,
    ServiceTimeModel,
    SimulationConfig,
    simulate,
)

SERVICE = ServiceTimeModel(
    base_ms=20.0, per_input_token_ms=0.6, per_output_token_ms=6.0, jitter_sigma=0.2, seed=7
)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    # ------------------------------------------------------------------
    section("1. Bursty workload against a 4-worker server")
    # ------------------------------------------------------------------
    workload = BurstyArrivals(rate=8.0, burst_size=12, duty_cycle=0.5, seed=11)
    config = SimulationConfig(
        n_parallel=4,
        duration_ms=60_000,
        workload=workload,
        service_model=SERVICE,
        batching_mode="continuous",
        max_batch_size=8,
        seed=11,
    )
    result = simulate(config)

    print(f"  arrived        {result.arrived:>6}")
    print(f"  completed      {result.completed:>6}")
    print(f"  dropped        {result.dropped:>6}")
    print(f"  queue depth    {result.queue_depth_max:>6} (max)")
    print(f"  utilisation    {result.utilization * 100:>5.1f}%")
    print(f"  throughput     {result.throughput_rps:>6.2f} rps")
    print()
    print("  latency percentiles (ms)")
    print(f"    {'':10} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    for label, stats in (
        ("wait", result.wait_ms),
        ("TTFT", result.ttft_ms),
        ("end-to-end", result.end_to_end_ms),
    ):
        print(
            f"    {label:<10} {stats.p50:>9.0f} {stats.p95:>9.0f} "
            f"{stats.p99:>9.0f} {stats.max:>9.0f}"
        )

    # ------------------------------------------------------------------
    section("2. Priority-aware load shedding")
    # ------------------------------------------------------------------
    controller = AdmissionController(
        service_model=SERVICE,
        policy=AdmissionPolicy.VALUE_DENSITY,
        queue_policy=QueuePolicy.PRIORITY,
        high_water_mark=6,
        workers=1,
    )
    arrivals = [
        (float(i * 100), RequestSpec(input_tokens=400, requested_output_tokens=200, priority=p))
        for i, p in enumerate([0, 1, 0, 5, 0, 9, 0, 2, 0, 7])
    ]
    decisions = []
    for at, request in arrivals:
        admitted = controller.admit(request, now_ms=at, available_workers=0)
        decisions.append((request.priority, admitted.decision.value))

    print("  high-water mark = 6, incoming priorities: 0 1 0 5 0 9 0 2 0 7")
    for priority, decision in decisions:
        marker = {"accept": "+", "queue": " ", "shed": "x", "reject": "-"}[decision]
        print(f"    priority {priority} -> {decision:<7} {marker}")
    print(f"  shed: {controller.shed}   still queued: {len(controller)}")
    print(f"  survivors: {sorted(q.priority for q in controller.pending)}")
    print(f"  evicted:   {sorted(q.priority for q in controller.evicted)}")

    # ------------------------------------------------------------------
    section("3. Capacity plan: 60 rps at p99 TTFT < 2000 ms")
    # ------------------------------------------------------------------
    planner = CapacityPlanner(
        target_throughput_rps=60.0,
        ttft_slo_ms=2000.0,
        duration_ms=60_000,
        service_model=SERVICE,
        batching_mode="continuous",
        max_batch_size=8,
        max_workers=256,
        seed=23,
        cost_per_worker_hour=1.75,
    )
    plan = planner.plan()
    print(f"  required workers   {plan.required_workers}")
    print(f"  achieved           {plan.achieved_throughput_rps:.1f} rps")
    print(f"  p99 TTFT           {plan.p99_ttft_ms:.0f} ms (budget 2000 ms)")
    print(f"  p99 end-to-end     {plan.p99_end_to_end_ms:.0f} ms")
    print(f"  utilisation        {plan.utilization * 100:.1f}%")
    print(f"  headroom           {plan.headroom_pct:.0f}%")
    print(f"  cost / 1M requests ${plan.cost_per_million_requests:.2f}")
    print(f"  advice: {plan.scaling_recommendation}")

    # ------------------------------------------------------------------
    section("4. Three-fold traffic spike")
    # ------------------------------------------------------------------
    spike = planner.analyze_spike(plan, 3.0)
    print(f"  offered load       {spike.target_throughput_rps:.0f} rps")
    print(f"  SLO broken         {spike.slo_broken}")
    print(f"  p99 TTFT           {spike.p99_ttft_ms:.0f} ms")
    print(f"  utilisation        {spike.utilization * 100:.1f}%")
    print(f"  workers needed     {spike.workers_needed}")
    print(f"  breaks above       {spike.breaking_multiplier:.2f}x traffic")
    print(f"  advice: {spike.recommendation}")

    # ------------------------------------------------------------------
    section("5. Can autoscaling save us?")
    # ------------------------------------------------------------------
    replicas, time_to_scale, explanation = planner.autoscale_recommendation(
        plan,
        target_utilization=0.7,
        model_load_ms=180_000,
        scheduler_latency_ms=30_000,
        spike_window_ms=300_000,
    )
    print(f"  recommended replicas {replicas}")
    print(f"  time to scale        {time_to_scale}")
    print(f"  useful for the spike {time_to_scale.useful}")
    print(f"  advice: {explanation}")

    # ------------------------------------------------------------------
    section("6. Little's Law check (L = lambda * W)")
    # ------------------------------------------------------------------
    steady = simulate(
        SimulationConfig(
            n_parallel=plan.required_workers,
            duration_ms=60_000,
            workload=PoissonArrivals(rate=plan.target_throughput_rps, seed=23),
            service_model=SERVICE,
            batching_mode="continuous",
            max_batch_size=8,
            seed=23,
        )
    )
    mean_wait_s = steady.wait_ms.mean / 1000.0
    little = steady.throughput_rps * mean_wait_s
    mean_e2e_s = steady.end_to_end_ms.mean / 1000.0
    print(f"  throughput (lambda)  {steady.throughput_rps:.2f} rps")
    print(f"  mean wait (W)        {mean_wait_s:.3f} s")
    print(f"  mean end-to-end      {mean_e2e_s:.3f} s")
    print(f"  L = lambda * W       {little:.1f} requests waiting")
    print(f"  L = lambda * E2E     {steady.mean_requests_in_system:.1f} requests in the system")
    print(f"  queue depth (max)    {steady.queue_depth_max}")


if __name__ == "__main__":
    main()
