"""Capacity planning for LLM inference fleets.

The question this module answers is the one that decides real money:

    *"How many GPUs do I need to serve 500 requests per second at a p99 TTFT
    under two seconds, and what happens if traffic triples tomorrow?"*

The answer is found by **searching**, not by formula. The relationship between
worker count and tail latency is non-linear (queueing effects kick in), so the
planner runs the simulator at increasing worker counts and returns the smallest
one that still meets the SLO.

It also answers the questions that separate a capacity plan from a capacity
*guess*:

* **What does a traffic spike do?** :meth:`CapacityPlanner.analyze_spike`
  re-runs the plan at a multiple of the baseline traffic and reports the
  breaking point.
* **Can autoscaling save me?** :meth:`CapacityPlanner.autoscale_recommendation`
  applies the standard replica formula and then adds the one caveat that
  matters for GPUs: *a new replica is useless until the model is loaded, which
  takes minutes*. A scale-up cannot save a three-minute promo spike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from queueguard.simulator import SimulationConfig, SimulationResult, simulate
from queueguard.workload import PoissonArrivals, ServiceTimeModel, Workload

__all__ = [
    "CapacityPlan",
    "CapacityPlanner",
    "SpikeAnalysis",
    "TimeToScale",
]


@dataclass(slots=True)
class CapacityPlan:
    """The answer to "how many workers do I need?".

    Attributes:
        required_workers: Minimum number of parallel workers meeting the SLO.
        achieved_throughput_rps: Throughput observed at that worker count.
        target_throughput_rps: Throughput that was asked for.
        p99_ttft_ms: p99 time to first token at the required worker count.
        p99_end_to_end_ms: p99 end-to-end latency at the required worker count.
        utilization: Fraction of worker time spent busy.
        cost_per_million_requests: Cost of serving a million requests.
        cost_per_worker_hour: Hourly cost of a single worker, used for the above.
        headroom_pct: Spare capacity before the SLO is breached.
        scaling_recommendation: Human-readable advice.
        simulation: The simulation the numbers came from.
    """

    required_workers: int
    achieved_throughput_rps: float
    target_throughput_rps: float
    p99_ttft_ms: float
    p99_end_to_end_ms: float
    utilization: float
    cost_per_million_requests: float
    cost_per_worker_hour: float
    headroom_pct: float
    scaling_recommendation: str
    simulation: SimulationResult = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def meets_slo(self) -> bool:
        """Whether the plan meets its target throughput and latency SLO.

        The throughput band is 10%: the number of arrivals in a fixed window
        is itself Poisson-distributed, so demanding an exact match would make
        the check flaky.
        """
        return self.achieved_throughput_rps >= self.target_throughput_rps * 0.9


@dataclass(slots=True)
class SpikeAnalysis:
    """What a traffic spike does to a capacity plan.

    Attributes:
        multiplier: The traffic multiplier that was simulated.
        target_throughput_rps: The offered load after the spike.
        workers_at_baseline: Worker count of the baseline plan.
        workers_needed: Worker count needed to hold the SLO under the spike.
        slo_broken: Whether the baseline worker count fails the SLO under the spike.
        p99_ttft_ms: p99 TTFT at the baseline worker count under the spike.
        utilization: Utilisation at the baseline worker count under the spike.
        breaking_multiplier: The largest multiplier the baseline fleet survives.
        recommendation: Human-readable advice.
    """

    multiplier: float
    target_throughput_rps: float
    workers_at_baseline: int
    workers_needed: int
    slo_broken: bool
    p99_ttft_ms: float
    utilization: float
    breaking_multiplier: float
    recommendation: str


@dataclass(slots=True)
class TimeToScale:
    """How long a scale-up actually takes for a GPU-backed inference fleet.

    Attributes:
        model_load_ms: Time to load the model weights onto a new replica.
        scheduler_latency_ms: Time for the scheduler to route traffic to it.
        total_ms: Total time before the new replica serves useful traffic.
        useful: Whether the scale-up can help at all within a given window.
    """

    model_load_ms: float
    scheduler_latency_ms: float
    total_ms: float
    useful: bool

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.total_ms / 1000:.0f}s to become useful "
            f"(model load {self.model_load_ms / 1000:.0f}s)"
        )


class CapacityPlanner:
    """Finds the cheapest fleet that meets a throughput + latency SLO.

    Args:
        target_throughput_rps: Throughput the fleet must sustain.
        ttft_slo_ms: p99 time-to-first-token budget in milliseconds.
        end_to_end_slo_ms: Optional p99 end-to-end latency budget.
        duration_ms: Length of the simulated window used for each trial.
        service_model: Model used to price requests.
        batching_mode: Serving regime to simulate.
        max_batch_size: Batch size in continuous batching mode.
        max_workers: Upper bound for the worker search.
        seed: Seed used for every trial, so the search is reproducible.
        speed_factor: Server speed multiplier.
        cost_per_worker_hour: Cost of one worker for one hour, used for the
            cost-per-million-requests figure.
        request_factory: Optional factory producing the request shape of the
            offered load.
    """

    def __init__(
        self,
        target_throughput_rps: float = 100.0,
        *,
        ttft_slo_ms: float = 2000.0,
        end_to_end_slo_ms: float | None = None,
        duration_ms: float = 60_000.0,
        service_model: ServiceTimeModel | None = None,
        batching_mode: str = "continuous",
        max_batch_size: int = 8,
        max_workers: int = 512,
        seed: int = 7,
        speed_factor: float = 1.0,
        cost_per_worker_hour: float = 1.5,
        request_factory=None,
    ) -> None:
        if target_throughput_rps <= 0:
            raise ValueError(f"target_throughput_rps must be positive, got {target_throughput_rps}")
        if ttft_slo_ms <= 0:
            raise ValueError(f"ttft_slo_ms must be positive, got {ttft_slo_ms}")
        if end_to_end_slo_ms is not None and end_to_end_slo_ms <= 0:
            raise ValueError(f"end_to_end_slo_ms must be positive, got {end_to_end_slo_ms}")
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        if cost_per_worker_hour < 0:
            raise ValueError(
                f"cost_per_worker_hour must be non-negative, got {cost_per_worker_hour}"
            )
        self._target_throughput_rps = float(target_throughput_rps)
        self._ttft_slo_ms = float(ttft_slo_ms)
        self._end_to_end_slo_ms = end_to_end_slo_ms
        self._duration_ms = float(duration_ms)
        self._service_model = service_model
        self._batching_mode = batching_mode
        self._max_batch_size = int(max_batch_size)
        self._max_workers = int(max_workers)
        self._seed = int(seed)
        self._speed_factor = float(speed_factor)
        self._cost_per_worker_hour = float(cost_per_worker_hour)
        self._request_factory = request_factory

    @property
    def target_throughput_rps(self) -> float:
        """The throughput the fleet must sustain."""
        return self._target_throughput_rps

    @property
    def ttft_slo_ms(self) -> float:
        """The p99 TTFT budget in milliseconds."""
        return self._ttft_slo_ms

    def _resolved_service_model(self) -> ServiceTimeModel:
        """The service model to use, defaulting to the library defaults."""
        if self._service_model is not None:
            return self._service_model
        return ServiceTimeModel()

    def _workload(self, rate: float, seed: int) -> Workload:
        return PoissonArrivals(rate, seed=seed, request_factory=self._request_factory)

    def _simulate(self, workers: int, rate: float) -> SimulationResult:
        config = SimulationConfig(
            n_parallel=workers,
            duration_ms=self._duration_ms,
            workload=self._workload(rate, self._seed),
            service_model=self._resolved_service_model(),
            batching_mode=self._batching_mode,
            max_batch_size=self._max_batch_size,
            speed_factor=self._speed_factor,
            seed=self._seed,
        )
        return simulate(config)

    def _meets_slo(self, result: SimulationResult) -> bool:
        """Whether a trial result satisfies throughput and latency SLOs.

        The throughput band is deliberately loose (10%): the number of arrivals
        in a fixed window is itself Poisson-distributed, so demanding the
        goodput to equal the offered rate exactly would make the search flaky.
        What really decides "can this fleet keep up" is the tail latency - a
        saturated fleet blows the TTFT budget long before it starts dropping
        work.
        """
        if result.throughput_rps < self._target_throughput_rps * 0.9:
            return False
        if result.ttft_ms.p99 > self._ttft_slo_ms:
            return False
        return not (
            self._end_to_end_slo_ms is not None
            and result.end_to_end_ms.p99 > self._end_to_end_slo_ms
        )

    def plan(self, *, max_workers: int | None = None) -> CapacityPlan:
        """Find the minimum worker count that meets the SLO.

        The search is a **binary search over worker counts**: the relationship
        between workers and tail latency is monotone in practice (more workers
        means shorter queues), so the number of simulations is logarithmic in
        the search range rather than linear.

        Args:
            max_workers: Optional override for the search upper bound.

        Returns:
            A :class:`CapacityPlan` describing the minimum viable fleet.

        Raises:
            RuntimeError: If no worker count up to the bound meets the SLO.
        """
        upper = int(max_workers or self._max_workers)
        if upper <= 0:
            raise ValueError(f"max_workers must be positive, got {upper}")

        # Feasibility check: can the fleet keep up at all?
        top = self._simulate(upper, self._target_throughput_rps)
        if not self._meets_slo(top):
            raise RuntimeError(
                f"no fleet up to {upper} workers meets the SLO "
                f"(p99 TTFT {top.ttft_ms.p99:.0f}ms > {self._ttft_slo_ms:.0f}ms budget); "
                f"increase max_workers, relax the SLO, or use a bigger GPU"
            )

        low, high = 1, upper
        best = top
        while low < high:
            mid = (low + high) // 2
            trial = self._simulate(mid, self._target_throughput_rps)
            if self._meets_slo(trial):
                best = trial
                high = mid
            else:
                low = mid + 1
        result = (
            best if best.n_parallel == low else self._simulate(low, self._target_throughput_rps)
        )
        if not self._meets_slo(result):
            result = best

        workers = result.n_parallel
        headroom_pct = max(0.0, (1.0 - result.utilization) * 100.0)
        cost = self.cost_per_million_requests(result)
        recommendation = self._scaling_recommendation(workers, result, headroom_pct)

        return CapacityPlan(
            required_workers=workers,
            achieved_throughput_rps=result.throughput_rps,
            target_throughput_rps=self._target_throughput_rps,
            p99_ttft_ms=result.ttft_ms.p99,
            p99_end_to_end_ms=result.end_to_end_ms.p99,
            utilization=result.utilization,
            cost_per_million_requests=cost,
            cost_per_worker_hour=self._cost_per_worker_hour,
            headroom_pct=headroom_pct,
            scaling_recommendation=recommendation,
            simulation=result,
        )

    def cost_per_million_requests(self, result: SimulationResult) -> float:
        """Cost of serving a million requests with the fleet behind ``result``.

        The fleet burns ``workers * wall_clock_hours`` of worker time, so the
        cost per million requests is::

            cost_per_million = (workers * hours) * cost_per_worker_hour * 1e6 / requests

        Args:
            result: A simulation result whose worker count and throughput are
                used for the arithmetic.

        Returns:
            The cost in the same currency as ``cost_per_worker_hour``.
        """
        if result.throughput_rps <= 0:
            return float("inf")
        hours = 1_000_000.0 / (result.throughput_rps * 3600.0)
        return result.n_parallel * hours * self._cost_per_worker_hour

    def _scaling_recommendation(
        self, workers: int, result: SimulationResult, headroom_pct: float
    ) -> str:
        if headroom_pct < 10.0:
            return (
                f"scale {workers}->{workers + 1} workers now: only {headroom_pct:.0f}% headroom "
                f"left, the p99 TTFT SLO breaks on the next traffic step"
            )
        return (
            f"scale {workers}->{workers + 1} workers before the next peak: "
            f"{headroom_pct:.0f}% headroom at {result.throughput_rps:.1f} rps"
        )

    def analyze_spike(
        self, plan: CapacityPlan, multiplier: float = 3.0, *, max_workers: int | None = None
    ) -> SpikeAnalysis:
        """Simulate a traffic spike against an existing plan.

        Args:
            plan: The baseline plan.
            multiplier: Traffic multiplier, e.g. ``3.0`` for a promo spike.
            max_workers: Optional upper bound for the worker search.

        Returns:
            A :class:`SpikeAnalysis` describing where the SLO breaks.
        """
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        upper = int(max_workers or self._max_workers)
        spiked_rate = plan.target_throughput_rps * multiplier

        baseline = self._simulate(plan.required_workers, spiked_rate)
        slo_broken = not self._meets_slo(baseline)

        workers_needed = plan.required_workers
        if slo_broken:
            workers_needed = self._search_workers(spiked_rate, upper, lower=plan.required_workers)

        # Largest multiplier the baseline fleet survives, by bisection.
        breaking = self._breaking_multiplier(plan, upper)

        if slo_broken:
            recommendation = (
                f"{multiplier:g}x traffic ({spiked_rate:.0f} rps) breaks the p99 TTFT SLO at "
                f"{plan.required_workers} workers (p99 {baseline.ttft_ms.p99:.0f}ms). "
                f"Pre-scale to {workers_needed} workers, and expect the fleet to be saturated "
                f"({baseline.utilization * 100:.0f}% utilisation) - shed low-priority traffic "
                f"rather than queueing it."
            )
            if breaking <= 1.05:
                recommendation += (
                    f" Note: the fleet is sized to the SLO boundary, so it already breaks at "
                    f"{breaking:.2f}x traffic - it has no headroom at all."
                )
        else:
            recommendation = (
                f"{multiplier:g}x traffic ({spiked_rate:.0f} rps) is absorbed by "
                f"{plan.required_workers} workers: p99 TTFT {baseline.ttft_ms.p99:.0f}ms, "
                f"{baseline.utilization * 100:.0f}% utilisation. No action needed, but the "
                f"fleet breaks above {breaking:.2f}x traffic."
            )

        return SpikeAnalysis(
            multiplier=multiplier,
            target_throughput_rps=spiked_rate,
            workers_at_baseline=plan.required_workers,
            workers_needed=workers_needed,
            slo_broken=slo_broken,
            p99_ttft_ms=baseline.ttft_ms.p99,
            utilization=baseline.utilization,
            breaking_multiplier=breaking,
            recommendation=recommendation,
        )

    def _search_workers(self, rate: float, upper: int, lower: int = 1) -> int:
        """Binary search for the smallest worker count meeting the SLO at ``rate``.

        Args:
            rate: The offered load in requests per second.
            upper: The largest worker count to try.
            lower: A known lower bound - the baseline fleet size, for instance.
        """
        top = self._simulate(upper, rate)
        if not self._meets_slo(top):
            return upper
        low, high = max(1, lower), upper
        best = top
        while low < high:
            mid = (low + high) // 2
            trial = self._simulate(mid, rate)
            if self._meets_slo(trial):
                best = trial
                high = mid
            else:
                low = mid + 1
        return best.n_parallel

    def _breaking_multiplier(self, plan: CapacityPlan, upper: int) -> float:
        """Largest traffic multiplier the baseline fleet still survives.

        The search expands geometrically until the fleet fails, then bisects.
        Each step is a full simulation, so the expansion is coarse and the
        bisection is short - a dozen simulations in total.
        """
        lo = 1.0
        hi: float | None = None
        for candidate in (1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
            trial = self._simulate(plan.required_workers, plan.target_throughput_rps * candidate)
            if not self._meets_slo(trial):
                hi = candidate
                break
            lo = candidate
        if hi is None:
            return lo
        for _ in range(6):
            mid = (lo + hi) / 2.0
            trial = self._simulate(plan.required_workers, plan.target_throughput_rps * mid)
            if self._meets_slo(trial):
                lo = mid
            else:
                hi = mid
        return lo

    def autoscale_recommendation(
        self,
        plan: CapacityPlan,
        *,
        target_utilization: float = 0.7,
        model_load_ms: float = 180_000.0,
        scheduler_latency_ms: float = 30_000.0,
        spike_window_ms: float = 300_000.0,
    ) -> tuple[int, TimeToScale, str]:
        """Recommend a replica count for the plan, with the GPU-warmup caveat.

        The standard autoscaling formula (used by the HPA) is::

            desired_replicas = ceil(current_replicas * current_utilization / target)

        That formula is *wrong* for GPU inference in one important way: a new
        replica is not a new CPU. It has to load the model weights into GPU
        memory first, which takes minutes. A scale-up therefore cannot rescue a
        short spike, and the recommendation says so explicitly.

        Args:
            plan: The plan to scale.
            target_utilization: Utilisation the autoscaler aims for.
            model_load_ms: Time to load the model onto a cold replica.
            scheduler_latency_ms: Time before the scheduler routes traffic to it.
            spike_window_ms: Length of the spike the scale-up is supposed to save.

        Returns:
            A tuple of ``(recommended_replicas, time_to_scale, explanation)``.
        """
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")
        current = plan.required_workers
        desired = int(-(-current * plan.utilization // target_utilization))  # ceil
        desired = max(current, desired)
        total = model_load_ms + scheduler_latency_ms
        tts = TimeToScale(
            model_load_ms=model_load_ms,
            scheduler_latency_ms=scheduler_latency_ms,
            total_ms=total,
            useful=total <= spike_window_ms,
        )
        if not tts.useful:
            explanation = (
                f"autoscaler would scale {current}->{desired} replicas at "
                f"{target_utilization:.0%} target utilisation, but a new replica needs "
                f"{total / 1000:.0f}s to load the model and start serving - longer than the "
                f"{spike_window_ms / 1000:.0f}s spike window (model load "
                f"{model_load_ms / 1000:.0f}s). Autoscaling cannot save this spike: "
                f"pre-scale, or shed low-priority traffic instead."
            )
        else:
            explanation = (
                f"autoscaler would scale {current}->{desired} replicas at "
                f"{target_utilization:.0%} target utilisation. The new replica becomes "
                f"useful after {total / 1000:.0f}s (model load {model_load_ms / 1000:.0f}s), "
                f"so pre-warm the pool for spikes shorter than {spike_window_ms / 1000:.0f}s."
            )
        return desired, tts, explanation
