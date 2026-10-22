"""Discrete-event queueing simulator for LLM inference endpoints.

The simulator answers the question every AI Cloud Solutions Architect has to
answer before quoting a number of GPUs: *if I send this much traffic at this
GPU fleet, what does the latency look like, and where does it break?*

Design
------
The engine is a **discrete-event simulation**. Time advances to the next
interesting instant (an arrival, a prefill finishing, a decode finishing) and
nothing in between is simulated. This is what makes it possible to reason
about a 24-hour trace in milliseconds of CPU time.

Two serving regimes are modelled:

``static``
    One request per worker. The classic "one slot, one sequence" server. The
    prefill runs, then the decode runs, then the worker is free.

``continuous``
    A worker keeps a *batch*. New requests join the batch; a new member's
    prefill waits for the GPU (and steals decode cycles, which is exactly what
    happens in a real engine), and the decoders that run concurrently share
    the GPU, each slowed down by ``batch_penalty`` per extra batch member.

The decomposition into prefill and decode is the point of the whole model: the
prefill is the **time to first token** that users actually perceive, and the
decode phase is where the tokens are produced.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from queueguard.admission import (
    AdmissionController,
    AdmissionDecision,
    AdmissionPolicy,
    QueuePolicy,
)
from queueguard.workload import PoissonArrivals, RequestSpec, ServiceTimeModel, Workload

__all__ = [
    "LatencyStats",
    "RequestOutcome",
    "Server",
    "SimulationConfig",
    "SimulationResult",
    "simulate",
]

_ARRIVAL = 0
_WORKER_TICK = 1

_EPS = 1e-9


@dataclass(slots=True)
class LatencyStats:
    """Summary statistics for a latency distribution, in milliseconds."""

    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


@dataclass(slots=True, eq=False)
class RequestOutcome:
    """What happened to a single request.

    Identity comparison is used deliberately: two outcomes with the same fields
    are still two different requests.
    """

    request: RequestSpec
    arrival_time_ms: float
    start_time_ms: float | None = None
    completion_time_ms: float | None = None
    wait_time_ms: float | None = None
    service_time_ms: float | None = None
    ttft_ms: float | None = None
    end_to_end_ms: float | None = None
    decision: AdmissionDecision = AdmissionDecision.QUEUE
    reason: str = ""
    dropped: bool = False
    drop_reason: str = ""

    @property
    def priority(self) -> int:
        """Priority of the request."""
        return self.request.priority

    @property
    def tenant_id(self) -> str:
        """Tenant that issued the request."""
        return self.request.tenant_id

    @property
    def deadline_missed(self) -> bool:
        """Whether the request missed its deadline."""
        if self.request.deadline_ms is None or self.end_to_end_ms is None:
            return False
        return self.end_to_end_ms > self.request.deadline_ms


@dataclass(slots=True)
class SimulationConfig:
    """Everything needed to run one simulation.

    Attributes:
        n_parallel: Number of parallel workers (GPU replicas / slots).
        duration_ms: Length of the simulated window in milliseconds.
        workload: Arrival process to draw requests from.
        service_model: Model used to price each request.
        batching_mode: ``"static"`` (one request per worker) or
            ``"continuous"`` (a worker keeps a batch).
        max_batch_size: Maximum members per batch in continuous mode.
        batch_penalty: Per-extra-member slowdown of the concurrent decode phase.
        prefill_decode_share: Fraction of the decode rate that survives while a
            prefill is occupying the GPU (chunked prefill).
        speed_factor: Server speed multiplier. ``2.0`` means a GPU twice as
            fast as the baseline model.
        seed: Seed for every random draw in the run.
        queue_policy: Order in which queued requests are served.
        admission_policy: Admission rule set applied at the front door.
        high_water_mark: Queue depth above which load shedding starts.
        value_density_threshold: Minimum accepted value density
            (priority-weighted tokens per GPU-second) when the
            ``value_density`` admission policy is active.
    """

    n_parallel: int = 1
    duration_ms: float = 60_000.0
    workload: Workload = field(default_factory=lambda: PoissonArrivals(rate=1.0))
    service_model: ServiceTimeModel = field(default_factory=ServiceTimeModel)
    batching_mode: str = "static"
    max_batch_size: int = 8
    batch_penalty: float = 0.08
    prefill_decode_share: float = 0.35
    speed_factor: float = 1.0
    seed: int = 0
    queue_policy: QueuePolicy = QueuePolicy.FIFO
    admission_policy: AdmissionPolicy = AdmissionPolicy.ALL
    high_water_mark: int = 10_000
    value_density_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.n_parallel <= 0:
            raise ValueError(f"n_parallel must be positive, got {self.n_parallel}")
        if self.duration_ms <= 0:
            raise ValueError(f"duration_ms must be positive, got {self.duration_ms}")
        if self.batching_mode not in ("static", "continuous"):
            raise ValueError(
                f"batching_mode must be 'static' or 'continuous', got {self.batching_mode!r}"
            )
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        if self.batch_penalty < 0:
            raise ValueError(f"batch_penalty must be non-negative, got {self.batch_penalty}")
        if not 0.0 <= self.prefill_decode_share <= 1.0:
            raise ValueError(
                f"prefill_decode_share must be in [0, 1], got {self.prefill_decode_share}"
            )
        if self.speed_factor <= 0:
            raise ValueError(f"speed_factor must be positive, got {self.speed_factor}")
        if self.high_water_mark <= 0:
            raise ValueError(f"high_water_mark must be positive, got {self.high_water_mark}")

    @property
    def effective_max_batch_size(self) -> int:
        """Batch size actually used: ``1`` in static mode."""
        return 1 if self.batching_mode == "static" else self.max_batch_size


@dataclass(slots=True)
class SimulationResult:
    """Outcome of a simulation run.

    All latency figures are in milliseconds. Percentiles use linear
    interpolation between order statistics (``numpy.percentile`` defaults).
    """

    outcomes: list[RequestOutcome]
    duration_ms: float
    n_parallel: int
    batching_mode: str
    arrived: int
    completed: int
    dropped: int
    rejected: int
    shed: int
    still_queued: int
    queue_depth_max: int
    utilization: float
    busy_time_ms: float
    worker_busy_time_ms: list[float]
    throughput_rps: float
    wait_ms: LatencyStats
    ttft_ms: LatencyStats
    end_to_end_ms: LatencyStats
    deadline_miss_rate: float

    def outcome_for(self, request: RequestSpec) -> RequestOutcome:
        """Find the outcome belonging to ``request``.

        Raises:
            KeyError: If the request is not part of this run.
        """
        for outcome in self.outcomes:
            if outcome.request is request:
                return outcome
        raise KeyError("request not present in this simulation result")

    @property
    def served(self) -> list[RequestOutcome]:
        """Outcomes of requests that ran to completion."""
        return [o for o in self.outcomes if o.completion_time_ms is not None]

    @property
    def mean_requests_in_system(self) -> float:
        """Time-average number of requests in the system, ``L`` in Little's Law.

        Every request occupies the system for ``end_to_end_ms``; summing those
        intervals and dividing by the window length gives the average number of
        requests resident at any instant. Little's Law says this must equal
        ``throughput * mean_end_to_end_time``.
        """
        if self.duration_ms <= 0:
            return 0.0
        resident = sum(o.end_to_end_ms for o in self.served)
        return resident / self.duration_ms


def _stats(values: list[float]) -> LatencyStats:
    """Summarise a latency sample."""
    if not values:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    return LatencyStats(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
    )


@dataclass(slots=True)
class _Member:
    """A request that has been placed on a worker.

    The decode phase is tracked as *remaining work* plus the *rate* it was
    last computed at. Whenever the batch composition changes the rate changes
    too, so the remaining work is re-based onto the new rate.
    """

    member_id: int
    request: RequestSpec
    outcome: RequestOutcome | None
    arrival_time_ms: float
    prefill_work: float
    decode_work: float
    decode_remaining: float = 0.0
    decode_rate: float = 0.0
    decode_updated_at: float = 0.0
    prefill_done: bool = False


class _Worker:
    """One GPU worker holding a single batch of requests.

    In ``static`` mode the worker holds at most one member, so the prefill and
    the decode run back to back. In ``continuous`` mode the worker keeps a
    batch: a new member's prefill queues behind the running prefill, and the
    decoders share the GPU, each slowed down by ``batch_penalty`` per extra
    batch member.

    All times are absolute milliseconds on the simulation clock, so the worker
    never has to integrate a clock of its own.
    """

    def __init__(
        self,
        index: int,
        *,
        max_batch_size: int,
        batch_penalty: float,
        prefill_decode_share: float,
        window_end_ms: float,
    ) -> None:
        self.index = index
        self.max_batch_size = max_batch_size
        self.window_end_ms = window_end_ms
        self.batch_penalty = batch_penalty
        self.prefill_decode_share = prefill_decode_share
        self.members: dict[int, _Member] = {}
        self.prefill_queue: list[int] = []
        self.prefill_active: int | None = None
        self.prefill_deadline: float | None = None
        self.busy_start: float | None = None
        self.busy_total = 0.0
        # The last instant this worker was settled at. Events at or before it
        # have already fired, which keeps a zero-length decode from being
        # re-scheduled forever.
        self.settled_at = -1.0e18
        # Cached time of the next transition.
        self.next_at: float | None = None
        # Bumped every time the worker is re-armed, so stale timers in the
        # global event heap can be recognised and discarded.
        self.gen = 0

    # -- introspection ---------------------------------------------------

    @property
    def load(self) -> int:
        """Number of requests currently held by the worker."""
        return len(self.members)

    @property
    def decoding(self) -> int:
        """Number of members that are past their prefill phase."""
        return sum(1 for m in self.members.values() if m.prefill_done)

    @property
    def has_capacity(self) -> bool:
        """Whether another request can be added to this worker."""
        return len(self.members) < self.max_batch_size

    def decode_rate(self) -> float:
        """Current decode rate in work units per millisecond.

        A batch of ``n`` decoders shares the GPU, so each runs at
        ``1 / (1 + batch_penalty * (n - 1))`` of the solo rate. While a prefill
        occupies the GPU the decoders only keep ``prefill_decode_share`` of
        that rate (chunked-prefill behaviour).
        """
        n = self.decoding
        if n == 0:
            return 0.0
        rate = 1.0 / (1.0 + self.batch_penalty * (n - 1))
        if self.prefill_active is not None:
            rate *= self.prefill_decode_share
        return rate

    def busy_time(self, now: float) -> float:
        """Total time the worker has held at least one request.

        The window end clamps the measurement: a worker that is still busy when
        the simulation stops only counts up to the end of the window.
        """
        total = self.busy_total
        if self.busy_start is not None:
            total += min(now, self.window_end_ms) - self.busy_start
        return total

    # -- simulation ------------------------------------------------------

    def advance(self, now: float) -> None:
        """Account for the decode work done between the last update and ``now``."""
        for member in self.members.values():
            if not member.prefill_done or member.decode_rate <= 0.0:
                continue
            elapsed = now - member.decode_updated_at
            if elapsed > 0.0:
                member.decode_remaining -= member.decode_rate * elapsed
            member.decode_updated_at = now

    def _rebase(self, now: float) -> None:
        """Re-base the decoders onto the current decode rate."""
        rate = self.decode_rate()
        for member in self.members.values():
            if member.prefill_done:
                member.decode_rate = rate
                member.decode_updated_at = now

    def add(self, member: _Member, now: float) -> None:
        """Place ``member`` on this worker and queue its prefill."""
        self.advance(now)
        self.members[member.member_id] = member
        self.prefill_queue.append(member.member_id)
        if self.busy_start is None:
            self.busy_start = now
        self._start_next_prefill(now)
        self._rebase(now)

    def _start_next_prefill(self, now: float) -> None:
        if self.prefill_active is None and self.prefill_queue:
            member_id = self.prefill_queue.pop(0)
            member = self.members[member_id]
            self.prefill_active = member_id
            self.prefill_deadline = now + member.prefill_work

    def finish_prefill(self, member_id: int, now: float) -> _Member | None:
        """Complete the prefill of ``member_id``."""
        member = self.members.get(member_id)
        if member is None:
            return None
        self.advance(now)
        member.prefill_done = True
        member.decode_remaining = member.decode_work
        if self.prefill_active == member_id:
            self.prefill_active = None
            self.prefill_deadline = None
        self._rebase(now)
        self._start_next_prefill(now)
        self._rebase(now)
        return member

    def finish_decode(self, member_id: int, now: float) -> _Member | None:
        """Complete the decode of ``member_id`` and release its slot."""
        self.advance(now)
        member = self.members.pop(member_id, None)
        if member is None:
            return None
        if self.prefill_active == member_id:
            self.prefill_active = None
            self.prefill_deadline = None
        if not self.members and self.busy_start is not None:
            self.busy_total += min(now, self.window_end_ms) - self.busy_start
            self.busy_start = None
        self._rebase(now)
        return member

    def settle(self, now: float) -> list[tuple[str, int, _Member]]:
        """Advance the worker to ``now`` and report the transitions that fired.

        The worker never integrates a clock; it recomputes absolute deadlines
        from the current time whenever the batch composition changes.

        Returns:
            A list of ``(transition, member_id, member)`` tuples where
            ``transition`` is ``"prefill"`` or ``"decode"``.
        """
        fired: list[tuple[str, int, _Member]] = []
        self.settled_at = now
        self.advance(now)
        # Prefill first: it is a hard deadline and it changes the decode rate.
        if (
            self.prefill_active is not None
            and self.prefill_deadline is not None
            and now >= self.prefill_deadline - _EPS
        ):
            member_id = self.prefill_active
            finished = self.finish_prefill(member_id, now)
            if finished is not None:
                fired.append(("prefill", member_id, finished))
        # Then decodes.
        for member_id in list(self.members):
            member = self.members[member_id]
            if not member.prefill_done or member.decode_remaining > _EPS:
                continue
            done = self.finish_decode(member_id, now)
            if done is not None:
                fired.append(("decode", member_id, done))
        return fired

    def next_event_time(self) -> float | None:
        """Absolute time of this worker's next internal transition."""
        candidates: list[float] = []
        if self.prefill_deadline is not None and self.prefill_deadline > self.settled_at:
            candidates.append(self.prefill_deadline)
        for member in self.members.values():
            if member.prefill_done and member.decode_remaining >= 0.0 and member.decode_rate > 0.0:
                candidates.append(
                    member.decode_updated_at + member.decode_remaining / member.decode_rate
                )
        pending = [t for t in candidates if t > self.settled_at]
        self.next_at = min(pending) if pending else None
        return self.next_at


class Server:
    """A serving tier: ``n_parallel`` workers behind a single admission queue.

    This is the object you point at a real deployment. It knows how many GPUs
    (or how many KV-cache slots on a GPU) it has, and it models the queue that
    forms in front of them.

    Args:
        config: The simulation configuration.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._controller = AdmissionController(
            service_model=config.service_model,
            policy=config.admission_policy,
            queue_policy=config.queue_policy,
            high_water_mark=config.high_water_mark,
            value_density_threshold=config.value_density_threshold,
            workers=config.n_parallel,
            speed_factor=config.speed_factor,
        )

    @property
    def config(self) -> SimulationConfig:
        """The configuration this server was built with."""
        return self._config

    @property
    def controller(self) -> AdmissionController:
        """The admission controller in front of the workers."""
        return self._controller

    def run(self) -> SimulationResult:
        """Run the discrete-event simulation.

        The loop keeps a priority queue of *arrivals* and, between them, jumps
        to the earliest internal transition of any worker (a prefill finishing
        or a decode finishing). Nothing in between is simulated, so the cost of
        the run is proportional to the number of requests, not to the length of
        the simulated window: a 24-hour trace costs the same CPU time as a
        24-second one.

        Returns:
            A :class:`SimulationResult` with per-request outcomes and aggregates.
        """
        cfg = self._config
        model = cfg.service_model
        rng = np.random.default_rng(cfg.seed)

        workers = [
            _Worker(
                i,
                max_batch_size=cfg.effective_max_batch_size,
                batch_penalty=cfg.batch_penalty,
                prefill_decode_share=cfg.prefill_decode_share,
                window_end_ms=cfg.duration_ms,
            )
            for i in range(cfg.n_parallel)
        ]

        arrivals = cfg.workload.generate(cfg.duration_ms)
        # Event tuple: (time, kind, index). Arrivals live in the queue for the
        # whole run; worker transitions are scheduled one at a time.
        events: list[tuple[float, int, int]] = [
            (at, _ARRIVAL, idx) for idx, (at, _) in enumerate(arrivals)
        ]
        heapq.heapify(events)

        outcomes: list[RequestOutcome] = []
        queue_depth_max = 0
        next_member_id = 0
        now = 0.0

        def dispatch(now: float, touched: set[int] | None = None) -> None:
            """Move queued requests onto workers that still have capacity.

            Args:
                now: The current simulation time.
                touched: Optional set that collects the indexes of the workers
                    that received a request, so their timers can be re-armed.
            """
            nonlocal next_member_id
            while True:
                available = [w for w in workers if w.has_capacity]
                if not available:
                    return
                item = self._controller.dequeue()
                if item is None:
                    return
                # Spread the load: pick the least loaded worker.
                worker = min(available, key=lambda w: (w.load, w.index))
                prefill_ms, decode_ms = model.split(item.request, rng)
                member = _Member(
                    member_id=next_member_id,
                    request=item.request,
                    outcome=None,
                    arrival_time_ms=item.arrival_time_ms,
                    prefill_work=prefill_ms / cfg.speed_factor,
                    decode_work=decode_ms / cfg.speed_factor,
                )
                next_member_id += 1
                outcome = RequestOutcome(
                    request=item.request,
                    arrival_time_ms=item.arrival_time_ms,
                    start_time_ms=now,
                    decision=AdmissionDecision.ACCEPT,
                    reason="dispatched to worker",
                )
                member.outcome = outcome
                outcomes.append(outcome)
                worker.add(member, now)
                if touched is not None:
                    touched.add(worker.index)

        # Worker timers live in their own heap. Each entry carries the
        # worker's generation, so a timer that was superseded is discarded
        # when it surfaces instead of being processed twice.
        worker_events: list[tuple[float, int, int]] = []

        def schedule_worker(worker: _Worker) -> None:
            """Re-arm the timer of one worker."""
            nxt = worker.next_event_time()
            if nxt is None:
                return
            worker.gen += 1
            heapq.heappush(worker_events, (nxt, worker.gen, worker.index))

        def next_worker_event() -> tuple[float, int] | None:
            """The earliest internal transition of any worker."""
            while worker_events:
                t, gen, index = worker_events[0]
                if gen != workers[index].gen:
                    heapq.heappop(worker_events)
                    continue
                return t, index
            return None

        while True:
            # The next thing that happens is either the next arrival or the
            # next worker transition - whichever comes first. Events are
            # processed in strict chronological order, so the clock never runs
            # backwards.
            arrival_time = events[0][0] if events else float("inf")
            worker_event = next_worker_event()
            worker_time = worker_event[0] if worker_event is not None else float("inf")

            if min(arrival_time, worker_time) > cfg.duration_ms:
                break

            if worker_time < arrival_time:
                heapq.heappop(worker_events)
                now = worker_time
                worker = workers[worker_event[1]]  # type: ignore[index]
                touched: set[int] = set()
                for transition, _member_id, member in worker.settle(now):
                    outcome = member.outcome
                    if outcome is None:
                        continue
                    if transition == "prefill":
                        outcome.ttft_ms = now - outcome.arrival_time_ms
                    else:
                        start = outcome.start_time_ms if outcome.start_time_ms is not None else now
                        outcome.completion_time_ms = now
                        outcome.service_time_ms = now - start
                        outcome.end_to_end_ms = now - outcome.arrival_time_ms
                        if outcome.wait_time_ms is None:
                            outcome.wait_time_ms = start - outcome.arrival_time_ms
                dispatch(now, touched)
                schedule_worker(worker)
                continue

            _t, _kind, index = heapq.heappop(events)
            now = arrival_time
            request = arrivals[index][1]
            available = sum(1 for w in workers if w.has_capacity)
            admitted = self._controller.admit(request, now_ms=now, available_workers=available)
            if admitted.decision in (AdmissionDecision.REJECT, AdmissionDecision.SHED):
                outcomes.append(
                    RequestOutcome(
                        request=request,
                        arrival_time_ms=now,
                        decision=admitted.decision,
                        reason=admitted.reason,
                        dropped=True,
                        drop_reason=admitted.reason,
                    )
                )
            touched: set[int] = set()
            dispatch(now, touched)
            for index in touched:
                schedule_worker(workers[index])
            # The queue depth is the number of requests *waiting* for a
            # worker, so it is sampled once the arrival has been dispatched.
            queue_depth_max = max(queue_depth_max, len(self._controller))

        # Anything still in the queue never started.
        for item in self._controller.pending:
            outcomes.append(
                RequestOutcome(
                    request=item.request,
                    arrival_time_ms=item.arrival_time_ms,
                    decision=AdmissionDecision.QUEUE,
                    reason="still queued at end of simulation window",
                    dropped=True,
                    drop_reason="still queued at end of simulation window",
                )
            )

        busy_total = sum(w.busy_time(cfg.duration_ms) for w in workers)
        capacity_ms = cfg.n_parallel * cfg.duration_ms
        utilization = busy_total / capacity_ms if capacity_ms > 0 else 0.0

        completed = [o for o in outcomes if o.completion_time_ms is not None]
        waits = [o.wait_time_ms for o in completed if o.wait_time_ms is not None]
        ttfts = [o.ttft_ms for o in completed if o.ttft_ms is not None]
        e2es = [o.end_to_end_ms for o in completed if o.end_to_end_ms is not None]
        with_deadlines = [o for o in completed if o.request.deadline_ms is not None]
        misses = sum(1 for o in with_deadlines if o.deadline_missed)

        return SimulationResult(
            outcomes=outcomes,
            duration_ms=cfg.duration_ms,
            n_parallel=cfg.n_parallel,
            batching_mode=cfg.batching_mode,
            arrived=len(arrivals),
            completed=len(completed),
            dropped=sum(
                1 for o in outcomes if o.dropped and o.decision is not AdmissionDecision.QUEUE
            ),
            rejected=self._controller.rejected,
            shed=self._controller.shed,
            still_queued=len(self._controller),
            queue_depth_max=queue_depth_max,
            utilization=utilization,
            busy_time_ms=busy_total,
            worker_busy_time_ms=[w.busy_time(cfg.duration_ms) for w in workers],
            throughput_rps=len(completed) / (cfg.duration_ms / 1000.0),
            wait_ms=_stats(waits),
            ttft_ms=_stats(ttfts),
            end_to_end_ms=_stats(e2es),
            deadline_miss_rate=(misses / len(with_deadlines)) if with_deadlines else 0.0,
        )


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a simulation described by ``config``.

    Args:
        config: The simulation configuration.

    Returns:
        The :class:`SimulationResult` of the run.
    """
    return Server(config).run()
