"""Admission control: what to serve, what to queue, and what to throw away.

An LLM inference endpoint is a *lossy* system. You cannot queue an unbounded
number of chat completions and expect them all to succeed: the queue latency
grows without limit and every user eventually times out. The correct behaviour
under overload is to **degrade gracefully** - keep the requests that matter,
shed the ones that do not, and never let the queue grow so deep that the
service becomes useless.

This module implements that decision:

* :class:`AdmissionPolicy` -- *value density*: is the business value of a
  request worth the GPU-seconds it will consume?
* :class:`QueuePolicy` -- once admitted, in what order do requests run?
* :class:`AdmissionController` -- the front door: ACCEPT / QUEUE / REJECT / SHED
  plus priority-aware load shedding above a high-water mark.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from enum import Enum

from queueguard.workload import RequestSpec, ServiceTimeModel

__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionResult",
    "AdmittedRequest",
    "QueuePolicy",
    "QueuedRequest",
    "order_by_policy",
]


class AdmissionDecision(str, Enum):
    """Outcome of an admission decision."""

    ACCEPT = "accept"
    """A worker is free: serve immediately."""

    QUEUE = "queue"
    """Admitted, but the request has to wait for a worker."""

    REJECT = "reject"
    """Refused outright: too expensive, or the queue is completely full."""

    SHED = "shed"
    """Dropped by load shedding to protect higher-priority traffic."""


class AdmissionPolicy(str, Enum):
    """Which admission rule set to apply."""

    ALL = "all"
    """Accept everything that fits in the queue."""

    VALUE_DENSITY = "value_density"
    """Reject requests whose value per GPU-second is below a threshold."""

    @classmethod
    def _missing_(cls, value: object) -> AdmissionPolicy | None:
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None


class QueuePolicy(str, Enum):
    """Ordering applied to the waiting queue."""

    FIFO = "fifo"
    """First in, first out."""

    PRIORITY = "priority"
    """Highest priority first, then FIFO within a priority level."""

    SJF = "sjf"
    """Shortest estimated job first, then FIFO within a tie."""

    EDF = "edf"
    """Earliest deadline first; requests without a deadline go last."""

    @classmethod
    def _missing_(cls, value: object) -> QueuePolicy | None:
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None


@dataclass(slots=True)
class AdmissionResult:
    """The decision taken for one request."""

    decision: AdmissionDecision
    reason: str
    queue_position: int = -1
    estimated_wait_ms: float = 0.0


@dataclass(slots=True)
class AdmittedRequest:
    """A request that went through the front door.

    Exposes the :class:`AdmissionResult` directly as attributes so callers can
    write ``admitted.decision`` without reaching through ``.result``.
    """

    request: RequestSpec
    arrival_time_ms: float
    result: AdmissionResult
    queued: QueuedRequest | None = None

    @property
    def decision(self) -> AdmissionDecision:
        """The decision that was taken."""
        return self.result.decision

    @property
    def reason(self) -> str:
        """Human-readable explanation of the decision."""
        return self.result.reason

    @property
    def queue_position(self) -> int:
        """Position in the service order, ``0`` means "next"; ``-1`` if refused."""
        return self.result.queue_position

    @property
    def estimated_wait_ms(self) -> float:
        """Estimated time before the request starts, in milliseconds."""
        return self.result.estimated_wait_ms


@dataclass(slots=True, eq=False)
class QueuedRequest:
    """A request waiting for a worker.

    Identity comparison is used deliberately: two queued requests with the same
    fields are still two different requests.
    """

    request: RequestSpec
    arrival_time_ms: float
    sequence: int
    estimated_service_ms: float
    value_density: float
    deadline_ms: float | None = None

    @property
    def priority(self) -> int:
        """Priority of the wrapped request."""
        return self.request.priority

    @property
    def strength(self) -> tuple[int, float]:
        """Ordering key used when choosing a shedding victim.

        Higher is more important: priority first, then value density. The
        comparison is the same one the controller uses to pick the weakest
        queued request, so the two are always consistent.
        """
        return (self.priority, self.value_density)


def _deadline_key(request: QueuedRequest) -> float:
    if request.deadline_ms is None:
        return math.inf
    return request.deadline_ms


def order_by_policy(
    requests: list[QueuedRequest], policy: QueuePolicy = QueuePolicy.FIFO
) -> list[QueuedRequest]:
    """Order ``requests`` according to ``policy``.

    Every policy is *stable*: requests that compare equal keep their arrival
    order, so the result is fully deterministic.

    Args:
        requests: The requests to order.
        policy: The ordering to apply.

    Returns:
        A new list holding the same requests in policy order.
    """
    if policy is QueuePolicy.FIFO:
        return sorted(requests, key=lambda r: r.sequence)
    if policy is QueuePolicy.PRIORITY:
        return sorted(requests, key=lambda r: (-r.priority, r.sequence))
    if policy is QueuePolicy.SJF:
        return sorted(requests, key=lambda r: (r.estimated_service_ms, r.sequence))
    if policy is QueuePolicy.EDF:
        return sorted(requests, key=lambda r: (_deadline_key(r), r.sequence))
    raise ValueError(f"unknown queue policy: {policy!r}")


@dataclass
class AdmissionController:
    """Front door for an inference endpoint.

    The controller owns the waiting queue. It accepts or refuses work, sheds
    low-value work when the queue crosses the high-water mark, and hands the
    next request to a free worker in the configured :class:`QueuePolicy` order.

    Args:
        service_model: Model used to estimate the cost of a request.
        policy: Admission rule set to apply.
        queue_policy: Order in which queued requests are served.
        high_water_mark: Queue depth above which load shedding kicks in.
        value_density_threshold: Minimum accepted value density, in
            priority-weighted tokens per GPU-second. ``0`` disables the check.
        workers: Number of workers sharing the queue, used to turn the amount
            of queued work into a wait-time estimate.
        speed_factor: Server speed multiplier applied to the cost estimate.
    """

    service_model: ServiceTimeModel = field(default_factory=ServiceTimeModel)
    policy: AdmissionPolicy = AdmissionPolicy.ALL
    queue_policy: QueuePolicy = QueuePolicy.FIFO
    high_water_mark: int = 100
    value_density_threshold: float = 0.0
    workers: int = 1
    speed_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.high_water_mark <= 0:
            raise ValueError(f"high_water_mark must be positive, got {self.high_water_mark}")
        if self.value_density_threshold < 0:
            raise ValueError(
                f"value_density_threshold must be non-negative, got {self.value_density_threshold}"
            )
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")
        if self.speed_factor <= 0:
            raise ValueError(f"speed_factor must be positive, got {self.speed_factor}")
        # The queue is kept sorted by the policy key, so admitting and
        # dequeuing are O(log n) / O(1) instead of a full re-sort.
        self._queue: list[tuple[tuple, int, QueuedRequest]] = []
        # The service time of every queued request, in the same order. Keeping
        # it in a parallel list lets the wait estimate use a C-level sum.
        self._work: list[float] = []
        self._total_work_ms = 0.0
        self._evicted: list[QueuedRequest] = []
        self._sequence = 0
        self._weakest: QueuedRequest | None = None
        self._live: set[int] = set()
        self._accepted = 0
        self._queued = 0
        self._rejected = 0
        self._shed = 0
        self._served = 0

    # -- introspection ---------------------------------------------------

    def __len__(self) -> int:
        """Number of requests currently waiting."""
        return len(self._queue)

    @property
    def pending(self) -> list[QueuedRequest]:
        """The waiting queue, in service order."""
        return [item for _key, _seq, item in self._queue]

    @property
    def accepted(self) -> int:
        """Number of requests admitted (accepted or queued)."""
        return self._accepted

    @property
    def queued(self) -> int:
        """Number of requests that were admitted but had to wait."""
        return self._queued

    @property
    def rejected(self) -> int:
        """Number of requests refused outright."""
        return self._rejected

    @property
    def shed(self) -> int:
        """Number of requests dropped by load shedding."""
        return self._shed

    @property
    def evicted(self) -> list[QueuedRequest]:
        """Queued requests that were shed to make room for a better one."""
        return list(self._evicted)

    @property
    def served(self) -> int:
        """Number of requests handed to a worker."""
        return self._served

    @property
    def queued_work_ms(self) -> float:
        """Total service time of the requests still waiting, in milliseconds.

        This is the workload the queue has to get through before the system is
        idle again.
        """
        return self._total_work_ms

    @property
    def dropped(self) -> int:
        """Total number of requests that never reached a worker."""
        return self._rejected + self._shed

    # -- admission -------------------------------------------------------

    def estimated_service_ms(self, request: RequestSpec) -> float:
        """Estimated GPU time the request will consume, in milliseconds."""
        return self.service_model.nominal_service_time_ms(request) / self.speed_factor

    def value_density(self, request: RequestSpec) -> float:
        """Business value per GPU-second for ``request``.

        Value is the priority-weighted token count of the request; the cost is
        the GPU-seconds it will occupy a worker. The unit is
        *priority-weighted tokens per GPU-second*.
        """
        service_s = self.estimated_service_ms(request) / 1000.0
        if service_s <= 0.0:
            return math.inf
        value = (request.priority + 1) * request.total_tokens
        return value / service_s

    def admit(
        self,
        request: RequestSpec,
        *,
        now_ms: float = 0.0,
        available_workers: int = 0,
    ) -> AdmittedRequest:
        """Decide what to do with ``request``.

        Args:
            request: The incoming request.
            now_ms: Current simulation time in milliseconds.
            available_workers: How many workers are idle right now.

        Returns:
            An :class:`AdmittedRequest` carrying the decision.
        """
        est_service_ms = self.estimated_service_ms(request)
        density = self.value_density(request)

        if (
            self.policy is AdmissionPolicy.VALUE_DENSITY
            and self.value_density_threshold > 0
            and density < self.value_density_threshold
        ):
            self._rejected += 1
            return self._result(
                request,
                now_ms,
                AdmissionDecision.REJECT,
                f"value density {density:.3g} tok/GPU-s below threshold "
                f"{self.value_density_threshold:.3g}",
            )

        if len(self._queue) >= self.high_water_mark:
            weakest = self._weakest_queued()
            if weakest is not None and (request.priority, density) <= weakest.strength:
                # The incoming request is worth less than everything already
                # queued, so it is the one that goes.
                self._shed += 1
                return self._result(
                    request,
                    now_ms,
                    AdmissionDecision.SHED,
                    f"queue at high-water mark ({self.high_water_mark}); incoming request is "
                    f"the lowest value in the system",
                )
            if weakest is not None:
                # The incoming request is worth more than the weakest queued
                # request, so the weakest one is evicted to make room.
                self._remove(weakest)
                self._live.discard(weakest.sequence)
                self._evicted.append(weakest)
                self._shed += 1

        self._sequence += 1
        queued = QueuedRequest(
            request=request,
            arrival_time_ms=now_ms,
            sequence=self._sequence,
            estimated_service_ms=est_service_ms,
            value_density=density,
            deadline_ms=request.deadline_ms,
        )
        key = self._order_key(queued)
        index = bisect.bisect_left(self._queue, (key, self._sequence, queued))
        self._queue.insert(index, (key, self._sequence, queued))
        self._work.insert(index, est_service_ms)
        self._live.add(self._sequence)
        self._accepted += 1
        self._total_work_ms += est_service_ms
        self._track_weakest(queued)

        # The request was inserted at `index`, so that is its queue position.
        position = index
        ahead_work_ms = sum(self._work[:position])
        estimated_wait_ms = ahead_work_ms / self.workers

        if available_workers > 0:
            decision = AdmissionDecision.ACCEPT
            reason = "worker available; admitted immediately"
        else:
            decision = AdmissionDecision.QUEUE
            reason = f"queued at position {position}; estimated wait {estimated_wait_ms:.1f}ms"
            self._queued += 1

        return self._result(
            request,
            now_ms,
            decision,
            reason,
            queue_position=position,
            estimated_wait_ms=estimated_wait_ms,
            queued=queued,
        )

    def dequeue(self) -> QueuedRequest | None:
        """Pop the next request to serve, or ``None`` if the queue is empty."""
        if not self._queue:
            return None
        _key, seq, chosen = self._queue.pop(0)
        del self._work[0]
        self._live.discard(seq)
        self._total_work_ms -= chosen.estimated_service_ms
        self._served += 1
        return chosen

    def _order_key(self, item: QueuedRequest) -> tuple:
        """The sort key of a queued request under the active policy."""
        policy = self.queue_policy
        if policy is QueuePolicy.PRIORITY:
            return (-item.priority, item.sequence)
        if policy is QueuePolicy.SJF:
            return (item.estimated_service_ms, item.sequence)
        if policy is QueuePolicy.EDF:
            return (_deadline_key(item), item.sequence)
        return (item.sequence,)

    def _remove(self, item: QueuedRequest) -> None:
        """Remove a queued request by identity."""
        for index, (_key, _seq, candidate) in enumerate(self._queue):
            if candidate is item:
                del self._queue[index]
                del self._work[index]
                self._total_work_ms -= item.estimated_service_ms
                return

    def _weakest_queued(self) -> QueuedRequest | None:
        """The queued request that is cheapest to throw away.

        The victim is the lowest priority, and among equal priorities the one
        with the lowest value density - the least valuable work per GPU-second.
        The result is cached because the scan runs on every admission; the
        cache is refreshed whenever the cached victim leaves the queue.
        """
        if not self._queue:
            return None
        if self._weakest is None or self._weakest.sequence not in self._live:
            self._weakest = min(
                (item for _key, _seq, item in self._queue),
                key=lambda item: (item.priority, item.value_density, -item.sequence),
            )
        return self._weakest

    def _track_weakest(self, item: QueuedRequest) -> None:
        """Keep the cached weakest victim up to date after an admission."""
        if self._weakest is None or self._weakest.sequence not in self._live:
            self._weakest = item
            return
        if (item.priority, item.value_density) < (
            self._weakest.priority,
            self._weakest.value_density,
        ):
            self._weakest = item

    def _result(
        self,
        request: RequestSpec,
        now_ms: float,
        decision: AdmissionDecision,
        reason: str,
        *,
        queue_position: int = -1,
        estimated_wait_ms: float = 0.0,
        queued: QueuedRequest | None = None,
    ) -> AdmittedRequest:
        return AdmittedRequest(
            request=request,
            arrival_time_ms=now_ms,
            result=AdmissionResult(
                decision=decision,
                reason=reason,
                queue_position=queue_position,
                estimated_wait_ms=estimated_wait_ms,
            ),
            queued=queued,
        )
