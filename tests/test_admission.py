"""Tests for the admission controller, shedding and queue policies."""

from __future__ import annotations

import pytest

from queueguard import (
    AdmissionController,
    AdmissionDecision,
    AdmissionPolicy,
    QueuePolicy,
    RequestSpec,
    ServiceTimeModel,
    order_by_policy,
)
from queueguard.admission import QueuedRequest

MODEL = ServiceTimeModel(jitter_sigma=0.0)


def _queued(
    sequence: int,
    *,
    priority: int = 0,
    service_ms: float = 100.0,
    deadline_ms: float | None = None,
) -> QueuedRequest:
    return QueuedRequest(
        request=RequestSpec(
            input_tokens=100,
            requested_output_tokens=100,
            priority=priority,
            deadline_ms=deadline_ms,
        ),
        arrival_time_ms=float(sequence),
        sequence=sequence,
        estimated_service_ms=service_ms,
        value_density=1.0,
        deadline_ms=deadline_ms,
    )


def test_fifo_policy_preserves_arrival_order() -> None:
    items = [_queued(3), _queued(1), _queued(2)]
    ordered = order_by_policy(items, QueuePolicy.FIFO)
    assert [i.sequence for i in ordered] == [1, 2, 3]


def test_priority_policy_orders_by_priority_then_arrival() -> None:
    items = [
        _queued(1, priority=0),
        _queued(2, priority=5),
        _queued(3, priority=0),
        _queued(4, priority=5),
    ]
    ordered = order_by_policy(items, QueuePolicy.PRIORITY)
    assert [i.sequence for i in ordered] == [2, 4, 1, 3]


def test_sjf_policy_orders_by_estimated_service_time() -> None:
    items = [
        _queued(1, service_ms=500.0),
        _queued(2, service_ms=100.0),
        _queued(3, service_ms=250.0),
    ]
    ordered = order_by_policy(items, QueuePolicy.SJF)
    assert [i.sequence for i in ordered] == [2, 3, 1]


def test_edf_policy_orders_by_deadline() -> None:
    items = [
        _queued(1, deadline_ms=5000.0),
        _queued(2, deadline_ms=1000.0),
        _queued(3, deadline_ms=None),
        _queued(4, deadline_ms=2000.0),
    ]
    ordered = order_by_policy(items, QueuePolicy.EDF)
    assert [i.sequence for i in ordered] == [2, 4, 1, 3]


def test_policy_ties_keep_arrival_order() -> None:
    items = [_queued(2, priority=3), _queued(1, priority=3)]
    ordered = order_by_policy(items, QueuePolicy.PRIORITY)
    assert [i.sequence for i in ordered] == [1, 2]


def test_unknown_queue_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown queue policy"):
        order_by_policy([_queued(1)], "lifo")  # type: ignore[arg-type]


def test_queue_policy_accepts_string_names() -> None:
    assert QueuePolicy("fifo") is QueuePolicy.FIFO
    assert QueuePolicy("PRIORITY") is QueuePolicy.PRIORITY
    assert AdmissionPolicy("value_density") is AdmissionPolicy.VALUE_DENSITY


def test_a_free_worker_means_accept() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1)
    admitted = controller.admit(RequestSpec(), now_ms=0.0, available_workers=1)
    assert admitted.decision is AdmissionDecision.ACCEPT
    assert admitted.queue_position == 0


def test_no_free_worker_means_queue() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1)
    controller.admit(RequestSpec(), now_ms=0.0, available_workers=0)
    admitted = controller.admit(RequestSpec(), now_ms=1.0, available_workers=0)
    assert admitted.decision is AdmissionDecision.QUEUE
    assert admitted.queue_position == 1
    assert admitted.estimated_wait_ms > 0.0


def test_shedding_removes_the_lowest_priority_first() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=2)
    # Fill the queue with a high-priority and a mid-priority request.
    controller.admit(RequestSpec(priority=9), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=5), now_ms=1.0, available_workers=0)
    assert len(controller) == 2

    # The queue is at the high-water mark and the incoming request is the
    # weakest of the three, so it is the one that gets shed.
    admitted = controller.admit(RequestSpec(priority=0), now_ms=2.0, available_workers=0)
    assert admitted.decision is AdmissionDecision.SHED
    assert controller.shed == 1
    # Both valuable requests survived.
    assert sorted(q.priority for q in controller.pending) == [5, 9]


def test_a_stronger_request_evicts_the_weakest_queued_one() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=2)
    controller.admit(RequestSpec(priority=9), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=0), now_ms=1.0, available_workers=0)

    admitted = controller.admit(RequestSpec(priority=5), now_ms=2.0, available_workers=0)
    assert admitted.decision is AdmissionDecision.QUEUE
    assert controller.shed == 1
    # The low-priority request was evicted to make room.
    assert sorted(q.priority for q in controller.pending) == [5, 9]
    assert [q.priority for q in controller.evicted] == [0]


def test_high_priority_requests_are_never_shed_while_low_ones_remain() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=2)
    controller.admit(RequestSpec(priority=9), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=0), now_ms=1.0, available_workers=0)
    admitted = controller.admit(RequestSpec(priority=9), now_ms=2.0, available_workers=0)
    # The high-water mark is reached, so the weakest *queued* request (the
    # low-priority one) is evicted to make room. The high-priority requests
    # are never touched.
    assert admitted.decision is AdmissionDecision.QUEUE
    assert controller.shed == 1
    assert 0 not in [q.priority for q in controller.pending]
    assert [q.priority for q in controller.pending].count(9) == 2


def test_the_high_water_mark_is_respected() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=4)
    for i in range(10):
        controller.admit(RequestSpec(priority=0), now_ms=float(i), available_workers=0)
    # The queue never grows past the mark: each new arrival evicts one.
    assert len(controller) <= 4


def test_value_density_rejects_cheap_requests() -> None:
    controller = AdmissionController(
        service_model=MODEL,
        policy=AdmissionPolicy.VALUE_DENSITY,
        value_density_threshold=1e9,
        workers=1,
    )
    admitted = controller.admit(
        RequestSpec(input_tokens=100, requested_output_tokens=100),
        now_ms=0.0,
        available_workers=1,
    )
    assert admitted.decision is AdmissionDecision.REJECT
    assert "value density" in admitted.reason
    assert controller.rejected == 1


def test_value_density_accepts_valuable_requests() -> None:
    controller = AdmissionController(
        service_model=MODEL,
        policy=AdmissionPolicy.VALUE_DENSITY,
        value_density_threshold=1.0,
        workers=1,
    )
    request = RequestSpec(input_tokens=100, requested_output_tokens=100, priority=1)
    admitted = controller.admit(request, now_ms=0.0, available_workers=1)
    assert admitted.decision is AdmissionDecision.ACCEPT


def test_value_density_is_priority_weighted_tokens_per_gpu_second() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1)
    request = RequestSpec(input_tokens=100, requested_output_tokens=100, priority=1)
    service_s = controller.estimated_service_ms(request) / 1000.0
    expected = 2 * 200 / service_s
    assert controller.value_density(request) == pytest.approx(expected)


def test_dequeue_returns_the_next_request_in_policy_order() -> None:
    controller = AdmissionController(
        service_model=MODEL, workers=1, queue_policy=QueuePolicy.PRIORITY
    )
    controller.admit(RequestSpec(priority=1), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=7), now_ms=1.0, available_workers=0)
    first = controller.dequeue()
    assert first is not None
    assert first.priority == 7


def test_dequeue_on_an_empty_queue_returns_none() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1)
    assert controller.dequeue() is None


def test_controller_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="high_water_mark must be positive"):
        AdmissionController(high_water_mark=0)
    with pytest.raises(ValueError, match="workers must be positive"):
        AdmissionController(workers=0)
    with pytest.raises(ValueError, match="speed_factor must be positive"):
        AdmissionController(speed_factor=0.0)
    with pytest.raises(ValueError, match="value_density_threshold must be non-negative"):
        AdmissionController(value_density_threshold=-1.0)


def test_controller_counters_track_the_decisions() -> None:
    controller = AdmissionController(
        service_model=MODEL, policy=AdmissionPolicy.VALUE_DENSITY, value_density_threshold=1e9
    )
    request = RequestSpec(input_tokens=100, requested_output_tokens=100)
    controller.admit(request, now_ms=0.0, available_workers=1)
    controller.admit(request, now_ms=1.0, available_workers=1)
    assert controller.rejected == 2
    assert controller.accepted == 0
    assert controller.dropped == 2


def test_the_weakest_queued_request_is_tracked_as_the_queue_grows() -> None:
    """A weaker request admitted later must become the next shedding victim."""
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=3)
    controller.admit(RequestSpec(priority=5), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=2), now_ms=1.0, available_workers=0)
    # The weakest queued request so far is the priority-2 one.
    assert controller._weakest_queued() is not None
    assert controller._weakest_queued().priority == 2  # type: ignore[union-attr]

    # A weaker request arrives and becomes the new weakest.
    controller.admit(RequestSpec(priority=0), now_ms=2.0, available_workers=0)
    weakest = controller._weakest_queued()
    assert weakest is not None
    assert weakest.priority == 0


def test_the_weakest_cache_follows_removals() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1, high_water_mark=3)
    controller.admit(RequestSpec(priority=0), now_ms=0.0, available_workers=0)
    controller.admit(RequestSpec(priority=5), now_ms=1.0, available_workers=0)
    # The priority-0 request is served first, so the weakest becomes priority 5.
    first = controller.dequeue()
    assert first is not None
    assert first.priority == 0
    weakest = controller._weakest_queued()
    assert weakest is not None
    assert weakest.priority == 5


def test_queued_work_ms_tracks_the_backlog() -> None:
    controller = AdmissionController(service_model=MODEL, workers=1)
    request = RequestSpec(input_tokens=100, requested_output_tokens=100)
    assert controller.queued_work_ms == 0.0
    controller.admit(request, now_ms=0.0, available_workers=0)
    expected = controller.estimated_service_ms(request)
    assert controller.queued_work_ms == pytest.approx(expected)
    controller.dequeue()
    assert controller.queued_work_ms == pytest.approx(0.0)
