"""Workload and service-time models for LLM inference.

Two orthogonal concerns live here:

* **When** requests arrive (:class:`PoissonArrivals`, :class:`BurstyArrivals`,
  :class:`TraceReplay`).
* **How long** each request takes once it reaches a GPU
  (:class:`ServiceTimeModel`).

The service-time model is deliberately *not* a constant. Real LLM serving
decomposes a request into a **prefill** phase (ingesting the prompt, producing
the first token) and a **decode** phase (emitting the remaining tokens). The
prefill phase is what users perceive as *time to first token* (TTFT) and is
compute-bound on the prompt length; the decode phase is memory-bandwidth-bound
and scales with the number of tokens that have to be emitted.

Everything here is deterministic given a seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BurstyArrivals",
    "PoissonArrivals",
    "RequestFactory",
    "RequestSpec",
    "ServiceTimeModel",
    "TraceReplay",
    "Workload",
    "default_request_factory",
]

#: Callable that produces a :class:`RequestSpec` for one arrival.
RequestFactory = Callable[[np.random.Generator], "RequestSpec"]


@dataclass(slots=True)
class RequestSpec:
    """A single inference request.

    Attributes:
        input_tokens: Prompt length in tokens. Drives prefill cost.
        requested_output_tokens: Number of tokens the client asked for. Drives decode cost.
        priority: Relative importance, ``0`` (least) upwards. Higher wins ties and
            is shed last.
        tenant_id: Logical tenant the request belongs to.
        deadline_ms: Optional end-to-end deadline in milliseconds. Requests whose
            deadline is missed are counted in the deadline-miss rate.
    """

    input_tokens: int = 0
    requested_output_tokens: int = 0
    priority: int = 0
    tenant_id: str = "default"
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError(f"input_tokens must be non-negative, got {self.input_tokens}")
        if self.requested_output_tokens < 0:
            raise ValueError(
                f"requested_output_tokens must be non-negative, got {self.requested_output_tokens}"
            )
        if self.priority < 0:
            raise ValueError(f"priority must be non-negative, got {self.priority}")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be positive, got {self.deadline_ms}")

    @property
    def total_tokens(self) -> int:
        """Prompt plus generated tokens."""
        return self.input_tokens + self.requested_output_tokens


class ServiceTimeModel:
    """Maps a request to an expected service time and a TTFT/total split.

    The model is affine in the token counts::

        prefill_ms = base_ms + per_input_token_ms * input_tokens
        decode_ms  = per_output_token_ms * output_tokens
        total_ms   = (prefill_ms + decode_ms) * jitter

    where ``jitter`` is a multiplicative lognormal draw from a seeded RNG
    (median ``1.0``, so the model is unbiased but heavy-tailed). The TTFT is the
    prefill phase; the remainder of the total is decode.

    Args:
        base_ms: Fixed per-request overhead (kernel launch, HTTP, scheduling).
        per_input_token_ms: Prefill cost per prompt token.
        per_output_token_ms: Decode cost per generated token.
        jitter_sigma: Standard deviation of the lognormal jitter. ``0`` disables it.
        seed: Seed for the jitter RNG.
    """

    def __init__(
        self,
        base_ms: float = 20.0,
        per_input_token_ms: float = 0.6,
        per_output_token_ms: float = 6.0,
        *,
        jitter_sigma: float = 0.18,
        seed: int = 0,
    ) -> None:
        if base_ms < 0:
            raise ValueError(f"base_ms must be non-negative, got {base_ms}")
        if per_input_token_ms < 0:
            raise ValueError(f"per_input_token_ms must be non-negative, got {per_input_token_ms}")
        if per_output_token_ms < 0:
            raise ValueError(f"per_output_token_ms must be non-negative, got {per_output_token_ms}")
        if jitter_sigma < 0:
            raise ValueError(f"jitter_sigma must be non-negative, got {jitter_sigma}")
        self._base_ms = float(base_ms)
        self._per_input_token_ms = float(per_input_token_ms)
        self._per_output_token_ms = float(per_output_token_ms)
        self._jitter_sigma = float(jitter_sigma)
        self._seed = int(seed)

    @property
    def base_ms(self) -> float:
        """Fixed per-request overhead in milliseconds."""
        return self._base_ms

    @property
    def per_input_token_ms(self) -> float:
        """Prefill cost per prompt token in milliseconds."""
        return self._per_input_token_ms

    @property
    def per_output_token_ms(self) -> float:
        """Decode cost per generated token in milliseconds."""
        return self._per_output_token_ms

    @property
    def jitter_sigma(self) -> float:
        """Standard deviation of the multiplicative lognormal jitter."""
        return self._jitter_sigma

    @property
    def seed(self) -> int:
        """Seed of the jitter RNG."""
        return self._seed

    def nominal_service_time_ms(self, request: RequestSpec) -> float:
        """Expected service time *without* jitter, in milliseconds."""
        return (
            self._base_ms
            + self._per_input_token_ms * request.input_tokens
            + self._per_output_token_ms * request.requested_output_tokens
        )

    def prefill_ms(self, request: RequestSpec) -> float:
        """Nominal prefill (TTFT) time in milliseconds."""
        return self._base_ms + self._per_input_token_ms * request.input_tokens

    def decode_ms(self, request: RequestSpec) -> float:
        """Nominal decode time in milliseconds."""
        return self._per_output_token_ms * request.requested_output_tokens

    def service_time_ms(
        self, request: RequestSpec, rng: np.random.Generator | None = None
    ) -> float:
        """Service time for ``request`` in milliseconds, including jitter.

        Args:
            request: The request to price.
            rng: Optional RNG to draw the jitter from. When ``None`` a fresh
                generator seeded from the model's seed is used, which keeps the
                call deterministic but *not* correlated with earlier draws.

        Returns:
            The total service time in milliseconds.
        """
        return self.nominal_service_time_ms(request) * self.jitter_factor(rng)

    def jitter_factor(self, rng: np.random.Generator | None = None) -> float:
        """Multiplicative jitter drawn from the model, mean ``1.0``.

        The distribution is lognormal with median ``1.0``, so the model is
        unbiased on average but heavy tailed: some requests take twice as long
        as the nominal prediction, which is exactly what produces a p99 tail.
        """
        if self._jitter_sigma == 0.0:
            return 1.0
        generator = self._rng() if rng is None else rng
        return float(generator.lognormal(mean=0.0, sigma=self._jitter_sigma))

    def split(
        self, request: RequestSpec, rng: np.random.Generator | None = None
    ) -> tuple[float, float]:
        """Split the service time into ``(prefill_ms, decode_ms)``, with jitter.

        The prefill phase is the whole of the TTFT; the decode phase produces
        the remaining tokens. A single jitter factor is applied to both so the
        TTFT-to-total ratio is preserved.

        Args:
            request: The request to price.
            rng: Optional RNG to draw the jitter from.

        Returns:
            A ``(prefill_ms, decode_ms)`` tuple in milliseconds.
        """
        jitter = self.jitter_factor(rng)
        return self.prefill_ms(request) * jitter, self.decode_ms(request) * jitter

    def ttft_fraction(self, request: RequestSpec) -> float:
        """Fraction of the total service time spent before the first token.

        The prefill phase is the whole of the TTFT; decode is the rest.
        """
        nominal = self.nominal_service_time_ms(request)
        if nominal <= 0.0:
            return 0.0
        return self.prefill_ms(request) / nominal

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed)


class Workload:
    """Base class for arrival processes.

    Subclasses implement :meth:`generate`, which returns the arrival times and
    request specifications sorted by time.
    """

    def generate(self, duration_ms: float) -> list[tuple[float, RequestSpec]]:
        """Generate arrivals over ``duration_ms`` of simulated time.

        Args:
            duration_ms: Length of the observation window in milliseconds.

        Returns:
            A list of ``(arrival_time_ms, request)`` tuples ordered by arrival time.
        """
        raise NotImplementedError

    def __call__(self, duration_ms: float) -> list[tuple[float, RequestSpec]]:
        """Alias for :meth:`generate`."""
        return self.generate(duration_ms)


class PoissonArrivals(Workload):
    """Homogeneous Poisson arrivals: exponential inter-arrival times.

    The mean inter-arrival time is ``1000 / rate`` milliseconds, so the offered
    load is exactly ``rate`` requests per second. This is the textbook M/M/c
    workload and the right baseline for capacity planning.

    Args:
        rate: Arrival rate in requests per second.
        seed: Seed for the inter-arrival RNG.
        request_factory: Callable producing the :class:`RequestSpec` for each
            arrival. Defaults to a fixed-size chat request.
    """

    def __init__(
        self,
        rate: float,
        *,
        seed: int = 0,
        request_factory: RequestFactory | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self._rate = float(rate)
        self._seed = int(seed)
        self._request_factory = request_factory or default_request_factory

    @property
    def rate(self) -> float:
        """Arrival rate in requests per second."""
        return self._rate

    @property
    def seed(self) -> int:
        """Seed of the inter-arrival RNG."""
        return self._seed

    def generate(self, duration_ms: float) -> list[tuple[float, RequestSpec]]:
        """Generate Poisson arrivals covering ``duration_ms``."""
        if duration_ms <= 0:
            return []
        rng = np.random.default_rng(self._seed)
        mean_gap_ms = 1000.0 / self._rate
        arrivals: list[tuple[float, RequestSpec]] = []
        t = 0.0
        while True:
            t += float(rng.exponential(mean_gap_ms))
            if t > duration_ms:
                break
            arrivals.append((t, self._request_factory(rng)))
        return arrivals


class BurstyArrivals(Workload):
    """Bursty arrivals: alternating busy and idle periods.

    A burst of ``burst_size`` requests is emitted back-to-back (spread across
    the active window of the period), then the process idles for the remainder
    of the period. This is the closest thing to a diurnal promo spike you can
    express with three numbers.

    Args:
        rate: Average arrival rate in requests per second, used to derive the
            burst period.
        burst_size: Number of requests emitted per burst.
        duty_cycle: Fraction of each period spent *active*; ``1.0`` is a
            continuous stream, ``0.5`` alternates busy/idle.
        seed: Seed for the RNG.
        request_factory: Callable producing the :class:`RequestSpec` per arrival.
    """

    def __init__(
        self,
        rate: float,
        burst_size: int = 20,
        duty_cycle: float = 0.5,
        *,
        seed: int = 0,
        request_factory: RequestFactory | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst_size <= 0:
            raise ValueError(f"burst_size must be positive, got {burst_size}")
        if not 0.0 < duty_cycle <= 1.0:
            raise ValueError(f"duty_cycle must be in (0, 1], got {duty_cycle}")
        self._rate = float(rate)
        self._burst_size = int(burst_size)
        self._duty_cycle = float(duty_cycle)
        self._seed = int(seed)
        self._request_factory = request_factory or default_request_factory

    @property
    def rate(self) -> float:
        """Average arrival rate in requests per second."""
        return self._rate

    @property
    def burst_size(self) -> int:
        """Number of requests emitted per burst."""
        return self._burst_size

    @property
    def duty_cycle(self) -> float:
        """Fraction of each period during which requests are emitted."""
        return self._duty_cycle

    @property
    def seed(self) -> int:
        """Seed of the RNG."""
        return self._seed

    def generate(self, duration_ms: float) -> list[tuple[float, RequestSpec]]:
        """Generate bursty arrivals covering ``duration_ms``."""
        if duration_ms <= 0:
            return []
        rng = np.random.default_rng(self._seed)
        # Requests per period = rate * period, so a period long enough to carry
        # `burst_size` requests is burst_size / rate seconds.
        burst_period_ms = (self._burst_size / self._rate) * 1000.0
        active_ms = burst_period_ms * self._duty_cycle
        idle_ms = burst_period_ms - active_ms
        # Inside a burst, requests are spread evenly across the active window.
        gap_ms = active_ms / self._burst_size if self._burst_size else 0.0

        arrivals: list[tuple[float, RequestSpec]] = []
        t = 0.0
        while t <= duration_ms:
            for i in range(self._burst_size):
                at = t + i * gap_ms
                if at > duration_ms:
                    return arrivals
                arrivals.append((at, self._request_factory(rng)))
            t += active_ms + idle_ms
        return arrivals


class TraceReplay(Workload):
    """Replay a recorded arrival trace.

    The trace is a list of ``(time_ms, request)`` tuples. Requests are replayed
    verbatim, which makes this the right workload for "what happened at
    09:00 yesterday" questions.

    Args:
        requests: The recorded ``(time_ms, request)`` pairs.
        scale: Multiplier applied to the replayed request *sizes* (token counts).
            ``1.0`` replays the trace as recorded.
    """

    def __init__(
        self,
        requests: list[tuple[float, RequestSpec]],
        *,
        scale: float = 1.0,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._requests = list(requests)
        self._scale = float(scale)

    @property
    def requests(self) -> list[tuple[float, RequestSpec]]:
        """The recorded trace."""
        return list(self._requests)

    @property
    def scale(self) -> float:
        """Multiplier applied to replayed request sizes."""
        return self._scale

    def generate(self, duration_ms: float) -> list[tuple[float, RequestSpec]]:
        """Replay the trace, clipped to ``duration_ms``."""
        if duration_ms <= 0:
            return []
        out: list[tuple[float, RequestSpec]] = []
        for t, req in self._requests:
            if t > duration_ms:
                break
            if self._scale == 1.0:
                out.append((float(t), req))
            else:
                out.append(
                    (
                        float(t),
                        RequestSpec(
                            input_tokens=int(req.input_tokens * self._scale),
                            requested_output_tokens=int(req.requested_output_tokens * self._scale),
                            priority=req.priority,
                            tenant_id=req.tenant_id,
                            deadline_ms=req.deadline_ms,
                        ),
                    )
                )
        return out


def default_request_factory(rng: np.random.Generator) -> RequestSpec:
    """Produce a representative chat request with mild size variation.

    The prompt length is drawn from a lognormal (long-tail) distribution clipped
    to a sane range, and the reply length from a uniform range. The resulting
    mix has a median prompt of roughly 500 tokens and a median reply of roughly
    250, which is what a typical RAG-augmented chat workload looks like.
    """
    input_tokens = int(np.clip(rng.lognormal(mean=6.2, sigma=0.55), 32, 8192))
    output_tokens = int(np.clip(rng.integers(32, 512), 1, 4096))
    return RequestSpec(input_tokens=input_tokens, requested_output_tokens=output_tokens)
