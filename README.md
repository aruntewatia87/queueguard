# queueguard

> Queueing-theory simulator and admission controller for LLM inference: backpressure, priority scheduling, capacity planning and tail-latency prediction.

[![CI](https://github.com/aruntewatia87/queueguard/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/queueguard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Every AI Cloud Solutions Architect eventually has to answer the same question out loud in front of a finance team: **"how many GPUs do I need to serve 500 requests per second at a p99 time-to-first-token under two seconds, and what happens if traffic triples tomorrow?"** Today the honest answer is "run a load test and hope". The tooling that exists is all about CPU and memory: Kubernetes HPA charts that scale a Deployment because the pod is at 80% CPU. That tooling has no idea that an LLM request's service time depends on its prompt length, that users perceive *time to first token* rather than total latency, that a 10x traffic spike must **not** be met with 10x replicas because loading a model onto a GPU takes minutes, and that the right response to overload is to *shed* work rather than to queue it until every user times out.

`queueguard` is an offline, deterministic simulator for exactly this problem. It models your workload, tells you the fleet size you need, predicts the tail latency, shows you where the SLO breaks, and gives you an admission controller that degrades gracefully instead of melting down. No cluster, no GPU, no API keys - just numpy and the laws of queueing theory.

---

## ✨ Highlights
- **Discrete-event simulation** of an LLM inference tier, fast enough to replay a 24-hour trace in milliseconds of CPU time.
- **Real service-time model** that splits every request into *prefill* (the TTFT users actually feel) and *decode* (where the tokens come from), with a heavy-tailed lognormal jitter.
- **Two serving regimes**: `static` (one request per worker) and `continuous` (a worker keeps a batch, and the batch's decoders share the GPU).
- **Admission control** with four decisions - ACCEPT / QUEUE / REJECT / SHED - priority-aware load shedding above a high-water mark, and a *value density* policy that weighs business value against GPU-seconds.
- **Four queue policies**: FIFO, Priority, Shortest-Job-First and Deadline-Aware (earliest deadline first).
- **Capacity planner** that binary-searches the *minimum* number of workers meeting a throughput + p99 TTFT SLO, and proves it by showing that one fewer worker fails.
- **Spike analysis** - the "Black Friday" question: simulate a 3x traffic jump and report exactly where the SLO breaks and what scale factor fixes it.
- **The GPU-warmup caveat**, which is the nuance that separates a real capacity plan from a spreadsheet: a scale-up is useless until the model is loaded, so autoscaling cannot rescue a short spike.
- **Zero dependencies** beyond numpy. Deterministic: the same seed always gives the same answer.

---

## 🚀 Quickstart
```bash
uv sync --all-extras --dev
uv run pytest -q
```

---

## 🔧 API Tour

**Simulate a workload and read the latency percentiles.**

```python
from queueguard import BurstyArrivals, ServiceTimeModel, SimulationConfig, simulate

service = ServiceTimeModel(
    base_ms=20.0,  # fixed per-request overhead
    per_input_token_ms=0.6,  # prefill cost per prompt token
    per_output_token_ms=6.0,  # decode cost per generated token
    jitter_sigma=0.2,  # heavy-tailed service-time variation
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

print(result.throughput_rps)  # 6.533333333333333
print(result.ttft_ms.p99)  # 7739.89725550218
print(result.utilization)  # 0.9984375
print(result.queue_depth_max)  # 63
```

**Admit requests with priority-aware load shedding.**

```python
from queueguard import AdmissionController, AdmissionPolicy, QueuePolicy, RequestSpec

controller = AdmissionController(
    service_model=service,
    policy=AdmissionPolicy.VALUE_DENSITY,
    queue_policy=QueuePolicy.PRIORITY,
    high_water_mark=6,
    workers=1,
)

admitted = controller.admit(
    RequestSpec(input_tokens=400, requested_output_tokens=200, priority=0),
    now_ms=0.0,
    available_workers=0,
)
print(admitted.decision)  # AdmissionDecision.QUEUE
print(admitted.queue_position)  # 0 - this request is next in line
print(admitted.estimated_wait_ms)  # estimated wait before the request starts
print(controller.shed)  # how many low-value requests were dropped
```

**Plan the fleet, then stress it with a spike.**

```python
from queueguard import CapacityPlanner

planner = CapacityPlanner(
    target_throughput_rps=60.0,
    ttft_slo_ms=2000.0,
    duration_ms=60_000,
    service_model=service,
    batching_mode="continuous",
    max_batch_size=8,
    max_workers=256,
    seed=23,
    cost_per_worker_hour=1.75,
)

plan = planner.plan()
print(plan.required_workers)  # 36
print(plan.cost_per_million_requests)  # 312.0433...
print(plan.scaling_recommendation)

spike = planner.analyze_spike(plan, 3.0)
print(spike.slo_broken)  # True
print(spike.workers_needed)  # 103

replicas, time_to_scale, why = planner.autoscale_recommendation(plan)
print(replicas, time_to_scale)  # 52 210s to become useful (model load 180s)
print(why)
```

**Replay a recorded trace.**

```python
from queueguard import RequestSpec, TraceReplay

trace = [(0.0, RequestSpec(input_tokens=512, requested_output_tokens=128))]
workload = TraceReplay(trace, scale=2.0)  # double every request size
```

---

## 🧠 How it works

### The service time is not a constant

An LLM request is not one operation, it is two, and they behave completely differently:

```
prefill_ms = base_ms + per_input_token_ms * input_tokens      # -> this IS the TTFT
decode_ms  = per_output_token_ms * requested_output_tokens     # -> the rest of the latency
total_ms   = (prefill_ms + decode_ms) * jitter
```

The prefill is **compute bound** on the prompt length and is the whole of the time-to-first-token the user perceives. The decode phase is **memory-bandwidth bound** and scales with the number of tokens to emit. `queueguard` models the split explicitly, which is why it can answer "what is my p99 TTFT" - a question a single-number queueing model cannot answer.

The jitter is a multiplicative lognormal with median `1.0`, so the model is unbiased on average but heavy tailed. That tail is where p99 comes from: a small number of requests take twice as long as predicted, and those are exactly the ones that break an SLO.

### The discrete-event loop

The simulator is a classic discrete-event simulation. Events live in a priority queue and the clock jumps to the earliest one; nothing in between is simulated:

```python
while there_is_an_event:
    now = min(next_arrival, next_worker_transition)
    advance_the_workers_to(now)
    apply_the_event(now)  # arrival, or prefill/decode finished
```

A worker is a single batch. In `static` mode it holds at most one request; in `continuous` mode it holds up to `max_batch_size`. When a prefill is running, the decoders in the same batch only keep `prefill_decode_share` of their rate - chunked-prefill behaviour, which is what really happens on a real engine. When `n` decoders share a GPU, each runs at

```
rate = 1 / (1 + batch_penalty * (n - 1))
```

Because the only events are arrivals and worker transitions, the cost of a run is proportional to the number of requests, not to the length of the simulated window. A 24-hour trace costs about the same CPU time as a 24-second one.

### Little's Law is the sanity check

The queueing theory that ties it all together is **Little's Law**:

```
L = lambda * W
```

the average number of requests in the system equals the arrival rate times the average time each request spends there. `queueguard` reports both sides so you can check the model against theory: `throughput_rps` is `lambda`, `wait_ms.mean` is `W` for the queue, and `mean_requests_in_system` is `L` measured directly by summing the residence intervals. If the two disagree, the model is wrong.

Little's Law also explains *why* the tail matters. A fleet sized so that the **mean** wait is acceptable will still fail its p99 SLO, because at 99% utilisation the queue length is governed by the variance of the arrival process, not its mean. The planner therefore searches on the **p99**, never on the average.

### Why binary search finds the minimum

The relationship between worker count and tail latency is monotone in practice - more workers means shorter queues means a smaller p99 - but it is not linear, so an analytic formula does not exist. `CapacityPlanner.plan()` therefore *searches*: simulate at `n` workers, check the SLO, and bisect. The result is the **minimum** fleet that meets the target, and the test suite asserts the crucial property: **one fewer worker fails the SLO**. A plan that merely suffices is worthless, because you would be paying for GPUs you do not need.

### The GPU-warmup caveat

The standard autoscaling formula, used by every HPA in production, is

```
desired_replicas = ceil(current_replicas * current_utilization / target_utilization)
```

`queueguard` applies it - and then tells you why it is not enough. A new GPU replica is not a new CPU: it has to load the model weights into device memory first. `model_load_ms` defaults to 180 seconds, and until that finishes the replica serves nothing. So:

> **A scale-up is useless for a spike shorter than the model load time.** Autoscaling can rescue a sustained load increase. It cannot rescue a three-minute promo.

`autoscale_recommendation` returns the replica count, a `TimeToScale` describing how long the new capacity takes to become useful, and an explicit statement of whether the scale-up helps at all within the spike window. When it does not, the recommendation is to *pre-scale* or to *shed* - not to wait for the HPA.

### Admission control is a loss problem

An inference endpoint is a **lossy** system. You cannot queue an unbounded number of chat completions and expect them all to succeed: the queue latency grows without limit and every user eventually times out. The correct behaviour under overload is to degrade gracefully.

- **High-water mark.** Above a queue-depth limit, new arrivals are compared against the weakest queued request. If the newcomer is worth less, *it* is shed; if it is worth more, the weakest queued request is evicted to make room. Priority first, then value density.
- **Value density** is business value per GPU-second: `(priority + 1) * total_tokens / service_seconds`. A request that will occupy a GPU for thirty seconds to produce forty low-priority tokens has a terrible value density, and the `value_density` policy refuses it outright.
- **Queue policies** decide who goes next: FIFO, Priority (highest first), SJF (shortest job first - optimal for minimising mean wait) or EDF (earliest deadline first - what a deadline-aware API needs).

---

## 📊 Example output

Verbatim output of `uv run python examples/quickstart.py`:

```text
1. Bursty workload against a 4-worker server
--------------------------------------------
  arrived           481
  completed         392
  dropped             0
  queue depth        63 (max)
  utilisation     99.8%
  throughput       6.53 rps

  latency percentiles (ms)
                     p50       p95       p99       max
    wait            3177      7028      7384      7582
    TTFT            3565      7397      7740      8134
    end-to-end      7767     12844     14320     15656

2. Priority-aware load shedding
-------------------------------
  high-water mark = 6, incoming priorities: 0 1 0 5 0 9 0 2 0 7
    priority 0 -> queue    
    priority 1 -> queue    
    priority 0 -> queue    
    priority 5 -> queue    
    priority 0 -> queue    
    priority 9 -> queue    
    priority 0 -> shed    x
    priority 2 -> queue    
    priority 0 -> shed    x
    priority 7 -> queue    
  shed: 4   still queued: 6
  survivors: [0, 0, 1, 5, 7, 9]
  evicted:   [0, 2]

3. Capacity plan: 60 rps at p99 TTFT < 2000 ms
----------------------------------------------
  required workers   36
  achieved           56.1 rps
  p99 TTFT           1931 ms (budget 2000 ms)
  p99 end-to-end     10499 ms
  utilisation        99.3%
  headroom           1%
  cost / 1M requests $312.04
  advice: scale 36->37 workers now: only 1% headroom left, the p99 TTFT SLO breaks on the next traffic step

4. Three-fold traffic spike
---------------------------
  offered load       180 rps
  SLO broken         True
  p99 TTFT           37087 ms
  utilisation        99.8%
  workers needed     103
  breaks above       1.00x traffic
  advice: 3x traffic (180 rps) breaks the p99 TTFT SLO at 36 workers (p99 37087ms). Pre-scale to 103 workers, and expect the fleet to be saturated (100% utilisation) - shed low-priority traffic rather than queueing it. Note: the fleet is sized to the SLO boundary, so it already breaks at 1.00x traffic - it has no headroom at all.

5. Can autoscaling save us?
---------------------------
  recommended replicas 52
  time to scale        210s to become useful (model load 180s)
  useful for the spike True
  advice: autoscaler would scale 36->52 replicas at 70% target utilisation. The new replica becomes useful after 210s (model load 180s), so pre-warm the pool for spikes shorter than 300s.

6. Little's Law check (L = lambda * W)
--------------------------------------
  throughput (lambda)  56.08 rps
  mean wait (W)        0.152 s
  mean end-to-end      4.523 s
  L = lambda * W       8.5 requests waiting
  L = lambda * E2E     253.7 requests in the system
  queue depth (max)    41
```

---

## 🧪 Testing
```bash
uv sync --all-extras --dev
uv run pytest -q --cov=queueguard --cov-report=term-missing
uv run ruff check .
```

---

## 📄 License
MIT © 2026 Arun Tewatia
