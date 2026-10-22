# queueguard architecture

Four modules, each with one job. The dependency graph is strictly layered -
nothing below imports anything above it.

```
workload.py     service-time model + arrival processes
     |
     v
admission.py    front door: ACCEPT / QUEUE / REJECT / SHED, queue ordering
     |
     v
simulator.py    discrete-event engine, workers, latency accounting
     |
     v
capacity.py     binary search for the minimum fleet, spike analysis, autoscale advice
```

`__init__.py` re-exports the public surface and nothing else.

## workload.py

- `RequestSpec` - one request: prompt tokens, requested output tokens, priority,
  tenant, optional deadline.
- `ServiceTimeModel` - splits a request into **prefill** (the TTFT the user
  perceives) and **decode** (where the tokens come from), then applies a
  multiplicative lognormal jitter. The jitter has median `1.0`, so the model is
  unbiased on average but heavy tailed - which is where the p99 comes from.
- `PoissonArrivals` / `BurstyArrivals` / `TraceReplay` - three arrival
  processes, each with its own seeded RNG.

## admission.py

The controller owns the waiting queue. It keeps the queue **sorted by the
active policy key** so admitting is `O(log n)` and dequeuing is `O(1)`, rather
than re-sorting on every arrival.

- **Load shedding.** Above `high_water_mark`, the incoming request is compared
  against the weakest queued request (lowest priority, then lowest value
  density). If the newcomer is worth less, *it* is shed; if it is worth more,
  the weakest queued request is evicted to make room. The weakest victim is
  cached, because the scan runs on every admission.
- **Value density.** `(priority + 1) * total_tokens / service_seconds` -
  business value per GPU-second. The `value_density` policy refuses requests
  below a threshold outright.
- **Queue policies.** FIFO, Priority, Shortest-Job-First, Deadline-Aware.

## simulator.py

The engine is a discrete-event simulation.

```
while there_is_an_event:
    now = min(next_arrival, next_worker_transition)
    advance_the_workers_to(now)
    apply_the_event(now)
```

Two heaps are involved: one for arrivals, one for worker timers. Each worker
timer carries the worker's generation counter, so a superseded timer is
discarded when it surfaces instead of being processed twice.

A worker is a single batch:

- `static` mode: at most one request, so prefill and decode run back to back.
- `continuous` mode: up to `max_batch_size` requests. A new member's prefill
  queues behind the running prefill, and while a prefill holds the GPU the
  decoders keep only `prefill_decode_share` of their rate. `n` concurrent
  decoders each run at `1 / (1 + batch_penalty * (n - 1))` of the solo rate.

The decode phase is tracked as *remaining work* plus the *rate* it was last
computed at, so a change in batch composition re-bases the remaining work
instead of integrating a clock.

## capacity.py

- `plan()` - binary search over worker counts. The relationship between
  workers and tail latency is monotone but not linear, so no closed form
  exists. The result is the **minimum** fleet that meets the SLO; the test
  suite asserts that one fewer worker fails.
- `analyze_spike()` - re-runs the plan at a multiple of the baseline traffic
  and reports where the SLO breaks, how many workers the spike needs, and the
  largest multiplier the baseline fleet survives.
- `autoscale_recommendation()` - applies the standard HPA formula and then
  adds the caveat that matters for GPUs: a new replica has to load the model
  first, so a scale-up cannot rescue a spike shorter than the model load time.
