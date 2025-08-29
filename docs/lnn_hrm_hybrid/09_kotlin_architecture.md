# Kotlin Port: Actors, Channels, and Virtual Dendrites

We sketch a port to Kotlin with an actor/channel model that mirrors cube message passing and liquid updates.

---

## 1. Actors

- BlockActor: wraps a Transformer block, exposes `process(h_in)`.
- CubeActor: stores keys/values; handles `query(k)` and `update(k,v)`; persists shards.
- LiquidActor: executes liquid steps with per-cube state.
- GateActor: computes α and blends.

---

## 2. Channels

- h_in → BlockActor → (k, y_T) → [CubeActor, LiquidActor] → GateActor → y
- Back-channel audit: Teacher recompute requests sampled by GateActor.

---

## 3. Virtual dendrites

- Logical connections between cubes across blocks (feed-forward of predictions) are represented as channels with typed messages and backpressure.

---

## 4. Concurrency

- Per-batch sharding of tokens across cube actors; bounded queues; cancellation on watchdog events.

---

## 5. Persistence

- On shutdown, flush cube shards; keep small metadata index; TTL cleanup jobs.

