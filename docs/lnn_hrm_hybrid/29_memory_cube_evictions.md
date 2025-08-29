# Memory Cube Evictions & Budgets — Design

Add TTL/LRU policies and UMA-aware backpressure for MemoryCube.

## Policies
- Capacity cap (existing): keep last N (FIFO); configurable.
- LRU (proposed): track hit counters; evict lowest.
- TTL (proposed): drop stale entries beyond T steps.

## UMA Hooks
- Budget signals from system (bytes used, alloc failures) → lower max_items; damp α.

## API
- `MemoryCube(..., policy='fifo'|'lru'|'ttl', ttl=None)`
- Telemetry: `cube_bytes, evict_count, policy`

## Tests
- Eviction correctness under each policy
- Budget downshift triggers lower α_max

