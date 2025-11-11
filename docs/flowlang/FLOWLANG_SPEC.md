---
title: JavaScript FlowLang — Blocks, Ports, Validation, Safe Mode
last-updated: 2025-09-24
---

# JavaScript FlowLang — Blocks, Ports, Validation, Safe Mode

This document proposes an explicit, FlowLang “blocks and wires” DSL implemented in real JavaScript/TypeScript. Dynamic
code defines the flow graph by instantiating blocks and connecting typed ports. The parent runtime executes that code
safely in a sandbox, validates the returned graph, and compiles it to a running pipeline.

The goal is code‑as‑configuration with a productive developer experience, without introducing a bespoke parser or
fragmentary config layers. The dynamic code is small, composable, and validated at runtime.

## Design Overview

- Blocks: reusable components such as `protocol.openai`, `codec.mcp_over_functions`, `codec.auto`, `router.mcp`,
  `tools.negotiation_advertise`, `family.qwen3`, advisor blocks, and sinks.
- Ports: typed interfaces for connecting blocks: `messages.in|out`, `stream.delta`, `tool_use.neutral`,
  `tool_result.neutral`, `events.out`, `usage.out`.
- Graph: a `Flow` object containing nodes (blocks) and edges (links). Dynamic code returns a `FlowGraph` value which is
  validated (e.g., via Zod) and then compiled.
- Neutral boundary: All tool invocations are normalized to neutral and executed via MCP; blocks enforce this contract.
  The runtime gates “one active tool channel” to prevent double execution when models mix formats.
- Security: Dynamic code executes in an isolated VM context with a narrowly scoped API (no filesystem/network access
  unless provided as tools). Execution has time/memory caps and is validated before use.

## Authoring Model (Dynamic Code)

Dynamic code imports the block catalog and returns a flow graph. It does not directly access the OS; it builds a graph
that the parent compiles and runs.

Example (TypeScript or JS):

```ts
// flows/lmstudio-qwen-functions.flow.js
export default function build({ blocks }) {
  // 1) Instantiate blocks
  const f = blocks.family.qwen3();
  const p = blocks.protocol.openai({ baseUrl: 'http://localhost:1234/v1' });
  const x = blocks.codec.mcp_over_functions();
  const r = blocks.router.mcp();
  const t = blocks.tools.negotiation_advertise();
  const sink = blocks.sink.ui.messages();

  // 2) Connect typed ports (wires)
  const edges = [
    { from: [f.id, 'messages.out'], to: [p.id, 'messages.in'] },
    { from: [p.id, 'stream.delta'], to: [x.id, 'stream.delta'] },
    { from: [x.id, 'tool_use.neutral'], to: [r.id, 'tool_use.neutral'] },
    { from: [r.id, 'tool_result.neutral'], to: [x.id, 'tool_result.neutral'] },
    { from: [x.id, 'messages.out'], to: [sink.id, 'events.out'] },
  ];

  // 3) Return the graph (validated by the parent)
  return {
    version: 1,
    name: 'LMStudio-Qwen Functions',
    nodes: [f, p, x, r, t, sink],
    edges,
  };
}
```

Notes:

- Blocks carry their port metadata, so the parent can type‑check edges before compiling.
- Families and codecs are explicitly chosen here to reflect model behavior we understand (e.g., LM Studio + Qwen via
  functions). As autodetection matures, authors can switch to `codec.auto()`.

## Parent Runtime (Sandbox + Validation)

The parent runtime loads and executes the user’s flow file inside a VM sandbox. The sandbox exposes only the block
catalog API and a small helper surface; no `require` or Node globals are available unless explicitly provided.

High‑level skeleton:

```ts
import vm from 'node:vm';
import { z } from 'zod';

// 1) Define block catalog (constructors + metadata)
// In practice, import from our runtime: protocol, codec, router, tools, family, sinks.
const blocks = createBlockCatalog();

// 2) Zod schemas for FlowGraph
const Edge = z.object({
  from: z.tuple([z.string(), z.string()]),
  to: z.tuple([z.string(), z.string()]),
});
const Node = z.object({ id: z.string(), type: z.string(), options: z.record(z.any()).default({}) });
const FlowGraph = z.object({
  version: z.number().int().min(1),
  name: z.string().min(1),
  nodes: z.array(Node).min(1),
  edges: z.array(Edge).min(1),
});

// 3) Runner
export async function runFlowScript(source: string) {
  const sandbox = {
    module: { exports: {} },
    exports: {},
    blocks, // Only the catalog, nothing else
  };
  vm.createContext(sandbox);
  const wrapped = `
    (function (exports, module, blocks) {
      ${source}
    })(exports, module, blocks);
  `;
  vm.runInContext(wrapped, sandbox, { timeout: 1000 });
  const fn = sandbox.module.exports?.default;
  if (typeof fn !== 'function') throw new Error('Flow module must export default function');
  const graph = await fn({ blocks });
  const parsed = FlowGraph.parse(graph); // shape check
  validatePortsAndTypes(parsed, blocks); // type check ports; ensure no cycles/multi-writers
  return parsed;
}
```

Validation rules:

- Unknown block type → error.
- Unknown port name → error.
- Port type mismatch → error (e.g., `stream.delta` → `messages.in`).
- Multiple writers to a single input → error.
- Cycles on message/tool paths → error (unless explicit queues/gates are supported later).

## Blocks and Ports (Catalog)

Each block constructor returns an object with a unique `id`, a `type` (stable name like `protocol.openai`), an `options`
object, and embedded `ports` metadata. The runtime uses metadata to validate connections before compiling.

Example (simplified):

```ts
// Example catalog entry (simplified)
function mcpRouter() {
  return {
    id: `router_${Math.random().toString(36).slice(2)}`,
    type: 'router.mcp',
    options: {},
    ports: {
      inputs: ['tool_use.neutral'],
      outputs: ['tool_result.neutral'],
    },
  };
}
```

Catalog (initial set):

- Protocol: `protocol.openai`, `protocol.ollama`
- Codec: `codec.mcp_over_functions`, `codec.mcp_over_xml`, `codec.auto`
- Router: `router.mcp`
- Tools: `tools.negotiation_advertise`
- Family: `family.qwen3`
- Advisor (optional): `advisor.gate`, `advisor.deliberate`, `scheduler.idle_nudge`
- Sinks: `sink.ui.messages`, `sink.plain.events`

### Kernel Blocks (Non‑Removable)

To protect connectivity and enforce invariants, the compiler treats the following as kernel blocks:

- `router.mcp` (neutral boundary and tool executor): always present. If a flow omits it, it is injected. Flows cannot
  bypass or remove it.
- Protocol → Codec → Router wiring: the compiler validates that one protocol feeds one codec, which feeds the router;
  flows that violate this are rejected.
- Negotiation (autodetection): handshake probes (cap_probe) and autodetection logic are owned by codecs/runtime and
  cannot be disabled from a JS flow.

## Safe Mode and Flow Repair

Flows are user‑editable code. To prevent a broken edit from severing connectivity, the runtime provides Safe Mode and a
guarded “flow_repair” tool.

- Triggers:
    - Flow compile/validation error (unknown block/port, type mismatch, multi‑writer, cycle).
    - Router detection error (“unsupported tool use …”).
- Safe Mode behavior:
    - Suspend the user flow; run a known‑good baseline graph (protocol → codec → router.mcp → sink) selected by Pipeline
      DSL.
    - Keep strict autodetection and MCP routing; no format guessing.
    - Emit a concise system message with failure + suggested action.
- Repair Tool (structured via MCP):
    - Name: `flow_repair`
    - Params (schema): `{ path: string; patch: { nodes?: Node[], edges?: Edge[] } }`
    - Constraints:
        - Kernel blocks (`router.mcp`, protocol→codec→router wiring) cannot be removed or bypassed.
        - Provider credentials, base URLs, and secrets are immutable.
        - Only nodes/edges/options may be added/adjusted when valid.
    - Flow is re‑validated (Zod + port type checks) before adoption.
- Exit Safe Mode:
    - On successful compile + validation of the repaired flow, resume normal execution.

This keeps the AI productive (can propose repairs) while preserving strict routing and safety guarantees.

## Security Model

- VM sandbox with only the `blocks` catalog and a narrow helper context. No filesystem/network access is exposed
  directly to dynamic scripts.
- Execution timeouts and memory limits enforced.
- Results must pass schema and port‑type validation before compilation.
- Tool execution remains behind the MCP boundary and CLI approval gates.

## JSON Graph (for Editors)

Although authors can write pure JavaScript, the flow graph is a plain JSON structure (
`{ version, name, nodes, edges }`). Editors (like a FlowLang‑aware UI) can generate/consume this JSON and round‑trip
back to JS as needed. The parent runtime treats JS and JSON identically after validation.

## Examples

1) LM Studio + Qwen via Functions

- Explicit functions path while behavior is being validated.

2) Ollama + Qwen via XML‑in‑text

- Use `codec.mcp_over_xml()` and wire the same router path; results render back through the codec.

3) Autodetection (maturing path)

- Swap to `codec.auto()` once behavior is confirmed for the target model/provider.

## Migration and Coexistence

- Pipeline DSL remains the simplest entry point. Over time, teams can export auto‑built flows to JS for explicit
  control.
- JS DSL and DSL rules can coexist: rules assemble defaults; JS files provide hand‑tuned graphs for specific scenarios.
- No ripping out working flows: adopt gradually on a per‑flow basis.

## Appendix: Dynamic Runtime Mini‑Example

The following standalone example shows running sandboxed JS that uses a tiny API and returns validated data. It mirrors
the approach above at a minimal scale.

```ts
import vm from 'node:vm';
import { z } from 'zod';

class ComponentOutput { constructor(public result: any, public error: string | null = null) {} }
const ComponentOutputSchema = z.object({ result: z.any(), error: z.string().nullable() });
const ResultSchema = z.object({ finalAnswer: z.number(), steps: z.array(z.string()) });

function execute(code: string, input: any) {
  const sandbox = { module: { exports: {} }, exports: {}, input, ComponentOutput };
  vm.createContext(sandbox);
  const wrapped = `(function(exports, module, input, ComponentOutput){ ${code} })(exports, module, input, ComponentOutput);`;
  vm.runInContext(wrapped, sandbox, { timeout: 1000 });
  return sandbox.module.exports;
}

const userCode = `
  const a = input.a, b = input.b;
  const sum = a + b; const prod = a * b;
  return new ComponentOutput({ finalAnswer: prod - sum, steps: [
    `${a} + ${b} = ${sum}`,
    `${a} * ${b} = ${prod}`,
    `${prod} - ${sum} = ${prod - sum}`
  ]});
`;

const raw = execute(userCode, { a: 10, b: 5 });
const validated = ComponentOutputSchema.parse(raw);
if (validated.error) throw new Error(validated.error);
const result = ResultSchema.parse(validated.result);
console.log('Final Answer:', result.finalAnswer);
```
