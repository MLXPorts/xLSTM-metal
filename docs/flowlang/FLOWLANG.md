---
title: FlowLang — Lightweight Pipeline Language
last-updated: 2025-09-02
---

# FlowLang — Lightweight Pipeline Language

FlowLang is a tiny, fast, line-based language to declare Codex pipelines as graphs of typed blocks (nodes) connected by
typed ports (edges). It’s inspired by flow-based systems (e.g., LangFlow) but optimized for CLI/runtime use: minimal
syntax, no heavy parsing, and tight integration with Codex blocks.

## Goals

- Minimal: human-readable, single-pass interpreter; no JSON/YAML overhead.
- Typed: blocks expose input/output ports with simple port type names.
- Composable: protocol (connectors), codec (transport), family (prompt shaping), tools (advertise), router (MCP) wire
  together.
- Config-first: FlowLang can be loaded from config, generated from provider specs, or overridden by DSL rules.

## Core Concepts

- Block (node): a component (e.g., `protocol.ollama`, `codec.mcp_over_xml`, `router.mcp`, `family.qwen3`,
  `tools.negotiation_advertise`).
- Port (typed): input/output endpoints, e.g., `messages.in`, `messages.out`, `stream.delta`, `tool_use.neutral`,
  `tool_result.neutral`, `hints`.
- Link (edge): connects compatible ports: `fromId.port -> toId.port`.
- Vars & Conditions: lightweight `set` and `when/endif` to conditionally include blocks/links.
- Macros: `use` includes predefined snippets (e.g., `tools.negotiation_advertise`).

## Directives

- `block <id> <type> [param=value[,value2...]]` — declare a node
  - Example: `block p protocol.ollama`
- `link <fromId>.<port> -> <toId>.<port>` — connect ports
  - Example: `link p.stream.delta -> x.stream.delta`
- `set <key>=<value>` — set a flow variable
  - Example: `set provider=ollama`
- `when <key>=<csv> [supports=<csv>]` — start a conditional block (ends with `endif`)
  - Example: `when provider=ollama family=qwen supports=xml`
- `endif` — end condition
- `use <name>` — include a predefined snippet/macro
  - Example: `use tools.negotiation_advertise`
- `hint <key>=<value>` — provide request hints (e.g., nonces)
- `end` — optional terminator
- `# ...` — comments

## Port Conventions (Typed)

- protocol.*
  - inputs: `messages.in`
  - outputs: `stream.delta`, `messages.out`
- codec.*
  - inputs: `stream.delta`, `tool_result.neutral`
  - outputs: `tool_use.neutral`, `messages.out`
- router.mcp
  - inputs: `tool_use.neutral`
  - outputs: `tool_result.neutral`
- family.*
  - inputs: `messages.in`
  - outputs: `messages.out`
- tools.negotiation_advertise (macro)
  - outputs: `messages.prefix`, `tools.function_defs`
- advisor.*
  - inputs: `messages.in`, `decisions.in`, `toolplan.in`
  - outputs: `messages.out`, `decisions.out`, `critiques.out`
- scheduler.idle_nudge
  - outputs: `messages.out`
- sink.ui (implicit)
  - inputs: `messages.out`

## Minimal Grammar (Informal)

- Tokens are space-delimited; key=value pairs allow characters like `:/.+-_` without quoting.
- EBNF:
  - `line := (comment | block | link | set | when | endif | use | hint | end)`
  - `block := "block" id type [args]`
  - `link := "link" id "." port "->" id "." port`
  - `set := "set" key "=" value`
  - `when := "when" cond { " " cond }`
    - `cond := key "=" csv | "supports=" csv`
  - `endif := "endif"`
  - `args := key "=" csv { "," csv }`
  - `csv := value { "," value }`
  - `use := "use" name`
  - `hint := "hint" key "=" value`
  - `end := "end"`
  - `comment := "#" text`

## Examples

### 1) Ollama + Qwen (XML-in-text)

```
block p protocol.ollama
block x codec.mcp_over_xml
block r router.mcp
block f family.qwen3
use tools.negotiation_advertise
link f.messages.out -> p.messages.in
link p.stream.delta -> x.stream.delta
link x.tool_use.neutral -> r.tool_use.neutral
link r.tool_result.neutral -> x.tool_result.neutral
link x.messages.out -> sink.ui.messages
end

### 1b) With Advisor Gate + Deliberation

```

block f family.qwen3
block a1 advisor.deliberate
block g advisor.gate threshold=medium when=preTool,prePatch
block p protocol.ollama
block x codec.mcp_over_xml
block r router.mcp
use tools.negotiation_advertise
link f.messages.out -> a1.messages.in
link a1.messages.out -> g.messages.in
link g.messages.out -> p.messages.in
link p.stream.delta -> x.stream.delta
link x.tool_use.neutral -> r.tool_use.neutral
link r.tool_result.neutral -> x.tool_result.neutral
link x.messages.out -> sink.ui.messages
end

```
```

### 2) LM Studio (unknown → auto; handshake decides)

```
set provider=lmstudio
block p protocol.openai base=http://localhost:1234/v1
block a codec.auto
block r router.mcp
use tools.negotiation_advertise
block f family.qwen3
link f.messages.out -> p.messages.in
link p.stream.delta -> a.stream.delta
link a.tool_use.neutral -> r.tool_use.neutral
link r.tool_result.neutral -> a.tool_result.neutral
link a.messages.out -> sink.ui.messages
end
```

### 3) Conditional (provider-specific)

```
set provider=ollama
set family=qwen
when provider=ollama family=qwen supports=xml
  block p protocol.ollama
  block x codec.mcp_over_xml
endif
when provider=lmstudio supports=functions
  block p protocol.openai
  block c codec.mcp_over_functions
endif
block r router.mcp
block f family.qwen3
use tools.negotiation_advertise
link f.messages.out -> p.messages.in
link p.stream.delta -> x.stream.delta
link x.tool_use.neutral -> r.tool_use.neutral
link r.tool_result.neutral -> x.tool_result.neutral
link x.messages.out -> sink.ui.messages
end
```

## Interpreter (Lightweight)

- Single-pass, no external dependencies.
- Maintains:
  - `nodes`: `id → { type, params }`
  - `edges`: `{ from: {id, port}, to: {id, port} }[]`
  - `vars`: from `set`, e.g., `provider`, `family`
  - `supports`: implied by provider specs + capability cache
  - `hints`: `key=value` for request hints (e.g., `cap_probe_nonce`)
- `when/endif` push/pop a boolean include flag; lines are skipped when false.
- `use` expands known macros (e.g., `tools.negotiation_advertise → block t tools.negotiation_advertise`).
- Ports are validated for compatibility using static block metadata.
- Compilation:
  - FlowLang nodes map to existing blocks (protocol.*, codec.*, router.mcp, family.*, tools.*).
  - Produces an ordered block list for the PipelineEngine; edges inform stream-time handoff & hints.

## Performance

- Simple tokenization; no JSON parse.
- Static metadata for blocks/ports; zero reflection at runtime.
- Compiled flows can be cached by `{provider, model, family}` key.

## Coexistence with Current System

- FlowLang can be:
  - Loaded explicitly from a file (`pipelines.flow: path`), or
  - Generated automatically from Provider Specs + capabilities when `pipelines.autobuild=true`, or
  - Overridden by Pipeline DSL rules (rules take precedence).
- Blocks and codepaths remain the same; FlowLang is a thin orchestration layer above them.

## Next Steps

1) Add block metadata (`getPorts()`) for protocol.*, codec.*, router.mcp, tools.*, and family.qwen3.
2) Implement parser/interpreter `FlowLang.parse/compile` (no external deps).
3) Provide example flows under `docs/flow-examples/` (Ollama+Qwen, LM Studio auto, OpenAI functions).
4) Add config key `pipelines.flow` to point at a FlowLang file (optional).
5) Integrate with Provider Spec DSL so `autobuild` can emit a FlowLang graph on the fly.
