---
title: FlowLang Overview
last-updated: 2025-09-27
---

# FlowLang Overview

FlowLang is a simple, file-centric flow description for Codex pipelines. It uses the concepts of blocks and links to
declare a directed graph that the runtime validates and executes.

- It is not LangFlow (a GUI project). We use the term FlowLang to refer to our text-based DSLs and the
  JavaScript/TypeScript-based flow modules documented here.
- Author flows as either:
  - Minimal line-oriented files (e.g., `.flow` with `block` and `link` lines), or
  - JavaScript/TypeScript modules that construct a graph programmatically and return a JSON `FlowGraph`.

Quick links:

- FlowLang JS DSL Spec: `FLOWLANG_SPEC.md`
- Flow inspector script: `scripts/check-flow.ts` at repo root (reads `.codex-dev/config.json` and prints blocks/links).

Guidance:

- Keep the MCP router on the neutral boundary. Tool invocations are normalized and executed via MCP regardless of how
  the model emits tool calls (functions/XML/JSON).
- Prefer explicit codecs for now (`mcp_over_functions`, `mcp_over_xml`); switch to `auto` once model/provider behavior
  is confirmed.
- Use feature flags in the CLI config to enable negotiation, memory, advisor messaging, and compaction as needed.
