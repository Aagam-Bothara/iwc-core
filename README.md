# IWC — Inference Workload Compiler

IWC is a toolkit for building, running, and analyzing reproducible LLM inference workloads.

Unlike ad-hoc benchmarking scripts, IWC makes workloads **explicit**, **inspectable**, and **comparable** by separating:

- **what** is being run (prompts, arrivals, sessions)
- **how** it is executed (model server, concurrency, retries)
- **how results are interpreted** (latency, tokens, truncation, errors)

This allows meaningful comparisons across models, backends, and time.

---

## Why IWC?

LLM inference benchmarks are often misleading because:

- The same dataset is formatted differently across tools
- Arrival patterns are implicit or undocumented
- Small prompt changes silently alter workload behavior
- Results lack structured error, retry, and token accounting

As a result, numbers look precise — but are not reproducible.

IWC fixes this by making both workloads and results first-class artifacts.

---

## What IWC Does Today

### Workload Compilation

- Compile datasets (e.g., Alpaca) into a canonical JSONL workload format
- Explicit arrival modeling (fixed-step, Poisson)
- Deterministic generation via seeds
- Manifest generation for traceability

### Validation & Analysis

- Schema validation for workloads
- Token statistics (prompt/output, dominance)
- Arrival analysis (RPS, burstiness)
- Session detection and prompt reuse
- Automatic workload classification
- Semantic workload diffing

### Execution

- Run workloads against vLLM servers
- Controlled concurrency and retries
- Structured, versioned result records
- Robust error handling and recovery

### Reporting

- Aggregate benchmark reports
- Latency percentiles (p50/p90/p95/p99)
- Throughput (req/s, tokens/s)
- Truncation detection
- Error breakdowns
- Metadata capture (model, server, versions)

---

## Quickstart

### 1. Compile a workload
```bash
iwc compile alpaca \
  --input alpaca.json \
  --output workload.jsonl \
  --arrival poisson \
  --rate-rps 10 \
  --seed 42
```

### 2. Validate the workload
```bash
iwc validate workload.jsonl
```

### 3. Analyze workload behavior
```bash
iwc analyze workload.jsonl
```

### 4. Run against a vLLM server
```bash
iwc run vllm \
  --workload workload.jsonl \
  --base-url http://localhost:8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --out results.jsonl \
  --concurrency 4
```

### 5. Generate a report
```bash
iwc report-results --input results.jsonl
```

---

## Architecture

IWC decouples inference benchmarking into discrete stages:
```
Dataset
   ↓
Compiler → Canonical Workload
   ↓
Validator / Analyzer
   ↓
Runner (vLLM, others)
   ↓
Structured Results
   ↓
Reports & Diffs
```

This design allows:

- The same workload to be reused across backends
- Meaningful diffs between workload versions
- Auditing and reproduction months later

---

## Result Schema

Each inference result includes:

- Schema and runner versioning
- Precise timing (start, end, latency)
- Retry and error metadata
- Token accounting (input/output/total)
- Truncation detection (`finish_reason`)
- Server and model identifiers

This enables defensible benchmarking, not just raw numbers.

---

## Status

IWC is an actively developed research and engineering tool.

Schemas are versioned and stable, but may evolve as new backends and features are added.

### Planned

- Additional runners (OpenAI-compatible, TGI)
- Streaming TTFT measurement
- Environment metadata capture
- Visualization-ready exports

---

## Who This Is For

- Systems and ML engineers benchmarking inference servers
- Researchers comparing model behavior under controlled workloads
- Anyone who needs reproducible, auditable LLM inference results

---

## License

MIT

