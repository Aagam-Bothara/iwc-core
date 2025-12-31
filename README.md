# IWC — Inference Workload Compiler & Characterizer

IWC is a focused CLI tool that **decouples LLM datasets from inference engines** by compiling them into a **canonical workload format**, and then **analyzing and comparing workload behavior** to ensure benchmarking is reproducible, comparable, and meaningful.

At its core, IWC answers two critical questions in LLM inference benchmarking:

> **Can I reproduce this workload exactly?**  
> **Is this workload behaviorally comparable to another one?**

Most tools answer the first only. IWC does both.

---

## Why IWC exists

While benchmarking LLM inference, the same issues kept appearing:

- Every dataset uses a different structure
- Inference tools assume their own request formats
- Arrival patterns (bursty vs steady vs Poisson) are implicit or undocumented
- Small prompt formatting changes silently alter workload behavior
- Re-running the "same" benchmark weeks later is rarely identical
- Two workloads with the same RPS can stress hardware very differently

IWC was built to make **inference workloads explicit, auditable, and comparable** — before you ever run a benchmark.

---

## What IWC does (high level)

IWC has **two tightly connected layers**:

### 1. Compile workloads (reproducibility)

Convert datasets into a single, schema-validated **canonical workload JSONL**, plus a **manifest** that records exactly how it was generated.

### 2. Characterize workloads (comparability)

Analyze workload behavior (tokens, arrivals, sessions) and **diff two workloads** to detect semantic drift — with optional CI gating.

Together, this forms a complete foundation for reliable inference benchmarking.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install tiktoken
```

---

## Part 1 — Workload Compilation

### Canonical workload format (JSONL)

Each line represents one inference request:

```json
{
  "request_id": "req-000001",
  "prompt": "Explain KV cache in one sentence.",
  "max_output_tokens": 128,
  "arrival_time_ms": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "streaming": false
}
```

The format is validated against: `schema/workload.schema.json`

### Manifest (reproducibility metadata)

For every workload, IWC also produces:

```
<workload>.manifest.yaml
```

It records:

- SHA256 hashes of inputs, outputs, and schema
- Compiler type and parameters
- Arrival model and seed
- Summary statistics (request count, arrival span, etc.)
- IWC version

This makes benchmarking auditable and reproducible.

### Quickstart (2 minutes)

```bash
iwc compile simple-json --input data.json --output workload.jsonl
iwc validate workload.jsonl
```

Outputs:

```
workload.jsonl
workload.jsonl.manifest.yaml
```

You can now feed `workload.jsonl` into any inference engine.

### Supported compilers

#### 1. Simple JSON

Accepted input forms:

```json
["prompt1", "prompt2"]
```

```json
[{ "prompt": "..." }]
```

Command:

```bash
iwc compile simple-json --input data.json --output out.jsonl
```

#### 2. ShareGPT

Supports common ShareGPT-style formats:

- conversations with `human`/`gpt` or `user`/`assistant`
- messages arrays with `role`/`content`

**Single-turn mode**

```bash
iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_single.jsonl \
  --mode single-turn
```

**Session mode**

```bash
iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_session.jsonl \
  --mode session \
  --user-tag "User" \
  --assistant-tag "Assistant" \
  --separator "\n"
```

### Arrival models

IWC explicitly models request arrival patterns.

**Fixed step (default)**

```bash
--arrival fixed-step --arrival-step-ms 100
```

**Poisson arrivals (realistic traffic)**

```bash
--arrival poisson --rate-rps 5 --seed 123
```

Arrivals are seeded for reproducibility.

---

## Part 2 — Workload Analysis

### Analyze a workload

```bash
iwc analyze workload.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

Example output:

```
WORKLOAD SUMMARY
----------------
Requests  : 5
Tokenizer : tiktoken:gpt-4o-mini

WORKLOAD TYPE : smooth, prefill-heavy, high-reuse
```

This surfaces properties that dominate inference performance:

- Prefill vs decode dominance
- Arrival variability
- Session reuse and context growth

### What the metrics mean (brief)

| Metric | Description |
|--------|-------------|
| **Prefill dominance** | Fraction of tokens spent in prompt processing vs output generation. High → memory bandwidth and KV-cache pressure. |
| **Burstiness (CV)** | Variability of inter-arrival times. High → scheduler stress and latency spikes. |
| **Prompt reuse ratio** | Fraction of prompt tokens reused across turns. High → chat-like workloads. |

### Primary workload class

- `bursty-api`
- `batch/offline`
- `interactive-chat` (prefill-heavy)

---

## Part 3 — Workload Diffing

### Compare two workloads

```bash
iwc diff A.jsonl B.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

### JSON output (CI / dashboards)

```bash
iwc diff A.jsonl B.jsonl \
  --format json \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

### Regression gating (CI)

```bash
iwc diff A.jsonl B.jsonl \
  --fail-on-prefill-delta 0.05 \
  --fail-on-reuse-delta 0.05 \
  --fail-on-burstiness-delta 0.5
```

If thresholds are exceeded:

- Diff is printed
- Exit code = 2
- CI fails

This prevents silent workload drift that invalidates benchmarks.

---

## Validation

```bash
iwc validate workload.jsonl
iwc validate ./folder_with_jsonl_files/
```

---

## Status & Roadmap

### Completed

- [x] Canonical workload compilation
- [x] Schema validation + manifests
- [x] Workload analysis and classification
- [x] Workload diffing
- [x] CI regression gating
- [x] Golden tests + GitHub Actions CI

### Planned

- [ ] Compact workload fingerprint export
- [ ] Additional dataset adapters (Alpaca, MT-Bench, OpenAI logs)
- [ ] Runner integrations (vLLM, TGI)
- [ ] Optional visualization exports