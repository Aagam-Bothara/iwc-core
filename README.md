COPY-PASTE README.md
# IWC — Inference Workload Compiler

IWC is a small but focused tool that **decouples datasets from LLM inference engines** by compiling them into a **canonical workload format**.

The goal is to make LLM inference benchmarking:
- reproducible
- comparable across engines
- independent of dataset quirks

---

## What Problem This Solves

While benchmarking LLM inference, I repeatedly ran into the same issues:

- Every dataset uses a different structure
- Inference tools assume their own request format
- Arrival patterns (burst vs steady vs poisson) are often implicit
- Re-running the *same* benchmark later is hard to reproduce exactly

IWC solves this by turning datasets into a **single canonical workload JSONL**, along with a **manifest** that captures how that workload was generated.

---

## What IWC Produces

### 1. Workload JSONL (canonical format)

Each line represents **one inference request**:

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


The format is validated against schema/workload.schema.json.

2. Manifest YAML (reproducibility metadata)

For every workload, IWC writes:

<output>.manifest.yaml


It contains:

sha256 hashes of input, output, and schema

compiler type

arrival model + parameters

summary stats (num_requests, arrival_span_ms, skipped_records, etc.)

This makes runs auditable and reproducible.

Install
python -m venv .venv
source .venv/bin/activate
pip install -e .

Quickstart (2 minutes)

Convert a dataset into a canonical inference workload:

iwc compile simple-json --input data.json --output workload.jsonl
iwc validate workload.jsonl


This produces:

workload.jsonl

workload.jsonl.manifest.yaml

You can now feed workload.jsonl into any inference engine.

Supported Compilers
1. Simple JSON

Accepted input formats:

["prompt1", "prompt2", ...]

[{"prompt": "..."}]

iwc compile simple-json --input data.json --output out.jsonl
iwc validate out.jsonl

2. ShareGPT

Supports common ShareGPT-style JSON variants:

conversations with human/gpt or user/assistant

messages with role/content

Single-turn mode

Takes the first user message per record.

iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_single.jsonl \
  --mode single-turn

iwc validate sh_single.jsonl

Session mode

Packs a multi-turn conversation into a single prompt transcript.

iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_session.jsonl \
  --mode session \
  --user-tag "User" \
  --assistant-tag "Assistant" \
  --separator "\n"

Arrival Models

IWC explicitly models request arrival patterns.

Fixed-step (default)
--arrival fixed-step --arrival-step-ms 100

Poisson arrivals (realistic traffic)
--arrival poisson --rate-rps 5 --seed 123


Arrivals are seeded for reproducibility.

Validation

Validate any workload against the schema:

iwc validate workload.jsonl
iwc validate ./folder_with_jsonl_files/

Design Notes

ShareGPT session mode emits one request per conversation

Per-turn workloads would require schema extensions (session_id, turn_id)

Manifest stores escaped separators for readability

iwc_version is read dynamically from package metadata

Roadmap

Additional dataset adapters (Alpaca, OpenAI chat logs, MT-Bench)

Optional schema extension for turn-level workloads

Runner integration (vLLM / TGI)

Metric hooks for latency, throughput, and energy efficiency


