# Experiment 1 — Concurrency sweep

**Goal:** quantify how tail latency changes with concurrency under a fixed workload.

- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Backend: vLLM OpenAI server
- Workload: fixed compiled workload reused across runs
- Concurrency sweep: 1, 2, 4, 8, 16

Outputs:
- summary.csv
- latency_vs_concurrency.(png|pdf)
- throughput_vs_concurrency.(png|pdf)

Key observation: Concurrency=1 shows significantly higher latency (64ms mean) due to sequential processing. Higher concurrency levels (2-16) achieve similar low latencies (~16-19ms) as requests run in parallel.
