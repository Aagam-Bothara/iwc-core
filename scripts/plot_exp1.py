from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

in_path = Path("runs/exp1/summary.csv")
out_dir = Path("runs/exp1")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_path)
df = df.sort_values("concurrency")

# 1) Latency vs concurrency
plt.figure()
plt.plot(df["concurrency"], df["p50_ms"], marker="o", label="p50")
plt.plot(df["concurrency"], df["p99_ms"], marker="o", label="p99")
plt.plot(df["concurrency"], df["mean_ms"], marker="o", label="mean")
plt.xscale("log", base=2)
plt.xlabel("Concurrency")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(out_dir / "latency_vs_concurrency.png", dpi=300)
plt.savefig(out_dir / "latency_vs_concurrency.pdf")
plt.close()

# 2) Throughput vs concurrency
plt.figure()
plt.plot(df["concurrency"], df["req_per_s"], marker="o")
plt.xscale("log", base=2)
plt.xlabel("Concurrency")
plt.ylabel("Requests / sec")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(out_dir / "throughput_vs_concurrency.png", dpi=300)
plt.savefig(out_dir / "throughput_vs_concurrency.pdf")
plt.close()

print("Wrote plots to runs/exp1/")
