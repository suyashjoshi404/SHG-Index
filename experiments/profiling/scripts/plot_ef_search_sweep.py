#!/usr/bin/env python3
"""Plot full-query-set recall against mean latency across efSearch repeats."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


def summarize(run_dir):
    measurements = defaultdict(list)
    for path in sorted(run_dir.glob("*/*/ef*_k*/repeat*_performance.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0]["phase"] != "search":
            raise ValueError(f"Expected exactly one search row in {path}")
        row = rows[0]
        if (int(row["M"]), int(row["ef_construction"]), int(row["k"]),
                int(row["threads"])) != (48, 80, 20, 1):
            raise ValueError(f"Expected M48/efC80/k20/threads1 in {path}")
        key = row["dataset"], row["algorithm"], int(row["ef_search"])
        measurements[key].append(row)
    expected = {(dataset, algorithm, ef)
                for dataset in ("sift", "enron") for algorithm in ("hnsw", "shg")
                for ef in (20, 24, 32, 48, 64, 96, 128, 192, 256, 512)}
    if set(measurements) != expected:
        raise ValueError(f"Incomplete or unexpected sweep: missing {expected - set(measurements)}, "
                         f"extra {set(measurements) - expected}")
    summary = []
    for (dataset, algorithm, ef), rows in sorted(measurements.items()):
        if len(rows) != 5:
            raise ValueError(f"Expected five repeats for {dataset}/{algorithm}/ef{ef}")
        count = 10000 if dataset == "sift" else 200
        if {int(r["count"]) for r in rows} != {count}:
            raise ValueError(f"Unexpected query count for {dataset}/{algorithm}/ef{ef}")
        recalls = [float(r["recall_at_k"]) for r in rows]
        if not all(math.isclose(r, recalls[0], abs_tol=1e-8) for r in recalls):
            raise ValueError(f"Recall differs between repeats for {dataset}/{algorithm}/ef{ef}")
        latencies = [float(r["average_latency_us"]) / 1000 for r in rows]
        if not all(math.isfinite(t) and t > 0 for t in latencies):
            raise ValueError("Latency must be finite and positive")
        summary.append(dict(dataset=dataset, algorithm=algorithm, M=48,
                            ef_construction=80, k=20, ef_search=ef, repeats=5,
                            query_count=count, mean_recall_at_k=fmean(recalls),
                            mean_latency_ms=fmean(latencies),
                            std_latency_ms=stdev(latencies),
                            min_latency_ms=min(latencies), max_latency_ms=max(latencies)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Observation directory for one complete sweep")
    args = parser.parse_args()
    summary = summarize(args.run_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.linewidth": 0.6, "axes.edgecolor": "0.5",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for index, (dataset, title) in enumerate((("sift", "SIFT1M"), ("enron", "Enron"))):
        ax = axes[index]
        for algorithm, color, marker in (("shg", "red", "o"), ("hnsw", "black", "D")):
            points = sorted((r for r in summary if r["dataset"] == dataset
                             and r["algorithm"] == algorithm), key=lambda r: r["ef_search"])
            ax.plot([r["mean_latency_ms"] for r in points],
                    [r["mean_recall_at_k"] for r in points],
                    color=color, marker=marker, markerfacecolor="none",
                    markersize=5, markeredgewidth=0.8, linestyle="-.",
                    linewidth=0.8, label=algorithm.upper())
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Recall@20")
        ax.set_ylim(top=1.005)
        ax.set_title(f"({chr(ord('a') + index)}) Performance on {title}",
                     fontfamily="DejaVu Serif", fontweight="bold", fontsize=11, pad=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(False)
    fig.suptitle("Search performance (M=48, efC=80, k=20)",
                 fontfamily="DejaVu Serif", fontweight="bold", fontsize=13)
    fig.text(0.5, 0.015,
             "efSearch = 20, 24, 32, 48, 64, 96, 128, 192, 256, 512; mean latency across five repeats.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), w_pad=2)
    output_dir = args.run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "hnsw_shg_ef_search_sweep_M48_efc80_k20_mean"
    for extension in ("png", "pdf", "svg"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(stem.with_suffix(".csv"))
    print(f"Validated {len(summary)} sweep points from {len(summary) * 5} search measurements.")


if __name__ == "__main__":
    main()
