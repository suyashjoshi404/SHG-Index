#!/usr/bin/env python3
"""Plot the latest HNSW and SHG build/search timings for each dataset."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OBSERVATIONS_DIR = EXPERIMENT_DIR / "observations"


def matching_rows(path, phase, M, ef_search, k, threads, ef_construction, seed):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching = [
        row for row in rows
        if row["phase"] == phase
        and row["M"] == str(M)
        and row["ef_construction"] == str(ef_construction)
        and row["threads"] == str(threads)
        and (phase != "build" or row["seed"] == str(seed))
        and (phase == "build" or (
            row["ef_search"] == str(ef_search)
            and row["k"] == str(k)
            and row["threads"] == str(threads)
        ))
    ]
    if not matching:
        raise ValueError(f"No matching {phase} row found in {path}")
    return matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("sift", "enron"),
                        default=("sift", "enron"))
    parser.add_argument("--hnsw-M", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--shg-M", type=int, default=48)
    parser.add_argument("--shg-ef-construction", type=int, default=80)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    for dataset in args.datasets:
        times = {"build": {}, "search": {}}
        for algorithm in ("hnsw", "shg"):
            M = getattr(args, f"{algorithm}_M")
            ef_construction = getattr(args, f"{algorithm}_ef_construction")
            path = OBSERVATIONS_DIR / f"{dataset}_{algorithm}_performance.csv"
            builds = matching_rows(path, "build", M, args.ef_search,
                                   args.k, args.threads, ef_construction, args.seed)
            searches = matching_rows(path, "search", M, args.ef_search,
                                     args.k, args.threads, ef_construction, args.seed)
            build = builds[-1]
            label = f"{algorithm.upper()}\nM={M}, efc={ef_construction}"
            times["build"][label] = float(
                build["total_build_seconds"] or build["seconds"]
            )
            times["search"][label] = median(
                float(row["seconds"]) for row in searches
            )

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
        colors = ("#4472C4", "#ED7D31")
        for axis, phase, title in zip(
            axes, ("build", "search"), ("Index build", "Search")
        ):
            values = times[phase]
            bars = axis.bar(list(values), list(values.values()), color=colors,
                            width=0.58)
            axis.set_title(title)
            axis.set_ylabel("Time (seconds)")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.bar_label(bars, fmt="%.3g s", padding=4)
            axis.margins(y=0.18)
        fig.suptitle(
            f"{dataset.title()}: build and search time (ef_search={args.ef_search}, k={args.k})"
        )
        output = OBSERVATIONS_DIR / f"{dataset}_build_search_time.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
