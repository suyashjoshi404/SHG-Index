#!/usr/bin/env python3
"""Plot recall against latency for recall-sorted, equal-sized query groups."""

import argparse
import csv
from pathlib import Path
from statistics import fmean


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OBSERVATIONS_DIR = EXPERIMENT_DIR / "observations"


def grouped_points(dataset, algorithm, M, ef_construction, ef_search, k, group_size,
                   num_groups=None, repeat_prefix=None):
    config_dir = (EXPERIMENT_DIR / "results" / dataset / algorithm
                  / f"M{M}_efc{ef_construction}_ef{ef_search}")
    path = config_dir / "per_query_latency.csv"
    rows = []
    if repeat_prefix is None and path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    query_rows = [
        row for row in rows
        if row["dataset"] == dataset
        and row["algorithm"] == algorithm
        and row["layer"] == "0"
        and row["M"] == str(M)
        and row["ef_search"] == str(ef_search)
        and row["k"] == str(k)
    ]
    if not query_rows:
        prefix = "repeat" if repeat_prefix is None else repeat_prefix
        repeats = sorted(config_dir.glob(f"{prefix}*_layer_stats.csv"))
        by_query = {}
        expected_ids = None
        for repeat in repeats:
            with repeat.open(newline="") as handle:
                matching = [row for row in csv.DictReader(handle)
                            if row["layer"] == "0" and row["k"] == str(k)
                            and row["dataset"] == dataset
                            and row["algorithm"] == algorithm
                            and row["M"] == str(M)
                            and row["ef_search"] == str(ef_search)]
            if not matching:
                continue
            ids = {row["query_id"] for row in matching}
            if len(ids) != len(matching) or (expected_ids is not None and ids != expected_ids):
                raise ValueError(f"Inconsistent query IDs in {repeat}")
            expected_ids = ids
            for row in matching:
                by_query.setdefault(row["query_id"], []).append(row)
        for query_id, observations in by_query.items():
            if len({row["recall_at_k"] for row in observations}) != 1:
                raise ValueError(f"Recall differs across repeats for query {query_id}")
            row = dict(observations[0])
            row["query_latency_us"] = fmean(float(r["query_latency_us"]) for r in observations)
            query_rows.append(row)
        if not query_rows:
            raise ValueError(f"No k={k} per-query rows found for {path.stem}")
        print(f"{dataset} {algorithm} M{M}: averaged {len(query_rows)} queries "
              f"over {len(next(iter(by_query.values())))} repeats")
    query_rows.sort(key=lambda row: (float(row["recall_at_k"]), int(row["query_id"])))
    if num_groups is not None:
        if len(query_rows) % num_groups or num_groups > len(query_rows):
            raise ValueError(f"{dataset}: {len(query_rows)} queries cannot form "
                             f"{num_groups} equal-sized nonempty groups")
        group_size = len(query_rows) // num_groups
    if len(query_rows) % group_size:
        raise ValueError(
            f"{dataset} has {len(query_rows)} queries, which is not divisible by "
            f"the requested group size {group_size}"
        )
    points = []
    for start in range(0, len(query_rows), group_size):
        group = query_rows[start:start + group_size]
        points.append((
            fmean(float(row["recall_at_k"]) for row in group),
            fmean(float(row["query_latency_us"]) for row in group) / 1000.0,
        ))
    return points, group_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("sift", "enron"),
                        default=("sift", "enron"))
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument("--group-size", type=int, default=20)
    grouping.add_argument("--num-groups", type=int,
                          help="Equal-sized groups per curve, independent of dataset size")
    parser.add_argument("--comparison", choices=("all", "configured"), default="all")
    parser.add_argument("--repeat-prefix", help="Use only this repeat file prefix, e.g. same_params_repeat")
    parser.add_argument("--hnsw-M", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--shg-M", type=int, default=48)
    parser.add_argument("--shg-ef-construction", type=int, default=80)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if args.num_groups is not None and args.num_groups <= 0:
        parser.error("--num-groups must be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # SHG.pdf, Section 5.3 / Figure 5: recall-sorted equal-sized subsets,
    # mean time on x, mean recall on y; red open circles for SHG and black
    # open diamonds for HNSW, joined with dash-dot lines, without a grid.
    # The paper does not state the subset size. Use the existing default 20.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.linewidth": 0.6, "axes.edgecolor": "0.5",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    comparisons = [
        ("Same build parameters: M=32, efC=128", (32, 128), (32, 128)),
        ("Same build parameters: M=48, efC=80", (48, 80), (48, 80)),
        ("HNSW M={}/efC={}; SHG M={}/efC={}".format(
            args.hnsw_M, args.hnsw_ef_construction,
            args.shg_M, args.shg_ef_construction),
         (args.hnsw_M, args.hnsw_ef_construction),
         (args.shg_M, args.shg_ef_construction)),
    ]
    if args.comparison == "configured":
        comparisons = comparisons[-1:]
    fig, axes = plt.subplots(len(comparisons), len(args.datasets), squeeze=False,
                             figsize=(5 * len(args.datasets), 3.5 * len(comparisons)))
    exported = []
    for row_index, (description, hnsw_config, shg_config) in enumerate(comparisons):
        for col_index, dataset in enumerate(args.datasets):
            ax = axes[row_index, col_index]
            for algorithm, config, color, marker in (
                ("shg", shg_config, "red", "o"),
                ("hnsw", hnsw_config, "black", "D"),
            ):
                M, ef_construction = config
                points, group_size = grouped_points(
                    dataset, algorithm, M, ef_construction,
                    args.ef_search, args.k, args.group_size, args.num_groups,
                    args.repeat_prefix)
                # Preserve recall order even when measured time is nonmonotonic.
                ax.plot([p[1] for p in points], [p[0] for p in points],
                        color=color, marker=marker, markerfacecolor="none",
                        markeredgewidth=0.8, markersize=5,
                        linestyle="-.", linewidth=0.8, label=algorithm.upper())
                for group_id, (recall, latency_ms) in enumerate(points):
                    exported.append((row_index + 1, dataset, algorithm, M,
                                     ef_construction, args.ef_search, args.k,
                                     group_size, group_id, recall, latency_ms))
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Recall")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_ylim(max(0, ax.get_ylim()[0]), min(1.02, ax.get_ylim()[1]))
            ax.grid(False)
            ax.legend(loc="best", fontsize=9, framealpha=0.85,
                      borderpad=0.3, labelspacing=0.2, handlelength=1.8)
            letter = chr(ord("a") + row_index * len(args.datasets) + col_index)
            dataset_label = "SIFT1M" if dataset == "sift" else "Enron"
            ax.set_title(f"({letter}) Performance on {dataset_label}",
                         fontfamily="DejaVu Serif", fontweight="bold", fontsize=11,
                         pad=24)
            ax.text(0.5, 1.015, description, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Search performance (k={args.k}, efSearch={args.ef_search})",
                 fontfamily="DejaVu Serif", fontweight="bold", fontsize=13)
    grouping_label = (f"{args.num_groups} equal-sized recall-sorted groups per curve"
                      if args.num_groups is not None
                      else f"Recall-sorted groups of {args.group_size} queries")
    fig.text(0.5, 0.012,
             f"{grouping_label}; mean recall and mean time per group.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3, w_pad=2)
    OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    grouping_suffix = (f"ng{args.num_groups}" if args.num_groups is not None
                       else f"g{args.group_size}")
    if args.comparison == "configured" and (args.hnsw_M, args.hnsw_ef_construction) == (args.shg_M, args.shg_ef_construction):
        grouping_suffix += f"_same_M{args.hnsw_M}_efc{args.hnsw_ef_construction}"
    stem = OBSERVATIONS_DIR / (
        f"hnsw_shg_paper_recall_latency_ef{args.ef_search}_k{args.k}_{grouping_suffix}"
    )
    for suffix in ("png", "pdf", "svg"):
        output = stem.with_suffix("." + suffix)
        fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
        print(output)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("comparison", "dataset", "algorithm", "M", "ef_construction",
                         "ef_search", "k", "group_size", "group_id",
                         "mean_recall_at_k", "mean_query_latency_ms"))
        writer.writerows(exported)
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
