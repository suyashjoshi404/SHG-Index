#!/usr/bin/env python3
"""Plot per-layer search work and SHG traversal from a matched-parameter run."""

import argparse
import csv
from pathlib import Path


DATASETS = (("sift", "SIFT1M"), ("enron", "Enron"))
ALGORITHMS = (("hnsw", "HNSW"), ("shg", "SHG"))
REPEATS = range(1, 6)


def layer_means(observations_root, dataset, algorithm, ef):
    totals = {}
    query_total = 0
    expected_layers = None
    for repeat in REPEATS:
        path = (observations_root / dataset / algorithm / f"ef{ef}_k20"
                / f"repeat{repeat}_layer_averages.csv")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        layers = {int(row["layer"]) for row in rows}
        if not rows or layers != set(range(len(rows))) or (expected_layers is not None and layers != expected_layers):
            raise ValueError(f"Missing, duplicate, or inconsistent layers in {path}")
        expected_layers = layers
        counts = {int(row["num_queries"]) for row in rows}
        if len(counts) != 1 or any(
            row["dataset"] != dataset or row["algorithm"] != algorithm
            or (int(row["M"]), int(row["ef_search"]), int(row["k"])) != (48, ef, 20)
            for row in rows
        ):
            raise ValueError(f"Unexpected parameters in {path}")
        count = counts.pop()
        query_total += count
        for row in rows:
            layer = int(row["layer"])
            entry = totals.setdefault(layer, {"hops": 0.0, "distances": 0.0})
            entry["hops"] += float(row["average_hops"]) * count
            entry["distances"] += float(row["average_distance_computations"]) * count
    return {layer: {key: value / query_total for key, value in totals[layer].items()}
            for layer in sorted(totals)}, query_total // len(REPEATS)


def full_traversal_fraction(results_root, dataset, ef, expected_count):
    fractions = []
    layer_count = None
    for repeat in REPEATS:
        path = (results_root / dataset / "shg" / f"M48_efc80_ef{ef}_k20"
                / f"repeat{repeat}_layer_stats.csv")
        queries = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] != dataset or row["algorithm"] != "shg" or (
                    int(row["M"]), int(row["ef_search"]), int(row["k"])
                ) != (48, ef, 20):
                    raise ValueError(f"Unexpected parameters in {path}")
                count = int(row["num_layers"])
                if layer_count is not None and count != layer_count:
                    raise ValueError(f"Layer count changed in {path}")
                layer_count = count
                query = int(row["query_id"])
                layer = int(row["layer"])
                record = queries.setdefault(query, {})
                if layer in record:
                    raise ValueError(f"Duplicate query/layer in {path}")
                record[layer] = (int(row["hops"]), int(row["levels_skipped"]))
        if len(queries) != expected_count or set(queries) != set(range(expected_count)):
            raise ValueError(f"Unexpected query count or IDs in {path}")
        if any(set(layers) != set(range(layer_count)) for layers in queries.values()):
            raise ValueError(f"Missing query layers in {path}")
        traversed = sum(all(hops > 0 and skipped == 0 for hops, skipped in layers.values())
                        for layers in queries.values())
        fractions.append(traversed / expected_count)
    return sum(fractions) / len(fractions), layer_count


def plot_metric(plt, summary, metric, title, ylabel, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True,
                             layout="constrained")
    colors = {"hnsw": "#24648c", "shg": "#c04b3a"}
    for ax, (dataset, label) in zip(axes, DATASETS):
        layers = sorted(summary[dataset]["hnsw"])
        if layers != sorted(summary[dataset]["shg"]):
            raise ValueError(f"Layer sets differ for {dataset}")
        for offset, (algorithm, name) in zip((-0.18, 0.18), ALGORITHMS):
            values = [summary[dataset][algorithm][layer][metric] for layer in layers]
            if any(value <= 0 for value in values):
                raise ValueError(f"Log scale requires positive {metric} values")
            ax.bar([layer + offset for layer in layers], values, width=0.34,
                   color=colors[algorithm], label=name)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(layers, [str(layer) for layer in layers])
        ax.set_xlabel("Graph layer (0 = base)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel + " per query (log scale)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle(title, fontweight="bold")
    fig.supxlabel("M=48  ·  efConstruction=80  ·  efSearch=64  ·  k=20  ·  5 repeats",
                  fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=220)
    plt.close(fig)


def plot_fraction(plt, fractions, counts, layers, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    colors = ("#24648c", "#c04b3a")
    for i, (dataset, label) in enumerate(DATASETS):
        value = fractions[dataset]
        ax.bar(i, value, width=0.48, color=colors[i])
        ax.text(i, value + 0.018, f"{value:.1%}", ha="center", fontweight="bold")
        ax.text(i, 0.05, f"{counts[dataset]:,} queries/run", ha="center",
                color="white", fontsize=9)
    ax.set_xticks(range(len(DATASETS)), [label for _, label in DATASETS])
    ax.set_ylim(0, 1.09)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
    ax.set_ylabel("Fraction of queries")
    ax.set_title("SHG queries traversing every graph layer", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.supxlabel(f"All {layers} layers had hops and no skipped levels  ·  "
                  "M=48  ·  efConstruction=80  ·  efSearch=64  ·  k=20  ·  5 repeats",
                  fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = Path(__file__).resolve().parents[1]
    parser.add_argument("--observations-root", type=Path,
                        default=experiment / "observations/ef_search_sweep/matched_params")
    parser.add_argument("--results-root", type=Path,
                        default=experiment / "results/ef_search_sweep/matched_params")
    parser.add_argument("--output-dir", type=Path,
                        default=experiment / "observations/ef_search_sweep/matched_params/plots")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})

    summary, counts, fractions, layer_counts = {}, {}, {}, {}
    for dataset, _ in DATASETS:
        summary[dataset] = {}
        for algorithm, _ in ALGORITHMS:
            summary[dataset][algorithm], count = layer_means(
                args.observations_root, dataset, algorithm, 64)
            if dataset in counts and counts[dataset] != count:
                raise ValueError(f"HNSW and SHG query counts differ for {dataset}")
            counts[dataset] = count
        fractions[dataset], layer_counts[dataset] = full_traversal_fraction(
            args.results_root, dataset, 64, counts[dataset])
    if len(set(layer_counts.values())) != 1:
        raise ValueError("Dataset layer counts differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(plt, summary, "hops", "Average hops by graph layer",
                "Average hops", args.output_dir / "average_hops_per_layer")
    plot_metric(plt, summary, "distances", "Average distance calculations by graph layer",
                "Average distance calculations",
                args.output_dir / "average_distances_per_layer")
    plot_fraction(plt, fractions, counts, next(iter(layer_counts.values())),
                  args.output_dir / "shg_all_layers_traversed_fraction")
    for dataset, _ in DATASETS:
        print(f"{dataset}: {counts[dataset]} queries/run; all-layer SHG fraction "
              f"{fractions[dataset]:.4f}")


if __name__ == "__main__":
    main()
