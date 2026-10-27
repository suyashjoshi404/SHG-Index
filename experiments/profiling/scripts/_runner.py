"""Shared standard-library CLI for the native HNSW/SHG profiler."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main(algorithm, phase):
    parser = argparse.ArgumentParser(description=f"{phase.title()} {algorithm.upper()} with the native profiler")
    parser.add_argument("--dataset", choices=("sift", "enron"), default="sift")
    parser.add_argument("--M", type=int)
    parser.add_argument("--ef-construction", type=int)
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    parser.add_argument("--data-root", help="Directory containing sift/ and enron/")
    parser.add_argument("--index-file")
    parser.add_argument("--performance-output")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/profiling/hnsw_profile")
    if phase == "build":
        parser.add_argument("--seed", type=int, default=100)
        parser.add_argument("--data-file")
        parser.add_argument("--force", action="store_true", help="Replace an existing index")
    else:
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--ef-search", type=int, default=64)
        parser.add_argument("--warmup", type=int, default=500)
        parser.add_argument("--query-file")
        parser.add_argument("--groundtruth-file")
        parser.add_argument("--output")
        parser.add_argument("--layer-averages-output")
    args = parser.parse_args()
    args.M = args.M if args.M is not None else (16 if algorithm == "hnsw" else 48)
    args.ef_construction = (
        args.ef_construction if args.ef_construction is not None
        else (200 if algorithm == "hnsw" else 80)
    )
    experiment_root = ROOT / "experiments/profiling"
    args.data_root = args.data_root or str(experiment_root / "data")
    args.index_file = args.index_file or str(
        experiment_root / "indexes" / args.dataset
        / f"{args.dataset}_{algorithm}_M{args.M}.index"
    )
    args.performance_output = args.performance_output or str(
        experiment_root / "observations" / f"{args.dataset}_{algorithm}_performance.csv"
    )
    if phase == "search":
        args.output = args.output or str(
            experiment_root / "results" / args.dataset / algorithm
            / f"M{args.M}_efc{args.ef_construction}_ef{args.ef_search}"
            / "layer_stats.csv"
        )
        args.layer_averages_output = args.layer_averages_output or str(
            experiment_root / "observations"
            / f"{args.dataset}_{algorithm}_layer_averages.csv"
        )
    executable = args.executable.expanduser().resolve()
    if not executable.is_file():
        parser.error(f"Runner not found: {executable}. Build it using experiments/profiling/README.md")
    command = [str(executable), "--algorithm", algorithm,
               "--build-only" if phase == "build" else "--search-only"]
    for name, value in vars(args).items():
        if name == "executable" or value is None:
            continue
        if name == "force":
            if value:
                command.append("--force")
        else:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    raise SystemExit("Use build_hnsw_index.py, build_shg_index.py, profile_hnsw_query.py or profile_shg_query.py")
