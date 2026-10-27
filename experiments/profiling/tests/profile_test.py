"""Small end-to-end fixtures; no external datasets or Python packages needed."""

import csv
import hashlib
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

EXECUTABLE = Path(sys.argv[1]).resolve()
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(*args, ok=True):
    result = subprocess.run([str(arg) for arg in args], capture_output=True, text=True)
    if (result.returncode == 0) != ok:
        raise AssertionError(f"Unexpected exit {result.returncode}: {args}\n{result.stdout}\n{result.stderr}")
    return result


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(None not in row and all(value is not None for value in row.values()) for row in rows)
    return rows


def write_fvecs(path, vectors):
    with path.open("wb") as handle:
        for vector in vectors:
            handle.write(struct.pack("<i" + "f" * len(vector), len(vector), *vector))


def main():
    with tempfile.TemporaryDirectory(prefix="shg-profile-test-") as temporary:
        root = Path(temporary)
        rng = random.Random(7)
        dimension, count, nq, k = 128, 192, 9, 5
        vectors = [[rng.uniform(-1, 1) for _ in range(dimension)] for _ in range(count)]
        queries = [[rng.uniform(-1, 1) for _ in range(dimension)] for _ in range(nq)]
        truth = [sorted(range(count), key=lambda i: sum((x - y) ** 2 for x, y in zip(query, vectors[i])))[:k]
                 for query in queries]
        for dataset in ("sift", "enron"):
            data_dir = root / dataset
            data_dir.mkdir()
            if dataset == "sift":
                write_fvecs(data_dir / "sift_base.fvecs", vectors)
            else:
                with (data_dir / "enron.data_new").open("wb") as handle:
                    handle.write(struct.pack("<III", 4, count, dimension))
                    for vector in vectors:
                        handle.write(struct.pack("<" + "f" * dimension, *vector))
            write_fvecs(data_dir / f"{dataset}_query.fvecs", queries)
            with (data_dir / f"{dataset}_groundtruth.ivecs").open("wb") as handle:
                for labels in truth:
                    handle.write(struct.pack("<" + "I" * (k + 1), k, *labels))
            run(EXECUTABLE, "--check-data", "--dataset", dataset, "--data-root", root, "--k", k)
            for algorithm in ("hnsw", "shg"):
                prefix = root / f"{dataset}_{algorithm}"
                index = prefix.with_suffix(".index")
                performance = prefix.with_suffix(".performance.csv")
                result = prefix.with_suffix(".results.csv")
                averages = prefix.with_suffix(".averages.csv")
                common = ["--dataset", dataset, "--data-root", root, "--M", 4,
                          "--index-file", index, "--performance-output", performance,
                          "--executable", EXECUTABLE]
                build = [sys.executable, SCRIPTS / f"build_{algorithm}_index.py", *common,
                         "--ef-construction", 32, "--seed", 100]
                search = [sys.executable, SCRIPTS / f"profile_{algorithm}_query.py", *common,
                          "--k", k, "--ef-search", 16, "--warmup", 3,
                          "--output", result, "--layer-averages-output", averages]
                run(*build)
                assert "--force" in run(*build, ok=False).stderr
                run(*build, "--force")
                saved = [index] + ([Path(str(index) + ".shg")] if algorithm == "shg" else [])
                digests = [hashlib.sha256(path.read_bytes()).digest() for path in saved]
                # Search succeeds with base data unavailable: the saved graph is reused.
                base = data_dir / ("sift_base.fvecs" if dataset == "sift" else "enron.data_new")
                hidden = base.with_suffix(".hidden")
                base.rename(hidden)
                try:
                    run(*search)
                    first = read_csv(result)
                    run(*search)
                    assert first == read_csv(result), "Repeated search changed graph work or recall"
                finally:
                    hidden.rename(base)
                assert digests == [hashlib.sha256(path.read_bytes()).digest() for path in saved]
                rows = read_csv(performance)
                assert len(rows) == 4 and [row["phase"] for row in rows] == ["build", "build", "search", "search"]
                for row in rows:
                    assert row["threads"] == "1" and row["algorithm"] == algorithm
                    assert row["dimension"] == str(dimension) and row["ef_construction"] == "32"
                    if row["phase"] == "build":
                        assert row["seed"] == "100" and row["count"] == str(count)
                        assert abs(float(row["total_build_seconds"]) - float(row["seconds"]) -
                                   float(row["shortcut_build_seconds"])) < 2e-8
                    else:
                        assert row["count"] == str(nq) and row["warmup"] == "3"
                        assert 0 <= float(row["recall_at_k"]) <= 1 and float(row["queries_per_second"]) > 0
                layers = int(first[0]["num_layers"])
                assert len(first) == nq * layers
                for q in range(nq):
                    group = [row for row in first if row["query_id"] == str(q)]
                    assert len(group) == layers
                    assert sum(int(row["hops"]) for row in group) == int(group[0]["total_query_hops"])
                    assert sum(int(row["distance_computations"]) for row in group) == int(group[0]["total_query_distance_computations"])
                    for row in group:
                        assert int(row["distance_computations"]) == int(row["full_distance_computations"]) + int(row["compressed_distance_computations"])
                assert len(read_csv(averages)) == layers * 2
                run(*search, "--threads", 2, ok=False)
                run(*search, "--M", 8, ok=False)
                if algorithm == "shg":
                    companion = Path(str(index) + ".shg")
                    companion.rename(companion.with_suffix(".hidden"))
                    run(*search, ok=False)
        print("Both datasets and algorithms: build/search wrappers, forced rebuilds, CSVs, repeatability and error handling passed")


if __name__ == "__main__":
    main()
