# Native HNSW and SHG profiling

This experiment builds and profiles SIFT1M or Enron with native
`hnswlib::HierarchicalNSW<float>` (HNSW) and `hnswlib::HEDS<float>` (SHG).
The same C++ driver runs both algorithms sequentially. Python scripts launch
it; FAISS and NumPy are not needed. Run all commands from the repository root.

## Build

You need a C++17 compiler, CMake 3.16+, and Python 3.8+. On Ubuntu/Debian:

```bash
sudo apt install build-essential cmake python3 numactl
```

```bash
cmake -S experiments/profiling -B build/profiling -DCMAKE_BUILD_TYPE=Release
cmake --build build/profiling -j2
```

The driver uses one thread and builds with `-O3` and `-march=native` when
supported. Pass `-DPROFILING_NATIVE=OFF` to CMake for a portable binary.
OpenMP and fast-math are disabled. Rebuild after changing headers or the
driver. Run `ctest --test-dir build/profiling --output-on-failure` to check the
driver with small fixtures, or set `-DBUILD_TESTING=OFF` to build only the driver.

## Prepare data

The default layout under `experiments/profiling/data/` is:

```text
sift/sift_base.fvecs
sift/sift_query.fvecs
sift/sift_groundtruth.ivecs
enron/enron.data_new
enron/enron_query.fvecs
enron/enron_groundtruth.ivecs
```

The `data` directory may be a symlink. For another location, pass
`--data-root PATH` pointing to a directory containing `sift/` or `enron/`.
Base vectors use fvecs (an `int32` dimension and float32 values per record)
or `data_new` (uint32 element byte size, vector count, and dimension, then
float32 values). Queries use fvecs; ground truth uses ivecs (record width and
uint32 zero-based IDs). Files must be little-endian on a little-endian host.

Check headers and file sizes before a large build:

```bash
build/profiling/hnsw_profile --check-data --dataset sift
build/profiling/hnsw_profile --check-data --dataset enron
```

Full loading also checks record headers, lengths, and dimensions.

## Build indexes and profile queries

```bash
python3 experiments/profiling/scripts/build_hnsw_index.py --dataset sift --M 16 --ef-construction 200 --seed 100 --force
python3 experiments/profiling/scripts/profile_hnsw_query.py --dataset sift --M 16 --k 10 --ef-search 64
python3 experiments/profiling/scripts/build_shg_index.py --dataset sift --M 48 --ef-construction 80 --seed 100 --force
python3 experiments/profiling/scripts/profile_shg_query.py --dataset sift --M 48 --k 10 --ef-search 64
```

Use `--dataset enron` for Enron. Both algorithms require `--threads 1`;
search requires `ef-search >= k`, with k no greater than the base count or
ground-truth width. Defaults are HNSW `M=16`, `ef-construction=200`; SHG
`M=48`, `ef-construction=80`; and `seed=100`, `k=10`, `ef-search=64`,
`warmup=500`.

Indexes live in `experiments/profiling/indexes/<dataset>/`. SHG needs both
its `.index` and companion `.index.shg` file. Searches load saved indexes.
Builds replace an index only with `--force`. The default filename includes M
but not construction breadth or seed; use `--index-file` to keep those graphs
separate. Scripts accept `--data-root` and `--executable` for custom paths.

For repeated measurements, output files, plots, counter definitions, and
comparison rules, see [BENCHMARKING.md](BENCHMARKING.md).
