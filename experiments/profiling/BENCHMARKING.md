# Reproducible HNSW and SHG profiling

## Recall–latency curve: sweep efSearch with fixed build parameters

Build the driver as shown in [README.md](README.md), then run from the
repository root. Fix `M=48`,
`ef_construction=80`, seed `100`, `k=20`, one thread, and 500 warm-up queries
for both algorithms. Sweep `ef_search` through
`20 24 32 48 64 96 128 192 256 512` on each saved index and query set. Each
value is at least k. Build each dataset/algorithm index once and run five
searches at each breadth: four builds and 200 search processes total.

This standalone sweep writes to a unique results and observations directory.
It rebuilds its indexes with `--force`. To reuse them later, omit the build
command and keep the index paths and construction parameters unchanged. On the
inspected host, CPU 2 is on NUMA node 0 and has SMT sibling CPU 50; keep both
idle. Choose valid CPU and NUMA IDs on other hosts.

```bash
(
set -e
scripts=experiments/profiling/scripts
data_root=experiments/profiling/data
run_id=$(date -u +%Y%m%dT%H%M%S%N)
results_root="experiments/profiling/results/ef_search_sweep/${run_id}"
observations_root="experiments/profiling/observations/ef_search_sweep/${run_id}"
M=48
ef_construction=80
k=20

for dataset in sift enron; do
  for algorithm in hnsw shg; do
    index_file="experiments/profiling/indexes/${dataset}/${dataset}_${algorithm}_sweep_M${M}_efc${ef_construction}_seed100.index"
    observation_dir="${observations_root}/${dataset}/${algorithm}"
    mkdir -p "$observation_dir"

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      numactl --physcpubind=2 --membind=0 \
      python3 "$scripts/build_${algorithm}_index.py" \
      --dataset "$dataset" --data-root "$data_root" \
      --M "$M" --ef-construction "$ef_construction" --seed 100 --threads 1 \
      --index-file "$index_file" --force \
      --performance-output "${observation_dir}/build_performance.csv"

    for ef_search in 20 24 32 48 64 96 128 192 256 512; do
      result_dir="${results_root}/${dataset}/${algorithm}/M${M}_efc${ef_construction}_ef${ef_search}_k${k}"
      search_observation_dir="${observation_dir}/ef${ef_search}_k${k}"
      mkdir -p "$result_dir" "$search_observation_dir"

      for repeat in 1 2 3 4 5; do
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
          numactl --physcpubind=2 --membind=0 \
          python3 "$scripts/profile_${algorithm}_query.py" \
          --dataset "$dataset" --data-root "$data_root" \
          --M "$M" --ef-construction "$ef_construction" \
          --k "$k" --ef-search "$ef_search" --threads 1 --warmup 500 \
          --index-file "$index_file" \
          --output "${result_dir}/repeat${repeat}_layer_stats.csv" \
          --performance-output "${search_observation_dir}/repeat${repeat}_performance.csv" \
          --layer-averages-output "${search_observation_dir}/repeat${repeat}_layer_averages.csv"
      done
    done
  done
done
echo "Sweep results: $results_root"
echo "Sweep observations: $observations_root"
)
```

Plot one point per `ef_search` value for each dataset and algorithm. Use full
query-set `recall_at_k` and mean `average_latency_us / 1000` across five
performance rows. Plot milliseconds against recall, connect points in
increasing `ef_search` order, and compare algorithms at matched recall. Verify
recall is consistent across repeats before averaging; error bars may show
latency spread. All points use the same complete query set. Do not group
queries by recall for this sweep: `plot_recall_latency.py` instead plots
grouped curves at a fixed `ef_search`.

Generate the sweep plot with Matplotlib installed, passing the observation
directory for the completed run:

```bash
python3 experiments/profiling/scripts/plot_ef_search_sweep.py \
  experiments/profiling/observations/ef_search_sweep/<run_id>
```

The script checks for all ten `ef_search` values and five repeats per
dataset/algorithm. It writes PNG, PDF, SVG, and a CSV of plotted means and
latency spread to that run's `plots/` directory. It reads performance
observations, not instrumented per-query files in `results/`.

Latency is measured in the timed pass with detailed counters disabled.
Per-query layer statistics come from a separate pass and are for diagnostics;
use performance CSVs for the curve. Keep each SHG `.index` beside its
`.index.shg` companion. The sweep measures these implementations' accuracy and
latency; the paper does not say its figures used this sweep.

## SIFT and Enron: two parameter setups, five searches each

Run from the repository root after [README.md](README.md). These commands use
`experiments/profiling/data/`, build each index once with `--force`, then run
five warmed searches. Setup 1 matches HNSW and SHG at `M=48`,
`ef_construction=80`. Setup 2 uses HNSW `M=16`, `ef_construction=200` and SHG
`M=48`, `ef_construction=80`. On the inspected Xeon host, CPU 2 is on NUMA
node 0 and shares a core with CPU 50; keep both idle during benchmarks.

```bash
scripts=experiments/profiling/scripts
data_root=experiments/profiling/data

# Setup 1: same build parameters for HNSW and SHG (M=48, ef_construction=80).
for dataset in sift enron; do
  for algorithm in hnsw shg; do
    M=48
    ef_construction=80
    setup=same_params
    index_file="experiments/profiling/indexes/${dataset}/${dataset}_${algorithm}_${setup}_M${M}.index"
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      numactl --physcpubind=2 --membind=0 \
      python3 "$scripts/build_${algorithm}_index.py" \
      --dataset "$dataset" --data-root "$data_root" \
      --M "$M" --ef-construction "$ef_construction" --seed 100 --threads 1 \
      --index-file "$index_file" --force

    for repeat in 1 2 3 4 5; do
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        numactl --physcpubind=2 --membind=0 \
        python3 "$scripts/profile_${algorithm}_query.py" \
        --dataset "$dataset" --data-root "$data_root" \
        --M "$M" --ef-construction "$ef_construction" --k 20 --ef-search 64 --threads 1 --warmup 500 \
        --index-file "$index_file" \
        --output "experiments/profiling/results/${dataset}/${algorithm}/M${M}_efc${ef_construction}_ef64/${setup}_repeat${repeat}_layer_stats.csv"
    done
  done
done

# Setup 2: use the algorithm-specific build parameters.
for dataset in sift enron; do
  for algorithm in hnsw shg; do
    setup=different_params
    if [ "$algorithm" = hnsw ]; then M=16; ef_construction=200
    else M=48; ef_construction=80; fi
    index_file="experiments/profiling/indexes/${dataset}/${dataset}_${algorithm}_${setup}_M${M}.index"
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      numactl --physcpubind=2 --membind=0 \
      python3 "$scripts/build_${algorithm}_index.py" \
      --dataset "$dataset" --data-root "$data_root" \
      --M "$M" --ef-construction "$ef_construction" --seed 100 --threads 1 \
      --index-file "$index_file" --force

    for repeat in 1 2 3 4 5; do
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        numactl --physcpubind=2 --membind=0 \
        python3 "$scripts/profile_${algorithm}_query.py" \
        --dataset "$dataset" --data-root "$data_root" \
        --M "$M" --ef-construction "$ef_construction" --k 20 --ef-search 64 --threads 1 --warmup 500 \
        --index-file "$index_file" \
        --output "experiments/profiling/results/${dataset}/${algorithm}/M${M}_efc${ef_construction}_ef64/${setup}_repeat${repeat}_layer_stats.csv"
    done
  done
done
```

To run five more searches without rebuilding, rerun a `repeat` loop with the
same variables set. Measurements append to the performance and layer-average
CSVs. The specified `--output` paths keep repeats 1–5 separate; rerunning the
loop replaces those result files. Report the mean and spread of the five
measurements.

The runner is serial and does not use BLAS. Environment settings make the
single-thread policy explicit; CPU and memory binding set placement. On other
machines, inspect the topology and choose valid CPU and NUMA IDs.

## Comparing measurements

- Keep machine, compiler options, dataset, insertion/query order, seed, and
  parameters fixed within each setup. Setup 1 matches HNSW and SHG at M=48 and
  ef_construction=80; Setup 2 uses HNSW M=16/ef_construction=200 and SHG
  M=48/ef_construction=80. Record the commit and local changes. Both builds are
  sequential.
- Build once and reuse indexes for search repetitions. Set k and search breadth
  explicitly; compare latency at matched recall as well as matched ef.
- Warm up (500 queries by default), run at least five timed searches, and report
  the mean and spread (for example, standard deviation or min/max). Observation
  CSVs append; per-query result files are replaced unless `--output` changes.
- To compare build time, repeat builds with `--force` and retain every row.
  Report graph time, shortcut time, and their sum; shortcut construction is
  part of SHG's build cost.
- Keep the machine quiet. CPU affinity does not reserve a core, so avoid work
  on the selected CPU and its SMT sibling. Keep CPU frequency, turbo, cache,
  and NUMA policies consistent and record any uncontrolled settings.

Useful inspection commands:

```bash
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware
uptime
ps -eo pid,psr,pcpu,comm --sort=-pcpu | head
```

These commands pin a process but do not isolate its CPU. Both CLIs reject
`--threads` values other than 1. For a latency/recall curve, search the same
saved index at different `--ef-search` values. HNSW and SHG build separate
graphs, so this compares complete implementations.

## Outputs and timing

Paths below are relative to `experiments/profiling/`:

| File | Write mode | Contents |
| --- | --- | --- |
| `observations/<dataset>_<algorithm>_performance.csv` | Append | Build and search time, latency, QPS, recall, parameters |
| `observations/<dataset>_<algorithm>_layer_averages.csv` | Append | Mean hops, full/compressed/seed distances, pruning, recall, QPS |
| `results/<dataset>/<algorithm>/M<M>_efc<ef-construction>_ef<ef-search>/layer_stats.csv` | Replace | Per-query, per-layer counters and latency |

Build time includes allocation, base-vector loading, and insertion. SHG
shortcut construction is timed separately; `total_build_seconds` sums the two.
Header checks and index saving are excluded. Search loads queries, truth, and
SHG query representations before warm-up. The timed pass includes one native
search call and result queue lifetime per query, with detailed counters off.
A separate pass collects recall and layer statistics. Index loading, recall,
and CSV writes are excluded. SHG scratch cleanup is inside each search call.
QPS and latency describe sequential queries.

## Plots

Install Matplotlib, then plot build/search results or grouped recall and
latency from per-query results:

```bash
python3 -m pip install matplotlib
python3 experiments/profiling/scripts/plot_build_search.py
python3 experiments/profiling/scripts/plot_recall_latency.py --group-size 20
```

The grouped plot sorts queries by recall and plots each group's mean recall
against mean latency. It uses layer-0 rows so each query counts once. Group
size must divide the query count; the paper does not specify one. Existing
measurements use `k=10`, while SHG.pdf Figure 5 uses `k=20`. To select k=20
measurements, pass `--k 20 --comparison configured`. For ten groups, pass
`--num-groups 10` instead of `--group-size`; this yields 1,000 queries per
SIFT1M group and 20 per Enron group. Outputs include PNG, PDF, SVG, and CSV.
For repeat files, `--repeat-prefix same_params_repeat` selects that run family;
set both algorithms' `--hnsw-M`, `--shg-M`, `--hnsw-ef-construction`, and
`--shg-ef-construction` to match the saved builds. If the dedicated CSV lacks
the requested k, the plotter averages matching repeat files per query.

To generate separate plots of mean hops per layer, mean distance calculations
per layer, and the fraction of SHG queries traversing every layer, run:

```bash
python3 experiments/profiling/scripts/plot_layer_metrics.py
```

This uses the five matched M48/efConstruction80/efSearch64/k20 repeats for
SIFT1M and Enron. It writes PNG and PDF files to
`observations/ef_search_sweep/matched_params/plots/`. Distance calculations
include full and compressed evaluations. A query counts as traversing every
layer when it has at least one hop and zero skipped levels on each layer.

## Counters and comparison scope

`hnswlib/search_stats.h` provides query-local `HNSWStats` (also named
`SHGSearchStats`) with `nhops`, `ndis`, per-layer getters, `reset()`, and
`combine(other)`. Use one stats object per searching thread. A hop is an
expanded node or adjacency-list scan, including empty and repeated scans.
`ndis` counts actual vector-distance calls, including SHG compressed screening
and seed distances, and equals the sum of per-layer counts. Full and
compressed counts partition those calls; seed counts are a subset. FAISS's
ordinary HNSW `ndis` omits the first seed distance, so raw totals differ.
`pruned_neighbors` counts skipped neighbor occurrences; `levels_skipped`
counts upper layers bypassed by shortcuts. Legacy
`metric_distance_computations` counts adjacency entries, not distance calls.

Use `searchKnn(query_vector, query_id, k, nullptr, &stats)` for native HNSW
and `searchKnnShortcuts(query, nullptr, &stats)` for SHG. HEDS also has
`searchKnnHNSW`, but the benchmark's HNSW baseline uses `HierarchicalNSW`.
Matching M, seed, order, and ef does not guarantee identical graphs because
construction distances can differ. The benchmark compares complete native
implementations.

SHG here differs from upstream: shortcut search uses `max(ef_, k)` at the
base layer; density excludes unused capacity; each search clears touched
scratch entries; `.index.shg` saves representations and shortcuts; and HEDS
ordinary HNSW search no longer prints an entry-point line. Other SHG formulas
and pruning remain. Base screening needs dimension at least 64; both supplied
datasets meet this requirement. Indexes are native to each class and should
be loaded on the same architecture used to build them.
