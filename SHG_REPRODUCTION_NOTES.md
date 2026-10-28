# SHG reproduction notes

The current efSearch sweep uses M=48, efConstruction=80, k=20, and five
repeats. Native HNSW achieves better recall at lower latency than SHG in
these runs. This does not yet reproduce the paper's performance claims.

## Problems found

- **Shortcuts were disabled in the recorded runs.** The saved shortcut tables contain
  58 entries for SIFT1M and 13 for Enron. `hnswlib/heds.h` disables shortcut
  lookup only for empty tables now; the previous barrier was 100 entries.
  Statistics show zero skipped levels throughout the recorded sweep, which
  predates this change. The sweep needs rerunning to measure its effect.
- **Screening adds substantial work.** At efSearch=64, SHG performs about
  1,302 compressed distance checks per SIFT query and 990 per Enron query at
  the base layer, while doing roughly as many full distance calculations as
  HNSW. Screening also runs before the visited check, allowing repeated work.
- **Pruning can discard untested neighbors.** A failed compressed screening
  check uses `break`, skipping the rest of the current node's neighbor list.
  This is a possible cause of lower recall; its effect needs an ablation.
- **Shortcut construction uses integer division.** The layer-count ratio and
  `1/value` expressions in `buildShortcuts()` can become zero, undermining
  the density calculation. Whether this causes the small tables needs testing.

## Differences from the paper and examples

- We use **SIFT1M**, while the paper evaluates **SIFT100M**. M=48,
  efConstruction=80, and k=20 match the paper's stated settings.
- `HEDS` in `hnswlib/heds.h` implements SHG. Our HNSW baseline uses the separate
  native `HierarchicalNSW` class. Matching parameters does not produce identical
  graphs: SHG uses compressed distances during construction.
- `examples/cpp/example_search.cpp` is incomplete as an HNSW comparison. It
  creates an HEDS object, inserts base data only for `"heds"`, and always
  calls shortcut search. Passing `"hnsw"` does not select a working vanilla
  HNSW benchmark there.
- SHG base search uses `max(efSearch, k)` again to make efSearch control
  search breadth. The released implementation passes `k`; measurements of
  that behavior are preserved separately in the `restored_k` run.
- Our runner saves/loads SHG representations and shortcuts and resets touched
  scratch entries inside each search. The original example builds in memory
  and clears the entire scratch buffer outside its per-query timer.
- Our curve plots full-query-set recall against mean latency across five
  repeats, with one point per efSearch. Section 5.3 describes recall-ordered
  query subsets; the released example does not contain the figure's plotting
  procedure. We cannot confirm that the paper used our sweep method.

## Next checks

Test pruning changes separately, validate shortcut arithmetic and activation,
and measure each change before repeating the full benchmark. These findings
identify likely causes, but do not establish a complete explanation of the
performance gap.

Sources: [SHG.pdf](SHG.pdf), [SHG implementation](hnswlib/heds.h),
[original example](examples/cpp/example_search.cpp), and sweep observations in
`experiments/profiling/observations/ef_search_sweep/matched_params/`.
