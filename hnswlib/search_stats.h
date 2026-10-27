#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hnswlib {

// Query-local instrumentation. A hop is one expanded node / neighbor-list
// scan, as in FAISS. Count actual distance evaluations, including seeds;
// legacy metric_distance_computations instead counts adjacency entries.
// Counters accumulate until reset(). Use one instance per searching thread.
struct HNSWStats {
    struct Layer {
        uint64_t hops = 0;
        uint64_t distance_computations = 0;
        uint64_t full_distance_computations = 0;
        uint64_t compressed_distance_computations = 0;
        uint64_t seed_distance_computations = 0;
        uint64_t pruned_neighbors = 0;
        uint64_t levels_skipped = 0;
    };

    uint64_t nhops = 0;
    uint64_t ndis = 0;
    std::vector<Layer> layers;

    explicit HNSWStats(size_t count = 0) : layers(count) {}

    void ensure_levels(size_t count) {
        if (layers.size() < count) layers.resize(count);
    }

    void add_hop(size_t level) {
        ++nhops;
        ++layers[level].hops;
    }

    void add_distance(size_t level, bool compressed = false, bool seed = false) {
        ++ndis;
        auto& row = layers[level];
        ++row.distance_computations;
        if (compressed) ++row.compressed_distance_computations;
        else ++row.full_distance_computations;
        if (seed) ++row.seed_distance_computations;
    }

    uint64_t get_nhops_at_level(int level) const {
        return level >= 0 && static_cast<size_t>(level) < layers.size()
            ? layers[level].hops : 0;
    }

    uint64_t get_ndis_at_level(int level) const {
        return level >= 0 && static_cast<size_t>(level) < layers.size()
            ? layers[level].distance_computations : 0;
    }

    void reset() {
        nhops = ndis = 0;
        std::fill(layers.begin(), layers.end(), Layer{});
    }

    void combine(const HNSWStats& other) {
        ensure_levels(other.layers.size());
        nhops += other.nhops;
        ndis += other.ndis;
        for (size_t i = 0; i < other.layers.size(); ++i) {
            auto& a = layers[i];
            const auto& b = other.layers[i];
            a.hops += b.hops;
            a.distance_computations += b.distance_computations;
            a.full_distance_computations += b.full_distance_computations;
            a.compressed_distance_computations += b.compressed_distance_computations;
            a.seed_distance_computations += b.seed_distance_computations;
            a.pruned_neighbors += b.pruned_neighbors;
            a.levels_skipped += b.levels_skipped;
        }
    }
};

using SHGSearchStats = HNSWStats;

} // namespace hnswlib
