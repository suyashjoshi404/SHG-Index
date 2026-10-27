#include <chrono>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include "hnswlib/hnswlib.h"
#include <filesystem>
#include <random>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Queue>
static bool same_results(Queue a, Queue b) {
    if (a.size() != b.size()) return false;
    while (!a.empty()) {
        if (a.top() != b.top()) return false;
        a.pop();
        b.pop();
    }
    return true;
}

// Count full-distance calls independently of the new instrumentation.
struct CountingSpace : hnswlib::SpaceInterface<float> {
    hnswlib::L2Space l2;
    size_t calls = 0;
    explicit CountingSpace(size_t dimension) : l2(dimension) {}
    static float distance(const void* a, const void* b, const void* context) {
        auto& self = *const_cast<CountingSpace*>(static_cast<const CountingSpace*>(context));
        ++self.calls;
        return self.l2.get_dist_func()(a, b, self.l2.get_dist_func_param());
    }
    size_t get_data_size() override { return l2.get_data_size(); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return distance; }
    void* get_dist_func_param() override { return this; }
};

static void check_stats(const hnswlib::HNSWStats& stats, size_t full_calls) {
    uint64_t full = 0, distances = 0, hops = 0;
    for (size_t i = 0; i < stats.layers.size(); ++i) {
        const auto& row = stats.layers[i];
        require(row.distance_computations == row.full_distance_computations + row.compressed_distance_computations,
                "Full/compressed counts do not partition distance calls");
        require(row.seed_distance_computations <= row.distance_computations, "Invalid seed count");
        require(stats.get_ndis_at_level(i) == row.distance_computations, "Distance getter mismatch");
        require(stats.get_nhops_at_level(i) == row.hops, "Hop getter mismatch");
        full += row.full_distance_computations;
        distances += row.distance_computations;
        hops += row.hops;
    }
    require(full == full_calls, "Counter does not match independently observed distance calls");
    require(distances == stats.ndis && hops == stats.nhops, "Layer totals mismatch");
    require(stats.get_ndis_at_level(-1) == 0 && stats.get_nhops_at_level(1000) == 0, "Invalid layer lookup");
}

struct EvenLabels : hnswlib::BaseFilterFunctor {
    bool operator()(hnswlib::labeltype id) override { return id % 2 == 0; }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass a temporary output directory");
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        constexpr size_t dimension = 128, n = 512, nq = 4, k = 5;
        CountingSpace space(dimension);
        std::mt19937 random(42);
        std::uniform_real_distribution<float> values(-1, 1);
        std::vector<std::vector<float>> vectors(n + nq, std::vector<float>(dimension));
        for (auto& vector : vectors) for (float& x : vector) x = values(random);
        hnswlib::HierarchicalNSW<float> hnsw(&space, n, 4, 32, 100);
        hnswlib::HNSWStats empty;
        require(hnsw.searchKnn(vectors[0].data(), 0, k, nullptr, &empty).empty(), "Empty search");
        require(empty.ndis == 0 && empty.nhops == 0, "Empty index counted work");
        for (size_t i = 0; i < n; ++i) hnsw.addPoint(vectors[i].data(), i);
        hnsw.setEf(32);
        EvenLabels filter;
        for (size_t q = n; q < n + nq; ++q) {
            for (auto* predicate : {static_cast<hnswlib::BaseFilterFunctor*>(nullptr),
                                    static_cast<hnswlib::BaseFilterFunctor*>(&filter)}) {
                const auto baseline = hnsw.searchKnn(vectors[q].data(), q, k, predicate);
                hnswlib::HNSWStats stats;
                space.calls = 0;
                require(same_results(baseline, hnsw.searchKnn(vectors[q].data(), q, k, predicate, &stats)),
                        "HNSW instrumentation changed results");
                check_stats(stats, space.calls);
                require(stats.ndis && stats.nhops, "HNSW did not count work");
                hnswlib::HNSWStats combined;
                combined.combine(stats);
                combined.combine(stats);
                require(combined.ndis == 2 * stats.ndis && combined.nhops == 2 * stats.nhops, "Combine mismatch");
                combined.reset();
                check_stats(combined, 0);
            }
        }
        hnsw.saveIndex((directory / "hnsw.index").string());
        hnswlib::HierarchicalNSW<float> loaded_hnsw(&space, (directory / "hnsw.index").string());
        loaded_hnsw.setEf(32);
        require(same_results(hnsw.searchKnn(vectors[n].data(), 0, k), loaded_hnsw.searchKnn(vectors[n].data(), 0, k)),
                "HNSW save/load changed results");
        // Single-node graph tests both seed evaluations and an empty adjacency list.
        hnswlib::HierarchicalNSW<float> single(&space, 1, 4, 32, 100);
        single.addPoint(vectors[0].data(), 0);
        hnswlib::HNSWStats single_stats;
        space.calls = 0;
        single.searchKnn(vectors[n].data(), 0, 1, nullptr, &single_stats);
        check_stats(single_stats, space.calls);
        require(single_stats.ndis == 2, "Single-node seed evaluations missing");

        Performance performance;
        hnswlib::HEDS<float> shg(&space, dimension, n + nq, 4, 32, 100);
        for (size_t i = 0; i < n; ++i) shg.addDataPoint(vectors[i].data(), i, 1, performance);
        int count = n;
        shg.buildShortcuts(count);
        shg.saveIndex((directory / "shg.index").string());
        shg.saveSearchState((directory / "shg.index.shg").string());
        hnswlib::HEDS<float> loaded(&space, (directory / "shg.index").string());
        loaded.loadSearchState((directory / "shg.index.shg").string(), dimension);
        loaded.resizeIndex(n + nq + 8); // exercises representation reallocation after load
        shg.setEf(32);
        loaded.setEf(32);
        shg.resultsProcessing.assign(n * (shg.maxlevel_ + 1), -1);
        loaded.resultsProcessing.assign(n * (loaded.maxlevel_ + 1), -1);
        for (size_t q = n; q < n + nq; ++q) {
            shg.addDataPoint(vectors[q].data(), q, -1, performance);
            loaded.addDataPoint(vectors[q].data(), q, -1, performance);
            Query query(q, k);
            const auto baseline = shg.searchKnnShortcuts(query);
            hnswlib::HNSWStats stats;
            space.calls = 0;
            require(same_results(baseline, shg.searchKnnShortcuts(query, nullptr, &stats)), "SHG counters changed results");
            check_stats(stats, space.calls);
            hnswlib::HNSWStats loaded_stats;
            require(same_results(baseline, loaded.searchKnnShortcuts(query, nullptr, &loaded_stats)), "SHG save/load changed results");
            require(stats.ndis == loaded_stats.ndis && stats.nhops == loaded_stats.nhops, "SHG save/load changed work");
            for (float value : shg.resultsProcessing) require(value == -1, "SHG query leaked scratch state");
            const auto vanilla = shg.searchKnnHNSW(query);
            hnswlib::HNSWStats vanilla_stats;
            space.calls = 0;
            require(same_results(vanilla, shg.searchKnnHNSW(query, nullptr, &vanilla_stats)), "HEDS vanilla counters changed results");
            check_stats(vanilla_stats, space.calls);
        }
        // Tiny SHG graphs can have no shortcut candidates.
        hnswlib::HEDS<float> tiny(&space, dimension, 1, 4, 32, 100);
        tiny.addDataPoint(vectors[0].data(), 0, 1, performance);
        count = 1;
        tiny.buildShortcuts(count);
        require(tiny.shortcutsSize == 0, "Unexpected shortcut in single-node graph");
        std::cout << "Statistics, filters, instrumentation parity, scratch reset and save/load checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
