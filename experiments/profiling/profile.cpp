#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset = "sift", algorithm = "hnsw";
    fs::path data_root;
    fs::path data_file, query_file, groundtruth_file, output_file, performance_file,
            layer_averages_file, index_file;
    int M = 48, ef_construction = 80, ef_search = 64, k = 10, warmup = 500, seed = 100;
    bool M_set = false, ef_construction_set = false;
    bool check_data = false, build_only = false, search_only = false, force = false;
};

static uint32_t read_u32(std::istream& in, const std::string& context) {
    uint32_t value = 0;
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("Truncated " + context);
    return value;
}

struct VectorFile {
    fs::path path;
    bool fvecs;
    uint32_t count, dimension;
    std::ifstream input;

    VectorFile(const fs::path& filename, bool is_fvecs) : path(filename), fvecs(is_fvecs), input(filename, std::ios::binary) {
        if (!input) throw std::runtime_error("Cannot open " + path.string());
        const uintmax_t bytes = fs::file_size(path);
        if (fvecs) {
            dimension = read_u32(input, path.string());
            if (!dimension || bytes % (4ULL * (dimension + 1)))
                throw std::runtime_error("Invalid fvecs size: " + path.string());
            count = bytes / (4ULL * (dimension + 1));
            if (!count) throw std::runtime_error("Empty vector file: " + path.string());
            input.seekg(0);
        } else {
            const uint32_t element_bytes = read_u32(input, path.string());
            count = read_u32(input, path.string());
            dimension = read_u32(input, path.string());
            if (element_bytes != sizeof(float) || !count || !dimension ||
                bytes != 12ULL + 4ULL * count * dimension)
                throw std::runtime_error("Invalid data_new header or size: " + path.string());
        }
    }

    void next(std::vector<float>& vector) {
        if (fvecs && read_u32(input, path.string()) != dimension)
            throw std::runtime_error("Inconsistent fvecs dimension: " + path.string());
        vector.resize(dimension);
        if (!input.read(reinterpret_cast<char*>(vector.data()), sizeof(float) * dimension))
            throw std::runtime_error("Truncated vectors: " + path.string());
    }
};

struct TruthFile {
    fs::path path;
    uint32_t count, width;
    std::ifstream input;
    explicit TruthFile(const fs::path& filename) : path(filename), input(filename, std::ios::binary) {
        if (!input) throw std::runtime_error("Cannot open " + path.string());
        const auto bytes = fs::file_size(path);
        width = read_u32(input, path.string());
        if (!width || bytes % (4ULL * (width + 1)))
            throw std::runtime_error("Invalid ivecs size: " + path.string());
        count = bytes / (4ULL * (width + 1));
        input.seekg(0);
    }
    std::vector<uint32_t> next() {
        if (read_u32(input, path.string()) != width)
            throw std::runtime_error("Inconsistent ivecs width: " + path.string());
        std::vector<uint32_t> row(width);
        if (!input.read(reinterpret_cast<char*>(row.data()), 4ULL * width))
            throw std::runtime_error("Truncated ground truth: " + path.string());
        return row;
    }
};

static int integer_arg(const char* value, const char* name, int minimum) {
    size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size() || parsed < minimum)
        throw std::runtime_error(std::string("Invalid ") + name + ": " + value);
    return parsed;
}

static Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-data") { opts.check_data = true; continue; }
        if (key == "--build-only") { opts.build_only = true; continue; }
        if (key == "--search-only") { opts.search_only = true; continue; }
        if (key == "--force") { opts.force = true; continue; }
        if (key == "--help") {
            std::cout << "Usage: hnsw_profile --algorithm hnsw|shg --dataset sift|enron [--M N] [--ef-construction N] "
                         "[--ef-search 64] [--k 10] [--warmup 500] [--seed 100] [--threads 1] [--data-root PATH] [--data-file PATH] "
                         "[--query-file PATH] [--groundtruth-file PATH] [--output PATH] "
                         "[--performance-output PATH] [--layer-averages-output PATH] [--index-file PATH] "
                         "[--build-only|--search-only] [--force] [--check-data]\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--dataset") opts.dataset = value;
        else if (key == "--algorithm") opts.algorithm = value;
        else if (key == "--data-root") opts.data_root = value;
        else if (key == "--seed") opts.seed = integer_arg(argv[i], "seed", 0);
        else if (key == "--threads") {
            if (integer_arg(argv[i], "threads", 1) != 1)
                throw std::runtime_error("Both profiling runners currently support --threads 1 only");
        }
        else if (key == "--M") { opts.M = integer_arg(argv[i], "M", 2); opts.M_set = true; }
        else if (key == "--ef-construction") {
            opts.ef_construction = integer_arg(argv[i], "ef-construction", 1);
            opts.ef_construction_set = true;
        }
        else if (key == "--ef-search") opts.ef_search = integer_arg(argv[i], "ef-search", 1);
        else if (key == "--k") opts.k = integer_arg(argv[i], "k", 1);
        else if (key == "--warmup") opts.warmup = integer_arg(argv[i], "warmup", 0);
        else if (key == "--data-file") opts.data_file = value;
        else if (key == "--query-file") opts.query_file = value;
        else if (key == "--groundtruth-file") opts.groundtruth_file = value;
        else if (key == "--output") opts.output_file = value;
        else if (key == "--performance-output") opts.performance_file = value;
        else if (key == "--layer-averages-output") opts.layer_averages_file = value;
        else if (key == "--index-file") opts.index_file = value;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (opts.dataset != "sift" && opts.dataset != "enron")
        throw std::runtime_error("--dataset must be sift or enron");
    if (opts.algorithm == "hnsw") {
        if (!opts.M_set) opts.M = 16;
        if (!opts.ef_construction_set) opts.ef_construction = 200;
    }
    if (opts.M > 10000) throw std::runtime_error("M must be at most 10000");
    if (opts.algorithm != "hnsw" && opts.algorithm != "shg")
        throw std::runtime_error("--algorithm must be hnsw or shg");
    if (opts.check_data && opts.build_only)
        throw std::runtime_error("--check-data requires query and ground truth files; omit --build-only");
    if (opts.build_only && opts.search_only)
        throw std::runtime_error("Choose either --build-only or --search-only");
    if (opts.ef_search < opts.k)
        throw std::runtime_error("ef-search must be at least k");
    const fs::path root = "experiments/profiling";
    const fs::path data_dir = (opts.data_root.empty() ? root / "data" : opts.data_root) / opts.dataset;
    if (opts.data_file.empty()) opts.data_file = data_dir / (opts.dataset == "sift" ? "sift_base.fvecs" : "enron.data_new");
    if (opts.query_file.empty()) opts.query_file = data_dir / (opts.dataset + "_query.fvecs");
    if (opts.groundtruth_file.empty()) opts.groundtruth_file = data_dir / (opts.dataset + "_groundtruth.ivecs");
    if (opts.output_file.empty()) opts.output_file = root / "results" / opts.dataset / opts.algorithm /
        ("M" + std::to_string(opts.M) + "_efc" + std::to_string(opts.ef_construction) +
         "_ef" + std::to_string(opts.ef_search)) / "layer_stats.csv";
    if (opts.performance_file.empty()) opts.performance_file = root / "observations" /
        (opts.dataset + "_" + opts.algorithm + "_performance.csv");
    if (opts.layer_averages_file.empty()) opts.layer_averages_file = root / "observations" /
        (opts.dataset + "_" + opts.algorithm + "_layer_averages.csv");
    if (opts.index_file.empty()) opts.index_file = root / "indexes" / opts.dataset /
        (opts.dataset + "_" + opts.algorithm + "_M" + std::to_string(opts.M) + ".index");
    return opts;
}

static std::ofstream csv_file(const fs::path& path, const std::string& header, bool append) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    const bool write_header = !append || !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream out(path, append ? std::ios::app : std::ios::out);
    if (!out) throw std::runtime_error("Cannot write " + path.string());
    // Catch disk-full and other write failures instead of silently losing a run.
    out.exceptions(std::ios::badbit | std::ios::failbit);
    if (write_header) out << header << '\n';
    out << std::fixed << std::setprecision(8);
    return out;
}

static const std::string performance_header =
    "dataset,algorithm,phase,count,dimension,M,ef_construction,ef_search,k,threads,"
    "seconds,average_latency_us,queries_per_second,recall_at_k,shortcut_build_seconds,"
    "total_build_seconds,seed,warmup";

static double recall_at_k(std::priority_queue<std::pair<float, hnswlib::labeltype>> labels,
                          const std::vector<uint32_t>& truth, int k) {
    std::unordered_set<uint32_t> expected(truth.begin(), truth.begin() + k);
    int hits = 0;
    while (!labels.empty()) {
        hits += expected.count(static_cast<uint32_t>(labels.top().second));
        labels.pop();
    }
    return static_cast<double>(hits) / k;
}

// Both algorithms use one C++ driver, input reader, wall clock and CSV schema.
// Their native construction/search routines remain separate.
template<bool SHG>
static int run(const Options& opts) {
    using Index = std::conditional_t<SHG, hnswlib::HEDS<float>, hnswlib::HierarchicalNSW<float>>;
    std::unique_ptr<VectorFile> base, query_file;
    std::unique_ptr<TruthFile> truth_file;
    if (!opts.search_only || opts.check_data)
        base = std::make_unique<VectorFile>(opts.data_file, opts.data_file.extension() == ".fvecs");
    if (!opts.build_only) {
        query_file = std::make_unique<VectorFile>(opts.query_file, true);
        truth_file = std::make_unique<TruthFile>(opts.groundtruth_file);
        if ((base && base->dimension != query_file->dimension) ||
            query_file->count != truth_file->count || opts.k > static_cast<int>(truth_file->width))
            throw std::runtime_error("Base/query dimensions, query/truth counts, or k/truth width disagree");
    }
    if (opts.check_data) {
        std::cout << "Headers: " << opts.dataset << ", base=" << base->count
                  << ", queries=" << query_file->count << ", dimension=" << base->dimension
                  << ", ground truth width=" << truth_file->width << '\n';
        return 0;
    }
    const uint32_t dimension = base ? base->dimension : query_file->dimension;
    if (SHG && dimension < 64)
        throw std::runtime_error("SHG's level-2 compressed pruning requires dimension >= 64");
    if (dimension > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Dimension exceeds supported int range");
    const uint64_t requested_capacity = (base ? uint64_t(base->count) : 0) +
                                       (SHG && query_file ? uint64_t(query_file->count) : 0);
    if (requested_capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Too many elements for this runner's internal IDs");
    if (!opts.search_only && !opts.force &&
        (fs::exists(opts.index_file) || (SHG && fs::exists(opts.index_file.string() + ".shg"))))
        throw std::runtime_error("Index already exists; use --force to rebuild: " + opts.index_file.string());

    hnswlib::L2Space space(dimension);
    std::unique_ptr<Index> holder;
    Performance preprocessing;
    std::vector<float> vector;
    const auto build_start = Clock::now();
    if (opts.search_only) {
        holder = std::make_unique<Index>(&space, opts.index_file.string(), false, 0);
        if (holder->label_offset_ - holder->offsetData_ != sizeof(float) * uint64_t(dimension))
            throw std::runtime_error("Saved index dimension does not match queries");
        if (holder->M_ != static_cast<size_t>(opts.M))
            throw std::runtime_error("Saved index M does not match --M");
        if (!holder->cur_element_count) throw std::runtime_error("Saved index is empty");
        if constexpr (SHG) {
            const uint64_t capacity = uint64_t(holder->cur_element_count) + query_file->count;
            if (capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Too many stored queries for SHG internal IDs");
            if (capacity > holder->max_elements_) holder->resizeIndex(capacity);
            holder->loadSearchState(opts.index_file.string() + ".shg", dimension);
        }
    } else {
        if constexpr (SHG)
            holder = std::make_unique<Index>(&space, dimension, requested_capacity,
                                            opts.M, opts.ef_construction, opts.seed);
        else
            holder = std::make_unique<Index>(&space, requested_capacity,
                                            opts.M, opts.ef_construction, opts.seed);
        for (uint32_t id = 0; id < base->count; ++id) {
            base->next(vector);
            if constexpr (SHG) holder->addDataPoint(vector.data(), id, 1, preprocessing);
            else holder->addPoint(vector.data(), id);
        }
    }
    const double build_seconds = std::chrono::duration<double>(Clock::now() - build_start).count();
    auto& index = *holder;
    const uint32_t base_count = index.cur_element_count;
    double shortcut_seconds = 0;
    if (!opts.search_only) {
        if constexpr (SHG) {
            const auto shortcut_start = Clock::now();
            int count = base_count;
            index.buildShortcuts(count);
            shortcut_seconds = std::chrono::duration<double>(Clock::now() - shortcut_start).count();
        }
        if (!opts.index_file.parent_path().empty()) fs::create_directories(opts.index_file.parent_path());
        index.saveIndex(opts.index_file.string());
        if constexpr (SHG) index.saveSearchState(opts.index_file.string() + ".shg");
        auto out = csv_file(opts.performance_file, performance_header, true);
        out << opts.dataset << ',' << opts.algorithm << ",build," << base_count << ','
            << dimension << ',' << index.M_ << ',' << index.ef_construction_ << ",,,1,"
            << build_seconds << ",,,," << shortcut_seconds << ',' << build_seconds + shortcut_seconds
            << ',' << opts.seed << ",\n";
        std::cout << "Graph build: " << build_seconds << " s; shortcuts: " << shortcut_seconds
                  << " s; total: " << build_seconds + shortcut_seconds << " s\n"
                  << "Saved index: " << opts.index_file << '\n';
        if (opts.build_only) return 0;
    }

    // Queries, ground truth, SHG query representations and scratch allocation
    // are all outside the timed search pass.
    std::vector<std::vector<float>> queries(query_file->count);
    std::vector<std::vector<uint32_t>> truth;
    truth.reserve(query_file->count);
    if (static_cast<uint32_t>(opts.k) > base_count)
        throw std::runtime_error("k exceeds the number of indexed vectors");
    for (uint32_t id = 0; id < query_file->count; ++id) {
        query_file->next(queries[id]);
        if constexpr (SHG) index.addDataPoint(queries[id].data(), base_count + id, -1, preprocessing);
        truth.push_back(truth_file->next());
        for (int j = 0; j < opts.k; ++j)
            if (truth.back()[j] >= base_count)
                throw std::runtime_error("Ground truth contains an out-of-range label (expected zero-based IDs)");
    }
    const int layers = index.maxlevel_ + 1;
    std::vector<uint32_t> nodes(layers, 0);
    for (uint32_t id = 0; id < base_count; ++id)
        for (int layer = 0; layer <= index.element_levels_[id]; ++layer) ++nodes[layer];
    if constexpr (SHG)
        index.resultsProcessing.assign(static_cast<size_t>(base_count) * layers, -1);
    index.setEf(opts.ef_search);
    const auto search = [&](uint32_t id, hnswlib::HNSWStats* stats) {
        if constexpr (SHG) {
            Query query(static_cast<int>(base_count + id), opts.k);
            return index.searchKnnShortcuts(query, nullptr, stats);
        } else {
            return index.searchKnn(queries[id].data(), id, opts.k, nullptr, stats);
        }
    };
    const uint32_t warmup = std::min<uint32_t>(opts.warmup, query_file->count);
    for (uint32_t id = 0; id < warmup; ++id) search(id, nullptr);
    uint64_t timed_results = 0;
    const auto search_start = Clock::now();
    for (uint32_t id = 0; id < query_file->count; ++id)
        timed_results += search(id, nullptr).size();
    const double search_seconds = std::chrono::duration<double>(Clock::now() - search_start).count();

    auto out = csv_file(opts.output_file,
        "dataset,algorithm,query_id,layer,nodes_on_layer,max_level,num_layers,M,ef_search,k,"
        "hops,distance_computations,full_distance_computations,compressed_distance_computations,"
        "seed_distance_computations,pruned_neighbors,levels_skipped,total_query_hops,"
        "total_query_distance_computations,query_latency_us,recall_at_k", false);
    hnswlib::HNSWStats totals(layers);
    double total_recall = 0;
    for (uint32_t id = 0; id < query_file->count; ++id) {
        hnswlib::HNSWStats stats(layers);
        const auto query_start = Clock::now();
        const auto result = search(id, &stats);
        const double query_latency_us =
            std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();
        const double recall = recall_at_k(result, truth[id], opts.k);
        totals.combine(stats);
        total_recall += recall;
        for (int layer = 0; layer < layers; ++layer) {
            const auto& row = stats.layers[layer];
            out << opts.dataset << ',' << opts.algorithm << ',' << id << ',' << layer << ',' << nodes[layer] << ','
                << layers - 1 << ',' << layers << ',' << index.M_ << ',' << opts.ef_search << ',' << opts.k << ','
                << row.hops << ',' << row.distance_computations << ',' << row.full_distance_computations << ','
                << row.compressed_distance_computations << ',' << row.seed_distance_computations << ','
                << row.pruned_neighbors << ',' << row.levels_skipped << ',' << stats.nhops << ',' << stats.ndis
                << ',' << query_latency_us << ',' << recall << '\n';
        }
    }
    const double nq = query_file->count;
    const double average_recall = total_recall / nq;
    const double qps = nq / search_seconds;
    auto performance_out = csv_file(opts.performance_file, performance_header, true);
    performance_out << opts.dataset << ',' << opts.algorithm << ",search," << query_file->count << ','
        << dimension << ',' << index.M_ << ',' << index.ef_construction_ << ',' << opts.ef_search << ',' << opts.k
        << ",1," << search_seconds << ',' << search_seconds / nq * 1e6 << ',' << qps << ',' << average_recall
        << ",,,," << warmup << '\n'; // build seed is not recoverable from legacy graph files
    auto averages = csv_file(opts.layer_averages_file,
        "dataset,algorithm,M,ef_search,k,num_queries,layer,nodes_on_layer,average_hops,"
        "average_distance_computations,recall_at_k,queries_per_second,average_full_distance_computations,"
        "average_compressed_distance_computations,average_seed_distance_computations,"
        "average_pruned_neighbors,average_levels_skipped", true);
    for (int layer = 0; layer < layers; ++layer) {
        const auto& row = totals.layers[layer];
        averages << opts.dataset << ',' << opts.algorithm << ',' << index.M_ << ',' << opts.ef_search << ','
            << opts.k << ',' << query_file->count << ',' << layer << ',' << nodes[layer] << ','
            << row.hops / nq << ',' << row.distance_computations / nq << ',' << average_recall << ',' << qps << ','
            << row.full_distance_computations / nq << ',' << row.compressed_distance_computations / nq << ','
            << row.seed_distance_computations / nq << ',' << row.pruned_neighbors / nq << ',' << row.levels_skipped / nq << '\n';
    }
    std::cout << "Algorithm: " << opts.algorithm << "; queries: " << query_file->count
              << "; layers: " << layers << "; timed results: " << timed_results << '\n'
              << "Search time: " << search_seconds << " s; latency: " << search_seconds / nq * 1e6
              << " us/query; QPS: " << qps << "; recall@" << opts.k << ": " << average_recall << '\n'
              << "Average hops: " << totals.nhops / nq << "; distance evaluations: " << totals.ndis / nq << '\n'
              << "Per-query/per-layer CSV: " << opts.output_file << '\n'
              << "Per-layer averages CSV: " << opts.layer_averages_file << '\n'
              << "Performance CSV: " << opts.performance_file << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        return opts.algorithm == "shg" ? run<true>(opts) : run<false>(opts);
    } catch (const std::exception& error) {
        std::cerr << "hnsw_profile: " << error.what() << '\n';
        return 1;
    }
}
