#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <queue>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <arm_neon.h>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// Flat 模式使用 flat_search_simd（本文件），不调用 flat_scan.h 中的 flat_search
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

struct SQIndex {
    std::vector<int8_t> codes;
    std::vector<float> scales;
};

struct PQIndex {
    size_t m = 8;
    size_t subdim = 0;
    size_t ks = 16;
    std::vector<float> codebooks;
    std::vector<uint8_t> codes;
};

static inline float hsum4(float32x4_t v) {
    float32x2_t p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    p = vpadd_f32(p, p);
    return vget_lane_f32(p, 0);
}

static inline float dot_simd(const float* a, const float* b, size_t dim) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    float sum = hsum4(acc);
    for(; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

static inline float l2_simd(const float* a, const float* b, size_t dim) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 4 <= dim; i += 4) {
        float32x4_t da = vld1q_f32(a + i);
        float32x4_t db = vld1q_f32(b + i);
        float32x4_t d = vsubq_f32(da, db);
        acc = vmlaq_f32(acc, d, d);
    }
    float sum = hsum4(acc);
    for(; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline void keep_topk(std::priority_queue<std::pair<float, uint32_t>>& heap, size_t k, float dis, uint32_t id) {
    if(heap.size() < k) heap.emplace(dis, id);
    else if(dis < heap.top().first) {
        heap.pop();
        heap.emplace(dis, id);
    }
}

// DEEP100K 使用内积相似度，距离定义为 1 - <x,q>，与课程模板一致；内积由 NEON dot_simd 计算
static std::priority_queue<std::pair<float, uint32_t>> flat_search_simd(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for(size_t i = 0; i < base_number; ++i) {
        float ip = dot_simd(base + i * vecdim, query, vecdim);
        float dis = 1.0f - ip;
        keep_topk(heap, k, dis, static_cast<uint32_t>(i));
    }
    return heap;
}

static SQIndex build_sq_index(const float* base, size_t base_number, size_t vecdim) {
    SQIndex sq;
    sq.codes.resize(base_number * vecdim);
    sq.scales.resize(base_number);
    for(size_t i = 0; i < base_number; ++i) {
        const float* x = base + i * vecdim;
        float mx = 0.0f;
        for(size_t j = 0; j < vecdim; ++j) mx = std::max(mx, std::abs(x[j]));
        float scale = (mx > 1e-12f) ? (mx / 127.0f) : 1.0f;
        sq.scales[i] = scale;
        int8_t* dst = sq.codes.data() + i * vecdim;
        for(size_t j = 0; j < vecdim; ++j) {
            int v = static_cast<int>(std::lround(x[j] / scale));
            dst[j] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
        }
    }
    return sq;
}

static inline float sq_dot_simd(const float* q, const int8_t* code, float scale, size_t vecdim) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 8 <= vecdim; i += 8) {
        int8x8_t c8 = vld1_s8(code + i);
        int16x8_t c16 = vmovl_s8(c8);
        int32x4_t c32_lo = vmovl_s16(vget_low_s16(c16));
        int32x4_t c32_hi = vmovl_s16(vget_high_s16(c16));
        float32x4_t cf_lo = vcvtq_f32_s32(c32_lo);
        float32x4_t cf_hi = vcvtq_f32_s32(c32_hi);
        float32x4_t q_lo = vld1q_f32(q + i);
        float32x4_t q_hi = vld1q_f32(q + i + 4);
        acc = vmlaq_f32(acc, q_lo, cf_lo);
        acc = vmlaq_f32(acc, q_hi, cf_hi);
    }
    float sum = hsum4(acc);
    for(; i < vecdim; ++i) sum += q[i] * static_cast<float>(code[i]);
    return sum * scale;
}

static size_t nearest_centroid(const float* x, const float* centroids, size_t ks, size_t subdim) {
    size_t best = 0;
    float best_dis = 1e30f;
    for(size_t c = 0; c < ks; ++c) {
        float dis = l2_simd(x, centroids + c * subdim, subdim);
        if(dis < best_dis) {
            best_dis = dis;
            best = c;
        }
    }
    return best;
}

static PQIndex build_pq_index(const float* base, size_t base_number, size_t vecdim) {
    PQIndex pq;
    while(vecdim % pq.m != 0 && pq.m > 1) --pq.m;
    pq.subdim = vecdim / pq.m;
    pq.codebooks.resize(pq.m * pq.ks * pq.subdim);
    pq.codes.resize(base_number * pq.m);

    for(size_t part = 0; part < pq.m; ++part) {
        float* cb = pq.codebooks.data() + part * pq.ks * pq.subdim;
        for(size_t c = 0; c < pq.ks; ++c) {
            const float* sample = base + ((c * 9973) % base_number) * vecdim + part * pq.subdim;
            std::copy(sample, sample + pq.subdim, cb + c * pq.subdim);
        }
        for(size_t i = 0; i < base_number; ++i) {
            const float* x = base + i * vecdim + part * pq.subdim;
            pq.codes[i * pq.m + part] = static_cast<uint8_t>(nearest_centroid(x, cb, pq.ks, pq.subdim));
        }
    }
    return pq;
}

static std::priority_queue<std::pair<float, uint32_t>> ann_search_dispatch(
    const std::string& method,
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    const SQIndex& sq,
    const PQIndex& pq)
{
    if(method == "flat") {
        return flat_search_simd(base, query, base_number, vecdim, k);
    }

    top_p = std::max(k, std::min(top_p, base_number));
    std::priority_queue<std::pair<float, uint32_t>> candidates;

    if(method == "pq") {
        std::vector<float> lut(pq.m * pq.ks);
        for(size_t part = 0; part < pq.m; ++part) {
            const float* cb = pq.codebooks.data() + part * pq.ks * pq.subdim;
            for(size_t c = 0; c < pq.ks; ++c) {
                lut[part * pq.ks + c] = dot_simd(query + part * pq.subdim, cb + c * pq.subdim, pq.subdim);
            }
        }
        for(size_t i = 0; i < base_number; ++i) {
            float approx = 0.0f;
            const uint8_t* code = pq.codes.data() + i * pq.m;
            for(size_t part = 0; part < pq.m; ++part) approx += lut[part * pq.ks + code[part]];
            keep_topk(candidates, top_p, 1.0f - approx, static_cast<uint32_t>(i));
        }
    } else {
        for(size_t i = 0; i < base_number; ++i) {
            const int8_t* code = sq.codes.data() + i * vecdim;
            float approx = sq_dot_simd(query, code, sq.scales[i], vecdim);
            keep_topk(candidates, top_p, 1.0f - approx, static_cast<uint32_t>(i));
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> result;
    while(!candidates.empty()) {
        uint32_t id = candidates.top().second;
        candidates.pop();
        float exact = dot_simd(base + static_cast<size_t>(id) * vecdim, query, vecdim);
        keep_topk(result, k, 1.0f - exact, id);
    }
    return result;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    std::string method = "pq"; // 可改为 flat / sq / pq
    size_t top_p = 256;
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    SQIndex sq_index;
    PQIndex pq_index;
    if(method == "sq") sq_index = build_sq_index(base, base_number, vecdim);
    if(method == "pq") pq_index = build_pq_index(base, base_number, vecdim);

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);
        (void)ret;

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        auto res = ann_search_dispatch(method, base, test_query + i*vecdim, base_number, vecdim, k, top_p, sq_index, pq_index);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        (void)ret;
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
