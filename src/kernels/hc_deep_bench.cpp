// src/kernels/hc_deep_bench.cpp - personal fork (2026-11-13, roofline): the hyper-connection multi read as plain / staged /
// deep (STRATA_HC_DEEP shapes), bit for bit against the plain read on every output (lo, rs, inject, mixed, R), and each
// one's time with the weights cold (rotating over `sets` copies of the 13.2 MB of weights, larger than the 6 MB L2), the
// stage split (norm / down / up) from GPU stamps.  Model shapes: n_embd 2560, hc 4, hc_lr 320.
//
//     hc_deep_bench [iters=400] [sets=16] [T min=1] [T max=8]
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(_WIN32)
    _putenv_s("STRATA_HC_PERSIST", "1");   // the existing opt-in persistent read (another summation order): timed and diffed below
#else
    setenv("STRATA_HC_PERSIST", "1", 1);
#endif
    const int iters = argc > 1 ? std::atoi(argv[1]) : 400;
    const int sets = argc > 2 ? std::max(1, std::atoi(argv[2])) : 16;
    const int t_lo = argc > 3 ? std::atoi(argv[3]) : 1, t_hi = argc > 4 ? std::atoi(argv[4]) : 8;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    std::vector<uint16_t*> cwd(sets), cwu(sets), cwi(sets);
    std::vector<float*> cwn(sets);
    for (int c = 0; c < sets; ++c) {
        CK(cudaMalloc((void**) &cwd[c], wd.size() * 2)); CK(cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &cwu[c], wu.size() * 2)); CK(cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &cwi[c], wi.size() * 2)); CK(cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &cwn[c], wn.size() * 4)); CK(cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice));
    }
    float *dR0, *dR, *dbo, *dinj, *dxn, *dlo, *drs, *dio, *dmix;
    CK(cudaMalloc((void**) &dR0, R.size() * 4)); CK(cudaMemcpy(dR0, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dR, R.size() * 4));
    CK(cudaMalloc((void**) &dbo, bo.size() * 4)); CK(cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dinj, inj.size() * 4)); CK(cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dxn, (size_t) TM * D * 4));
    CK(cudaMalloc((void**) &dlo, (size_t) TM * LR * 4));
    CK(cudaMalloc((void**) &drs, (size_t) TM * HC * 4));
    CK(cudaMalloc((void**) &dio, (size_t) TM * HC * 4));
    CK(cudaMalloc((void**) &dmix, (size_t) TM * N * 4));
    unsigned long long* dst;
    CK(cudaMalloc((void**) &dst, (size_t) iters * 4 * 8));
    std::vector<unsigned long long> hst((size_t) iters * 4);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    if (std::getenv("HC_BENCH_NOCHECK") == nullptr) K::fused_gr_check();
    K::fused_gr_persist_init();   // prints whether it runs here and its check against the default read   // the card's default variant, as the engine does (HC_BENCH_NOCHECK=1: skip, so a profiler sees the arms first)
    struct Arm { const char* name; int variant, wpb, pd, rpw; };
    const Arm arms[] = {{"plain", 1, -1, -1, -1}, {"staged", 3, -1, -1, -1}, {"w8r1p2", 4, 8, 2, 1}, {"w8r2p1", 4, 8, 1, 2},
                        {"w8r2p2", 4, 8, 2, 2}, {"w4r2p1", 4, 4, 1, 2}, {"w4r2p2", 4, 4, 2, 2}};
    const int NA = (int) (sizeof arms / sizeof arms[0]);
    int failures = 0;
    std::vector<K::FusedGrArgs> as((size_t) sets * TM);
    auto build = [&](int T, int apply, bool inplace) {
        for (int c = 0; c < sets; ++c)
            for (int t = 0; t < T; ++t) {
                K::FusedGrArgs& a = as[(size_t) c * TM + t];
                a = K::FusedGrArgs();
                a.R = (inplace ? dR : dR0) + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = apply != 0;
                a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                a.w_norm = cwn[c]; a.w_down = cwd[c]; a.w_up = cwu[c]; a.w_inject = cwi[c]; a.eps = 1e-6f;
                a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
            }
    };
    auto snapshot = [&](int T, std::vector<float>& o) {
        o.resize((size_t) T * (D + LR + HC + HC + N));
        float* p = o.data();
        CK(cudaMemcpy(p, dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost)); p += (size_t) T * D;
        CK(cudaMemcpy(p, dlo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost)); p += (size_t) T * LR;
        CK(cudaMemcpy(p, drs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost)); p += (size_t) T * HC;
        CK(cudaMemcpy(p, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost)); p += (size_t) T * HC;
        CK(cudaMemcpy(p, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost));
    };
    // 1) bitwise: every arm against plain, T x apply x (R separate / R in place), weight set 0
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inplace = 0; inplace < 2; ++inplace) {
                build(T, apply, inplace != 0);
                std::vector<float> ref, out;
                for (int a = 0; a < NA; ++a) {
                    K::fused_gr_deep_shape(arms[a].wpb, arms[a].pd, arms[a].rpw);
                    CK(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemset(dlo, 0xff, (size_t) TM * LR * 4)); CK(cudaMemset(drs, 0xff, (size_t) TM * HC * 4));
                    CK(cudaMemset(dio, 0xff, (size_t) TM * HC * 4)); CK(cudaMemset(dmix, 0xff, (size_t) TM * N * 4));
                    K::fused_gr_read_multi_variant(as.data(), T, dxn, s, arms[a].variant);
                    CK(cudaStreamSynchronize(s));
                    CK(cudaGetLastError());
                    if (a == 0) { snapshot(T, ref); continue; }
                    snapshot(T, out);
                    if (std::memcmp(ref.data(), out.data(), ref.size() * 4) != 0) {
                        size_t i = 0;
                        while (i < ref.size() && std::memcmp(&ref[i], &out[i], 4) == 0) ++i;
                        std::printf("BITWISE FAIL %s T %d apply %d inplace %d at float %zu\n", arms[a].name, T, apply, inplace, i);
                        ++failures;
                    }
                }
            }
    std::printf("bitwise: %d failures over T %d..%d x apply x inplace x %d arms\n", failures, t_lo, t_hi, NA - 1);
    // the persistent read (STRATA_HC_PERSIST, not bitwise): max relative difference against plain, and its cold time
    for (int T = t_lo; T <= t_hi; ++T) {
        build(T, 1, true);
        std::vector<float> ref, out;
        CK(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
        K::fused_gr_read_multi_variant(as.data(), T, dxn, s, 1);
        CK(cudaStreamSynchronize(s)); snapshot(T, ref);
        CK(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
        const bool ran = K::fused_gr_read_persist(as.data(), T, s);
        CK(cudaStreamSynchronize(s));
        if (!ran) { std::printf("PERSIST T %d: not run\n", T); continue; }
        snapshot(T, out);
        double mref = 0, md = 0, sq = 0, sqr = 0;
        for (size_t i = 0; i < ref.size(); ++i) { mref = std::max(mref, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(out[i] - ref[i])); sq += (double) (out[i] - ref[i]) * (out[i] - ref[i]); sqr += (double) ref[i] * ref[i]; }
        for (int rep = 0; rep < 2; ++rep) { for (int i = 0; i < iters; ++i) K::fused_gr_read_persist(&as[(size_t) (i % sets) * TM], T, s); CK(cudaStreamSynchronize(s)); }
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < iters; ++i) K::fused_gr_read_persist(&as[(size_t) (i % sets) * TM], T, s);
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / iters;
        std::printf("PERSIST T %d read %6.2f us (%5.0f GB/s) | max rel diff %.3g, rel rms %.3g vs plain\n", T, us, 13.230080e6 / (us * 1e3), md / mref, std::sqrt(sq / sqr));
    }
    // 2) cold timing, apply on, R in place (the engine's case), weights rotating over `sets` copies
    for (int T = t_lo; T <= t_hi; ++T) {
        build(T, 1, true);
        for (int a = 0; a < NA; ++a) {
            K::fused_gr_deep_shape(arms[a].wpb, arms[a].pd, arms[a].rpw);
            for (int rep = 0; rep < 2; ++rep) {
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi_variant(&as[(size_t) (i % sets) * TM], T, dxn, s, arms[a].variant, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                CK(cudaStreamSynchronize(s));
            }
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i)
                K::fused_gr_read_multi_variant(&as[(size_t) (i % sets) * TM], T, dxn, s, arms[a].variant);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            CK(cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost));
            std::vector<double> n_, d_, u_;
            for (int i = 1; i < iters; ++i) {
                const unsigned long long* p = &hst[(size_t) i * 4];
                n_.push_back((double) (p[1] - p[0])); d_.push_back((double) (p[2] - p[1])); u_.push_back((double) (p[3] - p[2]));
            }
            auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2] / 1e3; };
            const double us = 1e3 * ms / iters, gbs = 13.230080e6 / (us * 1e3);
            std::printf("COLD T %d %-8s read %6.2f us (%5.0f GB/s of the 13.23 MB) | stamps: norm %5.2f down %5.2f up %5.2f us\n",
                        T, arms[a].name, us, gbs, med(n_), med(d_), med(u_));
        }
    }
    std::printf("hc_deep_bench: %d bitwise failures\n", failures);
    return failures ? 1 : 0;
}
