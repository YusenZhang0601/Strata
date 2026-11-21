// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#else
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    uint64_t total = 0;
    for (const auto& g : regions) total += g.second;
    if (total == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // same growth as lock_resident, but for the sum of the regions and with ONE margin
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) total + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    for (const auto& g : regions) {
        uint8_t* base = (uint8_t*) g.first;
        for (uint64_t off = 0; off < g.second; off += chunk) {
            const uint64_t n = g.second - off < chunk ? g.second - off : chunk;
            if (!VirtualLock(base + off, (SIZE_T) n)) {
                r.note = "VirtualLock stopped after " + std::to_string((unsigned long long) (r.locked_bytes >> 20)) +
                         " of " + std::to_string((unsigned long long) (total >> 20)) + " MiB (error " +
                         std::to_string(GetLastError()) + ")";
                r.ok = r.locked_bytes > 0;
                return r;
            }
            r.locked_bytes += n;
        }
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (total >> 20)) + " MiB in " +
             std::to_string(regions.size()) + " regions via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    for (const auto& g : regions) {
        if (g.first == nullptr || g.second == 0) continue;
        if (mlock(g.first, g.second) != 0) { r.note = "mlock failed (raise ulimit -l)"; r.ok = r.locked_bytes > 0; return r; }
        r.locked_bytes += g.second;
    }
    r.ok = r.locked_bytes > 0;
    r.note = r.ok ? "mlock of " + std::to_string(regions.size()) + " regions" : "nothing to lock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
#if defined(__APPLE__)   // macOS: F_RDADVISE is the read-ahead hint
    {
        radvisory ra{(off_t) (offset + at), (int) std::min(kAdviseStep, bytes - at)};
        (void) fcntl(fd, F_RDADVISE, &ra);
    }
#else
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
#endif
}
#endif

ProcIo proc_io_sample() {
    ProcIo r;
#if defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        unsigned long long v;
        while (std::fscanf(f, "%63[^:]: %llu\n", key, &v) == 2)
            if (std::strcmp(key, "read_bytes") == 0) { r.read_bytes = v; r.valid = true; }
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        if (const char* p = std::strrchr(buf, ')')) {   // fields after the command: state ppid ... minflt cminflt majflt
            unsigned long long minflt = 0, cminflt = 0, majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %llu %llu %llu", &minflt, &cminflt, &majflt) == 3)
                r.major_faults = majflt;
        }
    }
#endif
    return r;
}

int conv_cache_numa_node() {
    static const int node = []() -> int {
        const char* v = std::getenv("STRATA_CONV_CACHE_NUMA_NODE");
        if (v == nullptr || v[0] == '\0') return -1;
        char* end = nullptr;
        const long parsed = std::strtol(v, &end, 10);
        if (end == v || *end != '\0' || parsed < 0) return -1;
#if defined(_WIN32)
        ULONG highest = 0;
        if (!GetNumaHighestNodeNumber(&highest) || (unsigned long) parsed > highest) {
            std::fprintf(stderr, "strata memory: STRATA_CONV_CACHE_NUMA_NODE=%ld out of range (highest=%lu); disabled\n",
                         parsed, (unsigned long) highest);
            return -1;
        }
        std::fprintf(stderr, "strata memory: STRATA_CONV_CACHE_NUMA_NODE=%ld enabled (VirtualAllocExNuma)\n", parsed);
        return (int) parsed;
#elif defined(__linux__)
        if (parsed >= 256) return -1;
        std::fprintf(stderr, "strata memory: STRATA_CONV_CACHE_NUMA_NODE=%ld enabled (mmap+mbind)\n", parsed);
        return (int) parsed;
#else
        (void) parsed;
        return -1;
#endif
    }();
    return node;
}

uint64_t numa_node_available_bytes(int node) {
#if defined(_WIN32)
    if (node < 0) return UINT64_MAX;
    using Fn = BOOL(WINAPI*)(USHORT, PULONGLONG);
    static const Fn fn = []() -> Fn {
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        return k32 ? (Fn) (void*) GetProcAddress(k32, "GetNumaAvailableMemoryNodeEx") : nullptr;
    }();
    ULONGLONG avail = 0;
    if (fn != nullptr && fn((USHORT) node, &avail)) return (uint64_t) avail;
    return UINT64_MAX;
#else
    (void) node;
    return UINT64_MAX;
#endif
}

void* numa_alloc_pages(size_t bytes, int node, bool& fell_back) {
    fell_back = false;
    if (bytes == 0) return nullptr;
#if defined(_WIN32)
    static const uint64_t min_free_bytes = []() -> uint64_t {
        const char* v = std::getenv("STRATA_PARK_NUMA0_MIN_FREE_MIB");
        const long long mib = v != nullptr ? std::atoll(v) : 0;
        return mib > 0 ? (uint64_t) mib << 20 : 0ull;
    }();
    if (node >= 0 && min_free_bytes > 0) {
        const uint64_t avail = numa_node_available_bytes(node);
        if (avail != UINT64_MAX && (avail <= min_free_bytes || avail - min_free_bytes < (uint64_t) bytes)) {
            static std::atomic<bool> warned_floor{false};
            if (!warned_floor.exchange(true))
                std::fprintf(stderr, "strata memory: NUMA node %d free %llu MiB below STRATA_PARK_NUMA0_MIN_FREE_MIB=%llu; "
                                     "falling back to VirtualAlloc\n",
                             node, (unsigned long long) (avail >> 20), (unsigned long long) (min_free_bytes >> 20));
            fell_back = true;
            return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        }
    }
    if (node >= 0) {
        if (void* p = VirtualAllocExNuma(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                                         PAGE_READWRITE, (DWORD) node))
            return p;
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "strata memory: VirtualAllocExNuma(node=%d, %zu bytes) failed (error %lu); "
                                 "falling back to VirtualAlloc\n",
                         node, bytes, (unsigned long) GetLastError());
        fell_back = true;
    }
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#elif defined(__linux__)
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
#if defined(SYS_mbind)
    if (node >= 0 && node < 256) {
        unsigned long mask[4] = {0, 0, 0, 0};
        mask[node / 64] = 1ul << (node % 64);
        if (syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, mask, 256, 0) != 0) fell_back = true;
    }
#else
    (void) node;
    fell_back = true;
#endif
    return p;
#else
    (void) node;
    fell_back = true;
    return std::malloc(bytes);
#endif
}

void numa_free_pages(void* ptr, size_t bytes) noexcept {
    if (ptr == nullptr) return;
#if defined(_WIN32)
    (void) bytes;
    (void) VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(__linux__)
    if (bytes > 0) (void) ::munmap(ptr, bytes);
#else
    (void) bytes;
    std::free(ptr);
#endif
}

bool pin_thread_to_numa_node(int node) {
#if defined(_WIN32)
    if (node < 0) return false;
    GROUP_AFFINITY ga{};
    if (!GetNumaNodeProcessorMaskEx((USHORT) node, &ga) || ga.Mask == 0) return false;
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
#else
    (void) node;
    return false;
#endif
}

bool verify_pages_on_numa_node(const void* ptr, size_t bytes, int expected_node,
                               size_t& on_node, size_t& off_node, size_t max_samples) {
    on_node = 0;
    off_node = 0;
#if defined(_WIN32)
    if (ptr == nullptr || bytes == 0 || expected_node < 0 || max_samples == 0) return false;
    constexpr uintptr_t pg = 4096;
    const uintptr_t base = (uintptr_t) ptr;
    const uintptr_t span = bytes - 1;
    const size_t n_pages = (bytes + pg - 1) / pg;
    const size_t samples = std::min<size_t>(256, std::min<size_t>(max_samples, std::max<size_t>(1, n_pages)));
    PSAPI_WORKING_SET_EX_INFORMATION w[256] = {};
    for (size_t i = 0; i < samples; ++i) {
        const uintptr_t off = samples > 1 ? (span / (samples - 1)) * i : 0;
        w[i].VirtualAddress = (PVOID) ((base + off) & ~(pg - 1));
    }
    if (!K32QueryWorkingSetEx(GetCurrentProcess(), w, (DWORD) (samples * sizeof(w[0])))) return false;
    for (size_t i = 0; i < samples; ++i) {
        if (!w[i].VirtualAttributes.Valid) continue;
        if ((int) w[i].VirtualAttributes.Node == expected_node) ++on_node;
        else ++off_node;
    }
    return true;
#else
    (void) ptr; (void) bytes; (void) expected_node; (void) max_samples;
    return false;
#endif
}

}  // namespace strata::platform
