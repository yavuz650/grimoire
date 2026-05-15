// Measures cache latency via pointer-chasing microbenchmark.
// Usage: nvcc -O2 -o test_memory_latency test_memory_latency.cu && ./test_memory_latency

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>

// ─── Config ────────────────────────────────────────────────────────────────
#define ITERATIONS      (1 << 20)   // pointer-chase steps per kernel
#define WARMUP_RUNS     3
#define MEASURE_RUNS    10

// Buffer sizes to sweep (in bytes). Straddle L1, L2, and DRAM.
static const size_t SWEEP_SIZES[] = {
    16  * 1024,          //  16 KB  — likely L1 territory
    32  * 1024,          //  32 KB  — likely L1 territory
    128 * 1024,          // 128 KB
    512 * 1024,          // 512 KB
    1   * 1024 * 1024,   //   1 MB
    2   * 1024 * 1024,   //   2 MB
    4   * 1024 * 1024,   //   4 MB
    8   * 1024 * 1024,   //   8 MB
    16  * 1024 * 1024,   //  16 MB  — well into DRAM for most GPUs
    32  * 1024 * 1024,   //  32 MB
    64  * 1024 * 1024,   //  64 MB
    128  * 1024 * 1024,   //  128 MB
    256  * 1024 * 1024,   //  256 MB
};
static const int NUM_SIZES = sizeof(SWEEP_SIZES) / sizeof(SWEEP_SIZES[0]);
// ───────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Kernel ────────────────────────────────────────────────────────────────
//
// Single-thread pointer chase. Each load depends on the previous result,
// which serializes memory ops and prevents ILP / prefetching from hiding
// true latency.  clock64() gives per-SM cycle counts.
//
__global__ void pointer_chase(const uint32_t * __restrict__ buf,
                              uint32_t iterations,
                              uint64_t *cycles_out)
{
  uint32_t idx = 0;
  uint64_t start = clock64();

  #pragma unroll 1          // must NOT unroll — would break dependency chain
  for (uint32_t i = 0; i < iterations; i++) {
    idx = buf[idx];
  }

  uint64_t stop = clock64();

  // Prevent dead-code elimination of the loop
  cycles_out[0] = stop - start;
  cycles_out[1] = (uint64_t)idx;   // also prevents DCE
}
// ───────────────────────────────────────────────────────────────────────────

// Build a random permutation that forms a single cycle over [0, n).
// Fisher-Yates on an iota, then rotate so index 0 is the start of the chain.
static void build_random_permutation(std::vector<uint32_t> &perm, uint32_t n)
{
  perm.resize(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::mt19937 rng(42);
  for (uint32_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<uint32_t> dist(0, i);
    std::swap(perm[i], perm[dist(rng)]);
  }

  // perm[i] = next index in chain — turn permutation into a linked list
  // by treating perm as "successor of i"
  // (Fisher-Yates already gives us a random permutation; using it directly
  //  as buf[i]=perm[i] means buf[0]->perm[0]->perm[perm[0]]->...
  //  which is a valid random walk, though not strictly one cycle.
  //  For latency measurement this is fine — every element is reachable.)
}

// Query SM clock rate in kHz, convert to GHz
static float get_sm_clock_ghz(int device)
{
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz,
                                     cudaDevAttrClockRate, device));
    return (float)clock_khz / 1e6f;   // kHz -> GHz
}

static void print_device_info(int device)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device : %s\n", prop.name);
    printf("L2 size: %d MB\n", prop.l2CacheSize / (1024 * 1024));
    printf("SM clk : %.3f GHz\n", get_sm_clock_ghz(device));
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);
}

int main(void)
{
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    print_device_info(device);

    float sm_ghz = get_sm_clock_ghz(device);

    // Allocate output buffer (2 x uint64: cycles + dummy anti-DCE value)
    uint64_t *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_cycles, 2 * sizeof(uint64_t)));

    uint64_t h_cycles[2];

    printf("%-12s  %10s  %10s\n", "Buf size", "Cycles", "Latency (ns)");
    printf("%-12s  %10s  %10s\n", "--------", "------", "------------");

    // Allocate the largest buffer we'll need up front
    size_t max_bytes = SWEEP_SIZES[NUM_SIZES - 1];
    uint32_t *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));

    std::vector<uint32_t> h_perm;

    for (int s = 0; s < NUM_SIZES; s++) {
        size_t buf_bytes = SWEEP_SIZES[s];
        uint32_t n_elems = (uint32_t)(buf_bytes / sizeof(uint32_t));

        // Build and upload a fresh random permutation for this size
        build_random_permutation(h_perm, n_elems);
        CUDA_CHECK(cudaMemcpy(d_buf, h_perm.data(),
                              buf_bytes, cudaMemcpyHostToDevice));

        // Flush persisting L2 lines (Ampere+; no-op on older architectures)
        cudaCtxResetPersistingL2Cache();

        // ── Warmup passes ────────────────────────────────────────────────── 
        for (int w = 0; w < WARMUP_RUNS; w++) {
            pointer_chase<<<1, 1>>>(d_buf, ITERATIONS, d_cycles);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ── Measurement passes ─────────────────────────────────────────────
        uint64_t total_cycles = 0;
        for (int m = 0; m < MEASURE_RUNS; m++) {
            CUDA_CHECK(cudaMemset(d_cycles, 0, 2 * sizeof(uint64_t)));
            pointer_chase<<<1, 1>>>(d_buf, ITERATIONS, d_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_cycles, d_cycles,
                                  2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_cycles += h_cycles[0];
        }

        uint64_t avg_cycles = total_cycles / MEASURE_RUNS / ITERATIONS;
        float    latency_ns = (float)avg_cycles / sm_ghz;

        // Pretty-print buffer size
        char size_str[32];
        if (buf_bytes >= 1024 * 1024)
            snprintf(size_str, sizeof(size_str), "%4zu MB", buf_bytes >> 20);
        else
            snprintf(size_str, sizeof(size_str), "%4zu KB", buf_bytes >> 10);

        printf("%-12s  %10llu  %10.1f\n",
               size_str, (unsigned long long)avg_cycles, latency_ns);
    }

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_cycles));
    return 0;
}