#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <mma.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda/ptx>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

enum class Layout {
  RowMajor,
  ColMajor
};

template <Layout L>
__device__ __forceinline__
int getIdx(int r, int c, int ld) {
  if constexpr (L == Layout::RowMajor)
    return r * ld + c;
  else
    return c * ld + r;
}

inline int idx_row_major(int r, int c, int ld) {
    return r * ld + c;
}
inline int idx_col_major(int r, int c, int ld) {
    return c * ld + r;
}

#endif