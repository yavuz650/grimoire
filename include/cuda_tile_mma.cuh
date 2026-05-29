#include "cuda_tile.h"
#include <cuda_fp16.h>
 
/* this kernel multiplies MxK and KxN matrices, where M=8 and N=16.  K is variable but must be divisible by 8.*/
__tile_global__ void cuda_tile_mma(__half* __restrict__ a, __half* __restrict__ b, float* __restrict__ c, int32_t M, int32_t N, int32_t K) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
 
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    c = ct::assume_aligned(c, 16_ic);
 
    auto aShape = ct::extents{M, K};
    auto bShape = ct::extents{K, N};
    auto cShape = ct::extents{M, N};

    auto aSpan = ct::tensor_span{a, aShape};
    auto bSpan = ct::tensor_span{b, bShape};
    auto cSpan = ct::tensor_span{c, cShape};
 
    auto aView = ct::partition_view{aSpan, ct::shape{4_ic, 8_ic}};
    auto bView = ct::partition_view{bSpan, ct::shape{8_ic, 4_ic}};
    auto cView = ct::partition_view{cSpan, ct::shape{4_ic, 4_ic}};
     
    using f32x4x4 = ct::tile<float, ct::shape<4, 4>>;
    auto accTile = ct::full<f32x4x4>(0);
 
    auto [xBlock, yBlock, dummy] = ct::bid();
    for (auto idx : ct::irange(0, K/8)) {
        auto aTile = aView.load_masked(yBlock, idx);
        auto bTile = bView.load_masked(idx, xBlock);
        accTile = ct::mma(aTile, bTile, accTile);
    }
 
    cView.store_masked(accTile, yBlock, xBlock);
}

