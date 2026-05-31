#include "cuda_tile.h"
#include <cuda_fp16.h>

#include "common.cuh"
 
template <Layout LayoutA, Layout LayoutB, Layout LayoutC>
__tile_global__ void cuda_tile_mma(__half* __restrict__ a, __half* __restrict__ b, float* __restrict__ c, int32_t M, int32_t N, int32_t K) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;
 
  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);
  c = ct::assume_aligned(c, 16_ic);
 
  auto aShape = ct::extents{M, K};
  auto bShape = ct::extents{K, N};
  auto cShape = ct::extents{M, N};

  // Construct spans immediately using lambdas
  auto aSpan = [&] {
    if constexpr (LayoutA == Layout::RowMajor) 
      return ct::tensor_span{a, aShape, ct::layout_right{}};
    else                                       
      return ct::tensor_span{a, aShape, ct::layout_left{}};
  }();

  auto bSpan = [&] {
    if constexpr (LayoutB == Layout::RowMajor) 
      return ct::tensor_span{b, bShape, ct::layout_right{}};
    else                                       
      return ct::tensor_span{b, bShape, ct::layout_left{}};
  }();

  auto cSpan = [&] {
    if constexpr (LayoutC == Layout::RowMajor) 
      return ct::tensor_span{c, cShape, ct::layout_right{}};
    else                                       
      return ct::tensor_span{c, cShape, ct::layout_left{}};
  }();
 
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

