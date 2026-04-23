#ifndef __MMA_INTRINSICS_CUH__
#define __MMA_INTRINSICS_CUH__

#include <mma.h>

__device__ void ldmatrix_x4_m8n8_b16(uint32_t &dst0, uint32_t &dst1, uint32_t &dst2, uint32_t &dst3, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(src));
}

__device__ void ldmatrix_x4_trans_m8n8_b16(uint32_t &dst0, uint32_t &dst1, uint32_t &dst2, uint32_t &dst3, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(src));
}

__device__ void ldmatrix_x2_m8n8_b16(uint32_t &dst0, uint32_t &dst1, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(src));
}

__device__ void ldmatrix_x2_trans_m8n8_b16(uint32_t &dst0, uint32_t &dst1, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(src));
}

__device__ void ldmatrix_x1_m8n8_b16(uint32_t &dst0, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst0)
        :  "r"(src));
}

__device__ void ldmatrix_x1_trans_m8n8_b16(uint32_t &dst0, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst0)
        :  "r"(src));
}

__device__ void mma_m16n8k16_row_col_s32_s8_s8_s32(int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3,
                                                    uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
                "{%0, %1, %2, %3},"
                "{%4, %5},"
                "{%6},"
                "{%7, %8, %9, %10};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      :  "r"(a0), "r"(a1),
          "r"(b0),
          "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

__device__ void mma_m16n8k16_row_col_f32_f16_f16_f32(float &d0, float &d1, float &d2, float &d3,
                                                     uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%10, %11, %12, %13};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
         "r"(b0), "r"(b1),
         "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

// Use inline PTX mma instructions to do matrix multiplication
// s8 * s8 -> s32
// All inputs and outputs are supposed to be in shared memory
__device__ void mma_m16n8k16_s8_s8_smem_row_col(int8_t *A, int8_t *B, int32_t *C, int ldmA=16, int ldmB=16) {
  // For A, we do x2 8x8 ldmatrix, so the first 15 threads provide row addresses
  int laneID = threadIdx.x & 31;
  int8_t *a = A + (laneID & 0x1f) * ldmA;
  uint32_t cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
  uint32_t dstA[2];
  ldmatrix_x2_m8n8_b16(dstA[0], dstA[1], cvt_a);

  // For B, we do x1 8x8 ldmatrix
  int8_t *b = B + (laneID & 0x1f) * ldmB;
  uint32_t cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  uint32_t dstB;
  ldmatrix_x1_m8n8_b16(dstB, cvt_b);

  int groupID = laneID >> 2;
  int threadID_in_group = laneID % 4;

  mma_m16n8k16_row_col_s32_s8_s8_s32(C[groupID*8+threadID_in_group*2], C[groupID*8+threadID_in_group*2 + 1],
                                     C[(groupID+8)*8+threadID_in_group*2], C[(groupID+8)*8+threadID_in_group*2 + 1],
                                     dstA[0], dstA[1], dstB);
}

// Use inline PTX mma instructions to do matrix multiplication
// f16 * f16 -> f32
// All inputs and outputs are supposed to be in shared memory
// A is row major in shared memory, and B is column major in shared memory
__device__ void mma_m16n8k16_f16_f16_smem_row_col(__half *A, __half *B, float *C, int ldmA=16, int ldmB=16, int ldmC=16) {
  // For A, we do x4 8x8 ldmatrix, so the first 32 threads provide row addresses
  int laneID = threadIdx.x & 31;
  __half *a = laneID >= 16 ? A + (laneID%16)*ldmA + 8 : A + laneID * ldmA;
  uint32_t cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
  uint32_t dstA[4];
  ldmatrix_x4_m8n8_b16(dstA[0], dstA[1], dstA[2], dstA[3], cvt_a);

  // For B, we do x2 8x8 ldmatrix
  __half *b = laneID >= 8 ? B + (laneID%8)*ldmB + 8 : B + laneID * ldmB;
  uint32_t cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  uint32_t dstB[2];
  ldmatrix_x2_m8n8_b16(dstB[0], dstB[1], cvt_b);

  int groupID = laneID >> 2;
  int threadID_in_group = laneID % 4;
  /* Rows and column of C are calculated using this formula taken from the PTX documentation

    row =      groupID                               for ci where i <  2
         groupID + 8                             for ci where i >= 2
    col =  (threadID_in_group * 2) + (i & 0x1)        for ci where i = {0,..,3}
  
  */
  mma_m16n8k16_row_col_f32_f16_f16_f32(C[groupID*ldmC+threadID_in_group*2], C[groupID*ldmC+threadID_in_group*2 + 1],
                                       C[(groupID+8)*ldmC+threadID_in_group*2], C[(groupID+8)*ldmC+threadID_in_group*2 + 1],
                                       dstA[0], dstA[1], dstA[2], dstA[3], dstB[0], dstB[1]);
}

// A is col major in shared memory, and B is row major in shared memory
__device__ void mma_m16n8k16_f16_f16_smem_col_row(__half *A, __half *B, float *C, int ldmA=16, int ldmB=16, int ldmC=16) {
  // For A, we do x4 8x8 ldmatrix, so the first 32 threads provide row addresses
  int laneID = threadIdx.x & 31;
  __half *a = laneID >= 16 ? A + (laneID%16)*ldmA + 8 : A + laneID * ldmA;
  uint32_t cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
  uint32_t dstA[4];
  ldmatrix_x4_trans_m8n8_b16(dstA[0], dstA[2], dstA[1], dstA[3], cvt_a);

  // For B, we do x2 8x8 ldmatrix
  //__half *b = laneID >= 8 ? B + (laneID%8)*ldmB + 8 : B + laneID * ldmB;
  __half *b = B + laneID * ldmB;
  uint32_t cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  uint32_t dstB[2];
  ldmatrix_x2_trans_m8n8_b16(dstB[0], dstB[1], cvt_b);

  int groupID = laneID >> 2;
  int threadID_in_group = laneID % 4;

  mma_m16n8k16_row_col_f32_f16_f16_f32(C[groupID*ldmC+threadID_in_group*2], C[groupID*ldmC+threadID_in_group*2 + 1],
                                       C[(groupID+8)*ldmC+threadID_in_group*2], C[(groupID+8)*ldmC+threadID_in_group*2 + 1],
                                       dstA[0], dstA[1], dstA[2], dstA[3], dstB[0], dstB[1]);
}

// Use inline PTX mma instructions to do matrix multiplication
// f16 * f16 -> f32
// All inputs and outputs are supposed to be in shared memory
// A is 64x64 row major in shared memory, and B 64x64 is column major in shared memory
// Assumes 4 warps linearly laid out per CTA
__device__ void mma_m16n8k16_f16_f16_smem_row_col_64x64(__half *A, __half *B, float *C) {
  int warpID = threadIdx.x / 32;
  int laneID = threadIdx.x & 31;

  // Load 8 tiles of A
  uint32_t dstA[32];
  __half *a = laneID >= 16 ? A + (laneID%16)*64 + 8 : A + laneID * 64;
  // Warps 0 and 1 load rows 0 to 31
  // Warps 2 and 3 load rows 32 to 63
  a = warpID & 2 ? a + 32*64 : a;
  size_t offset = 0;
  uint32_t cvt_a;
  for (int i = 0; i < 32; i+=4) {
    offset = (i/16) * 16*64 + ((i/4)%4) * 16;
    cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(a+offset));
    // For A, we do x4 8x8 ldmatrix, so the 32 threads provide row addresses
    ldmatrix_x4_m8n8_b16(dstA[i], dstA[i+1], dstA[i+2], dstA[i+3], cvt_a);
  }

  // For B, we do x2 8x8 ldmatrix
  uint32_t dstB[32];
  __half *b = laneID >= 8 ? B + (laneID%8)*64 + 8 : B + laneID * 64;
  b = warpID & 1 ? b + 32*64 : b;
  uint32_t cvt_b;
  for (int i = 0; i < 32; i+=2) {
    offset = (i/8) * 8*64 + ((i/2)%4) * 16;
    cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(b+offset));
    ldmatrix_x2_m8n8_b16(dstB[i], dstB[i+1], cvt_b);
  }
  
  // Now do MMAs, 32 in total (4*8)
  for (int i = 0; i < 8; i++) {
    int subtileRow = (warpID/2)*2 + i/4;
    int subtileCol = (warpID%2)*4 + i%4;
    int row = subtileRow*16;
    int col = subtileCol*8;
    int groupID = laneID >> 2;
    int threadID_in_group = laneID % 4;
    /* Rows and column of C are calculated using this formula taken from the PTX documentation

      row =      groupID                               for ci where i <  2
          groupID + 8                             for ci where i >= 2
      col =  (threadID_in_group * 2) + (i & 0x1)        for ci where i = {0,..,3}
    
    */
    int offsetA = (i/4)*16;
    int offsetB = (i%4)*8;
    for (int j = 0, k = 0; j < 16; j+=4, k+=2) {
      mma_m16n8k16_row_col_f32_f16_f16_f32(C[row*64+col+groupID*64+threadID_in_group*2], C[row*64+col+groupID*64+threadID_in_group*2 + 1],
                                           C[row*64+col+(groupID+8)*64+threadID_in_group*2], C[row*64+col+(groupID+8)*64+threadID_in_group*2 + 1],
                                           dstA[offsetA+j], dstA[offsetA+j+1], dstA[offsetA+j+2], dstA[offsetA+j+3], dstB[offsetB+k], dstB[offsetB+k+1]);
    }
  }
}

// Use inline PTX mma instructions to do matrix multiplication
// f16 * f16 -> f32
// All inputs and outputs are supposed to be in shared memory
// A is 64x64 row major in shared memory, and B 64x64 is column major in shared memory
// Assumes 4 warps linearly laid out per CTA
// Split C into two halves for TMA copy. Necessary when using swizzling.
template <bool splitC=true>
__device__ void mma_m16n8k16_f16_f16_smem_row_col_64x64_swizzle(__half *A, __half *B, float *C) {
  int warpID = threadIdx.x / 32;
  int laneID = threadIdx.x & 31;

  // Load 8 tiles of A
  uint32_t dstA[32];
  // One chunk is a 16B(8*fp16) contiguous piece in memory
  // One row has 8 chunks, i.e. 128B
  int chunkRowStart = laneID%16;
  // Warps 0 and 1 load rows 0 to 31
  // Warps 2 and 3 load rows 32 to 63
  chunkRowStart = warpID >= 2 ? chunkRowStart + 32 : chunkRowStart;
  int chunkColumnStart = laneID / 16;

  size_t offset = 0;
  uint32_t cvt_a;
  for (int i = 0; i < 8; i++) {
    int chunkRow = chunkRowStart + i/4 * 16;
    int chunkCol = chunkColumnStart + (i%4)*2;
    int swizzledRow = chunkRow;
    int swizzledCol = (chunkRow%8) ^ chunkCol; 
    cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(A+swizzledRow*64+swizzledCol*8));
    ldmatrix_x4_m8n8_b16(dstA[i*4], dstA[i*4+1], dstA[i*4+2], dstA[i*4+3], cvt_a);
  }

  // For B, we do x2 8x8 ldmatrix
  uint32_t dstB[32];
  chunkRowStart = laneID/8;
  chunkColumnStart = laneID%8;
  // Warps 0 and 2 load cols 0 to 31
  // Warps 1 and 3 load cols 32 to 63
  chunkColumnStart = warpID & 1 ? chunkColumnStart+32 : chunkColumnStart;
  uint32_t cvt_b;
  for (int i = 0; i < 16; i++) {
    int chunkRow = chunkRowStart + (i%4)*2;
    int chunkCol = chunkColumnStart + (i/4)*8;
    int swizzledRow = (chunkCol%8) ^ chunkRow;
    int swizzledCol = chunkCol;
    cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(B+swizzledRow*8+swizzledCol*64));
    ldmatrix_x2_m8n8_b16(dstB[2*i], dstB[2*i+1], cvt_b);
  }
  
  float dstC[4];
  // Now do MMAs, 32 in total (4*8)
  for (int i = 0; i < 8; i++) {
    int subtileRow = (warpID/2)*2 + i/4;
    int subtileCol = (warpID%2)*4 + i%4;
    int row = subtileRow*16;
    int col = subtileCol*8;
    int groupID = laneID >> 2;
    int threadID_in_group = laneID % 4;
    row += groupID;
    col += threadID_in_group*2;
    int splitCOffset = warpID % 2 == 0 ? 0 : 32*64;
    /* Rows and column of C are calculated using this formula taken from the PTX documentation

      row =      groupID                               for ci where i <  2
          groupID + 8                             for ci where i >= 2
      col =  (threadID_in_group * 2) + (i & 0x1)        for ci where i = {0,..,3}
    
    */
    int offsetA = (i/4)*16;
    int offsetB = (i%4)*8;
    dstC[0] = 0;
    dstC[1] = 0;
    dstC[2] = 0;
    dstC[3] = 0;
    for (int j = 0, k = 0; j < 16; j+=4, k+=2) {
      // Accumulate to dstC registers
      mma_m16n8k16_row_col_f32_f16_f16_f32(dstC[0], dstC[1], dstC[2], dstC[3],
                                           dstA[offsetA+j], dstA[offsetA+j+1], dstA[offsetA+j+2], dstA[offsetA+j+3], 
                                           dstB[offsetB+k], dstB[offsetB+k+1]);
    }
    // Store dstC registers to shared memory
    auto swizzle = [](int r, int c) {
      int chunkCol = c / 4;
      int swizzledChunk = chunkCol ^ (r % 8);
      return swizzledChunk * 4 + (c % 4);
    };
    if constexpr(splitC) {
      int swizzledCol = swizzle(row, col)%32;
      C[row*32     + swizzledCol + splitCOffset]    += dstC[0];
      C[row*32     + swizzledCol+1 + splitCOffset]  += dstC[1];
      C[(row+8)*32 + swizzledCol + splitCOffset]    += dstC[2];
      C[(row+8)*32 + swizzledCol+1 + splitCOffset] += dstC[3];
    }
    else {
      C[row*64     + col]    += dstC[0];
      C[row*64     + col+1]  += dstC[1];
      C[(row+8)*64 + col]    += dstC[2];
      C[(row+8)*64 + col+1] += dstC[3];
    }
  }
}

#endif /* __MMA_INTRINSICS_CUH__ */

