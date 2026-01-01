#include <mma.h>

__device__ void ldmatrix_x2_m8n8_b16(uint32_t &dst0, uint32_t &dst1, uint32_t src) {
  asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
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

// Use inline PTX mma instructions to do matrix multiplication
// s8 * s8 -> s32
// All inputs and outputs are supposed to be in shared memory
__device__ void mma_m16n8k16_s8_s8_smem(int8_t *A, int8_t *B, int32_t *C) {
  // For A, we do x2 8x8 ldmatrix, so the first 15 threads provide row addresses
  int laneID = threadIdx.x & 31;
  int8_t *a = A + (laneID & 0x1f) * 16;
  uint32_t cvt_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
  uint32_t dstA[2];
  ldmatrix_x2_m8n8_b16(dstA[0], dstA[1], cvt_a);

  // For B, we do x1 8x8 ldmatrix
  int8_t *b = B + (laneID & 0x1f) * 16;
  uint32_t cvt_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  uint32_t dstB;
  ldmatrix_x1_m8n8_b16(dstB, cvt_b);

  int groupID = laneID >> 2;
  int threadID_in_group = laneID % 4;

  mma_m16n8k16_row_col_s32_s8_s8_s32(C[groupID*8+threadID_in_group*2], C[groupID*8+threadID_in_group*2 + 1],
                                     C[(groupID+8)*8+threadID_in_group*2], C[(groupID+8)*8+threadID_in_group*2 + 1], 
                                     dstA[0], dstA[1], dstB);

  // This requires sm_90 or above lmao
  // int32_t *c = smemC + (threadIdx.x & 0x1f) * 16;
  // uint32_t smem_c = static_cast<uint32_t>(__cvta_generic_to_shared(c));
  // asm volatile ("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
  //     :: "r"(smem_c),
  //        "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));

}