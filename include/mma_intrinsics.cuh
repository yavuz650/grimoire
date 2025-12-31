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

__device__ void mma_m16n8k16_row_col_s32_s8_s8_s32 (uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
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

// __device__ void stmatrix_x4_m8n8(uint32_t smem_dst, )

//   int groupID = threadIdx.x >> 2;
//   int threadID_in_group = threadIdx.x % 4;
  
//   smemC[groupID*8 + threadID_in_group*2] = d[0];
//   smemC[groupID*8 + threadID_in_group*2 + 1] = d[1];
//   smemC[(groupID+8)*8 + threadID_in_group*2] = d[2];
//   smemC[(groupID+8)*8 + threadID_in_group*2 + 1] = d[3];

// row =      groupID                           for ci where i <  2
//          groupID + 8                         for ci where i >= 2

// col =  (threadID_in_group * 2) + (i & 0x1)    for ci where i = {0,..,3}

  // This requires sm_90 or above lmao
  // int32_t *c = smemC + (threadIdx.x & 0x1f) * 16;
  // uint32_t smem_c = static_cast<uint32_t>(__cvta_generic_to_shared(c));  
  // asm volatile ("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
  //     :: "r"(smem_c),
  //        "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
