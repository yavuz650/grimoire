SOURCES := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, $(BUILD_DIR)/tests/%, $(SOURCES))

all: $(TARGETS)

$(BUILD_DIR)/tests/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/fmha_fwd: fmha_fwd.cu ../include/fmha_fwd.cuh ../include/utils.hpp ../include/buffer.cuh
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/mma_f16_f16_tma: mma_f16_f16_tma.cu ../include/sm120_tma_gemm.cuh ../include/utils.hpp ../include/buffer.cuh
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/mma_f16_f16_tma_ws: mma_f16_f16_tma_ws.cu ../include/sm120_tma_gemm.cuh ../include/utils.hpp ../include/buffer.cuh
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/mma_f16_f16_tma_ws_persistent: mma_f16_f16_tma_ws_persistent.cu ../include/sm120_tma_gemm.cuh ../include/utils.hpp ../include/buffer.cuh
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/cuda_tile_mma: cuda_tile_mma.cu ../include/cuda_tile_mma.cuh ../include/utils.hpp ../include/buffer.cuh
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) --enable-tile $< -o $@

clean:
	rm -f $(TARGETS)

