NVCC        := nvcc
ARCH        := -arch=sm_120a
CXXFLAGS    := -O3 -std=c++17
INCLUDES    := -I include -I cutlass/include -I cutlass/tools/util/include
NVCC_FLAGS   := $(CXXFLAGS) $(ARCH) $(INCLUDES) --expt-relaxed-constexpr -lcuda -lineinfo -g
# Allow the user to specify additional flags
# Example: make OPT_FLAGS=-O2
OPT_FLAGS ?=

# Directory for build artifacts
BUILD_DIR   ?= build
TARGET      := $(BUILD_DIR)/mma_test_fp16

SRC         := mma_test_fp16.cu

all: $(TARGET)

$(TARGET): $(SRC) include/sm86_gemm.cuh include/sm120_tma_gemm.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $< $(NVCC_FLAGS) $(OPT_FLAGS) -o $@ 

# Misc targets
device_query: device_query.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $< $(NVCC_FLAGS) $(OPT_FLAGS) -o $(BUILD_DIR)/$@

test_memory_latency: test_memory_latency.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $< $(NVCC_FLAGS) $(OPT_FLAGS) -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)/$(TARGET)
