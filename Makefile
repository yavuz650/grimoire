NVCC        := nvcc
ARCH        := -arch=sm_86
CXXFLAGS    := -O3 -std=c++17
INCLUDES    := -I include -I cutlass/include -I cutlass/tools/util/include
NVCC_FLAGS   := $(CXXFLAGS) $(ARCH) $(INCLUDES) --expt-relaxed-constexpr

# Define the source and target files
SRC = mma_test_fp16.cu
TARGET = mma_test_fp16

# Allow the user to specify additional flags
# Example: make OPT_FLAGS=-O2
OPT_FLAGS ?=

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC) include/gemm.cuh
	$(NVCC) $< $(NVCC_FLAGS) $(OPT_FLAGS) -o $@ 

# Clean up generated files
clean:
	rm -f $(TARGET)
