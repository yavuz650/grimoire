NVCC        := nvcc
ARCH        := -arch=sm_86
CXXFLAGS    := -O3 -std=c++17
INCLUDES    := -I.. -I ../cutlass/include -I ../cutlass/tools/util/include
NVCCFLAGS   := $(CXXFLAGS) $(ARCH) $(INCLUDES) --expt-relaxed-constexpr

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu ../include/gemm.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
