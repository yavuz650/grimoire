NVCC        := nvcc
ARCH        := -arch=sm_120a
CXXFLAGS    := -O3 -std=c++17
INCLUDES    := -I.. -I ../cutlass/include -I ../cutlass/tools/util/include
NVCCFLAGS   := $(CXXFLAGS) $(ARCH) $(INCLUDES) --expt-relaxed-constexpr -lcuda

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu ../include/sm86_gemm.cuh ../include/sm120_tma_gemm.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
