# Common flags
NVCC        := nvcc
ARCH        := -arch=sm_120a
CXX_FLAGS    := -O3 -std=c++20
INCLUDES    := -I$(CURDIR) -I$(CURDIR)/cutlass/include -I$(CURDIR)/cutlass/tools/util/include
NVCC_FLAGS   := $(CXX_FLAGS) $(ARCH) $(INCLUDES) --expt-relaxed-constexpr -lcuda -lineinfo -g
# Allow the user to specify additional flags
# Example: make OPT_FLAGS=-O2
OPT_FLAGS ?=
# Directory for build artifacts
BUILD_DIR   := $(CURDIR)/build
export NVCC ARCH CXX_FLAGS INCLUDES NVCC_FLAGS OPT_FLAGS BUILD_DIR

.PHONY: all benchmark tests misc clean

all: benchmark tests misc 

benchmark:
	mkdir -p $(BUILD_DIR)/benchmark
	$(MAKE) -C benchmark

tests:
	mkdir -p $(BUILD_DIR)/tests
	$(MAKE) -C tests

misc:
	mkdir -p $(BUILD_DIR)/misc
	$(MAKE) -C misc

clean:
	rm -rf $(BUILD_DIR)

