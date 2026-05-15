SOURCES := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, $(BUILD_DIR)/tests/%, $(SOURCES))

all: $(TARGETS)

$(BUILD_DIR)/tests/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

$(BUILD_DIR)/tests/fmha_fwd: fmha_fwd.cu ../include/fmha_fwd.cuh ../include/utils.cuh ../include/buffer.cuh 

# Stop make from trying to build the headers?
../include/%.cuh: ;

clean:
	rm -f $(TARGETS)

