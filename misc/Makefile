SOURCES := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, $(BUILD_DIR)/misc/%, $(SOURCES))

all: $(TARGETS)

$(BUILD_DIR)/misc/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

# Stop make from trying to build the headers?
../include/%.cuh: ;

clean:
	rm -f $(TARGETS)

