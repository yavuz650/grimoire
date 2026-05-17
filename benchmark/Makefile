SOURCES := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, $(BUILD_DIR)/benchmark/%, $(SOURCES))

all: $(TARGETS)

$(BUILD_DIR)/benchmark/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

