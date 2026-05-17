SOURCES := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, $(BUILD_DIR)/misc/%, $(SOURCES))

all: $(TARGETS)

$(BUILD_DIR)/misc/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $(OPT_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

