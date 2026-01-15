HOST_CXX      = clang++
HOST_TARGET   = host

DPU_CC        = dpu-upmem-dpurte-clang
DPU_TARGET    = B+Tree.dpu

HOST_SRC      := host.cpp
DPU_SRC       := dpu/B+Tree.c

HOST_CXXFLAGS += -O2 -std=c++17 $(shell dpu-pkg-config --cflags dpu)
HOST_LDFLAGS  += $(shell dpu-pkg-config --libs dpu)

DPU_CFLAGS    += -O2 -DNR_TASKLETS=16 -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048

.PHONY: all clean run

all: $(HOST_TARGET) $(DPU_TARGET)

$(HOST_TARGET): $(HOST_SRC)
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $< $(HOST_LDFLAGS)

$(DPU_TARGET): $(DPU_SRC)
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $<

run: all
	./$(HOST_TARGET) $(DPU_TARGET)

clean:
	rm -f $(HOST_TARGET) $(DPU_TARGET)
