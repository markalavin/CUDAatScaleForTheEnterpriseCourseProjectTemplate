# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
COMMON = ./NPP/Common
UtilNPP = $(COMMON)/UtilNPP

CXXFLAGS = -std=c++11 -I/usr/local/cuda/include -Iinclude -I$(UtilNPP) -I$(COMMON) -I./3rdParty
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

# AAA.004: Automatically find all .cpp files and define their targets
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
TARGETS = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SOURCES))

# Default rule builds everything in the TARGETS list
all: $(TARGETS)

# Generic rule: How to build ANY target in bin/ from a .cpp in src/
$(BIN_DIR)/%: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Building application: $@"
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Rule for running the Median Filter specifically
run_median: $(BIN_DIR)/imageMedianFilterNPP
	./$(BIN_DIR)/imageMedianFilterNPP --input $(DATA_DIR)/Lena.png --output $(DATA_DIR)/Lena_median_filter.png

# Rule for running the original Rotation
run_rotation: $(BIN_DIR)/imageRotationNPP
	./$(BIN_DIR)/imageRotationNPP --input $(DATA_DIR)/Lena.png --output $(DATA_DIR)/Lena_rotated.png

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

help:
	@echo "Available commands:"
	@echo "  make             - Build all apps in src/"
	@echo "  make run_median  - Build and run Median Filter"
	@echo "  make run_rotation- Build and run Rotation"
	@echo "  make clean       - Remove all binaries"b