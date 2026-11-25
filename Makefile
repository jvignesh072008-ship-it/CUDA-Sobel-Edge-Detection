.RECIPEPREFIX := >
NVCC ?= nvcc
ARCH ?= sm_75
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I lib
TARGET = bin/sobel_batch
SRC = src/sobel_batch.cu
STB_URL = https://raw.githubusercontent.com/nothings/stb/master

all: deps $(TARGET)

deps: lib/stb_image.h lib/stb_image_write.h

lib/stb_image.h:
> mkdir -p lib
> curl -L -o $@ $(STB_URL)/stb_image.h

lib/stb_image_write.h:
> mkdir -p lib
> curl -L -o $@ $(STB_URL)/stb_image_write.h

$(TARGET): $(SRC) lib/stb_image.h lib/stb_image_write.h
> mkdir -p bin
> $(NVCC) $(CXXFLAGS) -arch=$(ARCH) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
> rm -f $(TARGET)
> rm -rf output

.PHONY: all deps clean
