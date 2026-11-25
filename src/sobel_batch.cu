// Copyright 2026 <Vignesh J>
//
// sobel_batch.cu: Batch Sobel edge detection on the GPU using CUDA.
//
// Reads every image in an input directory (recursively), converts it to
// grayscale, applies a 3x3 Sobel filter on the GPU and writes the edge map.

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

namespace fs = std::filesystem;

constexpr int kDefaultBlockSize = 16;
constexpr int kMaxBlockSize = 32;  // block_size^2 must not exceed 1024 threads.

struct Options {
  std::string input_dir = "data";
  std::string output_dir = "output";
  int block_size = kDefaultBlockSize;
  int max_images = 0;  // 0 means process every image found.
  bool save_gray = false;
};

struct ImageResult {
  int width = 0;
  int height = 0;
  float kernel_ms = 0.0f;
};

void CheckCuda(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", what,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

// Owns the two device buffers and grows them when a larger image arrives.
class DeviceBuffers {
 public:
  DeviceBuffers() = default;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  ~DeviceBuffers() {
    cudaFree(input_);
    cudaFree(output_);
  }

  void Reserve(size_t bytes) {
    if (bytes <= capacity_) {
      return;
    }
    cudaFree(input_);
    cudaFree(output_);
    CheckCuda(cudaMalloc(&input_, bytes), "cudaMalloc input");
    CheckCuda(cudaMalloc(&output_, bytes), "cudaMalloc output");
    capacity_ = bytes;
  }

  unsigned char* input() const { return input_; }
  unsigned char* output() const { return output_; }

 private:
  unsigned char* input_ = nullptr;
  unsigned char* output_ = nullptr;
  size_t capacity_ = 0;
};

// Applies the 3x3 Sobel operator. Border pixels are set to zero.
__global__ void SobelKernel(const unsigned char* __restrict__ input,
                            unsigned char* __restrict__ output, int width,
                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[idx] = 0;
    return;
  }

  const int top_left = input[idx - width - 1];
  const int top = input[idx - width];
  const int top_right = input[idx - width + 1];
  const int left = input[idx - 1];
  const int right = input[idx + 1];
  const int bottom_left = input[idx + width - 1];
  const int bottom = input[idx + width];
  const int bottom_right = input[idx + width + 1];

  const int gx = -top_left + top_right - 2 * left + 2 * right - bottom_left +
                 bottom_right;
  const int gy = -top_left - 2 * top - top_right + bottom_left + 2 * bottom +
                 bottom_right;
  const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
  output[idx] = static_cast<unsigned char>(fminf(magnitude, 255.0f));
}

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --input_dir DIR    folder with images, searched recursively "
      "(default: data)\n"
      "  --output_dir DIR   folder for results (default: output)\n"
      "  --block_size N     CUDA block is N x N threads, 1-%d (default: %d)\n"
      "  --max_images N     process at most N images, 0 = all (default: 0)\n"
      "  --save_gray        also save the grayscale input as <name>_gray.png\n"
      "  --help             show this message\n",
      program, kMaxBlockSize, kDefaultBlockSize);
}

bool ParseArguments(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (arg == "--save_gray") {
      options->save_gray = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << arg << "\n";
      return false;
    }
    const std::string value = argv[++i];
    if (arg == "--input_dir") {
      options->input_dir = value;
    } else if (arg == "--output_dir") {
      options->output_dir = value;
    } else if (arg == "--block_size") {
      options->block_size = std::atoi(value.c_str());
    } else if (arg == "--max_images") {
      options->max_images = std::atoi(value.c_str());
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }
  if (options->block_size < 1 || options->block_size > kMaxBlockSize) {
    std::cerr << "--block_size must be between 1 and " << kMaxBlockSize
              << "\n";
    return false;
  }
  if (options->max_images < 0) {
    std::cerr << "--max_images must be 0 or greater\n";
    return false;
  }
  return true;
}

bool IsImageFile(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
         extension == ".bmp" || extension == ".tga" || extension == ".pgm" ||
         extension == ".ppm";
}

std::vector<fs::path> FindImages(const std::string& directory) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file() && IsImageFile(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

// Loads one image as grayscale, runs the Sobel kernel and saves the results.
bool ProcessImage(const fs::path& path, const Options& options,
                  DeviceBuffers* buffers, ImageResult* result) {
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char* pixels =
      stbi_load(path.string().c_str(), &width, &height, &channels, 1);
  if (pixels == nullptr) {
    return false;
  }

  const size_t bytes = static_cast<size_t>(width) * height;
  buffers->Reserve(bytes);
  CheckCuda(cudaMemcpy(buffers->input(), pixels, bytes,
                       cudaMemcpyHostToDevice),
            "copy input to device");

  const dim3 block(options.block_size, options.block_size);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

  cudaEvent_t start;
  cudaEvent_t stop;
  CheckCuda(cudaEventCreate(&start), "create start event");
  CheckCuda(cudaEventCreate(&stop), "create stop event");
  CheckCuda(cudaEventRecord(start), "record start event");
  SobelKernel<<<grid, block>>>(buffers->input(), buffers->output(), width,
                               height);
  CheckCuda(cudaEventRecord(stop), "record stop event");
  CheckCuda(cudaEventSynchronize(stop), "synchronize stop event");
  CheckCuda(cudaGetLastError(), "launch SobelKernel");
  CheckCuda(cudaEventElapsedTime(&result->kernel_ms, start, stop),
            "read elapsed time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::vector<unsigned char> edges(bytes);
  CheckCuda(cudaMemcpy(edges.data(), buffers->output(), bytes,
                       cudaMemcpyDeviceToHost),
            "copy output to host");

  const fs::path out_dir(options.output_dir);
  const std::string stem = path.stem().string();
  const std::string sobel_path = (out_dir / (stem + "_sobel.png")).string();
  stbi_write_png(sobel_path.c_str(), width, height, 1, edges.data(), width);
  if (options.save_gray) {
    const std::string gray_path = (out_dir / (stem + "_gray.png")).string();
    stbi_write_png(gray_path.c_str(), width, height, 1, pixels, width);
  }

  stbi_image_free(pixels);
  result->width = width;
  result->height = height;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArguments(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
  if (!fs::is_directory(options.input_dir)) {
    std::cerr << "Input directory not found: " << options.input_dir << "\n";
    return EXIT_FAILURE;
  }
  fs::create_directories(options.output_dir);

  cudaDeviceProp properties;
  CheckCuda(cudaGetDeviceProperties(&properties, 0), "get device properties");
  CheckCuda(cudaSetDevice(0), "set device");
  std::printf("Using device 0: %s\n", properties.name);

  std::vector<fs::path> images = FindImages(options.input_dir);
  if (options.max_images > 0 &&
      images.size() > static_cast<size_t>(options.max_images)) {
    images.resize(options.max_images);
  }
  std::printf("Found %zu images in %s (block size %dx%d)\n", images.size(),
              options.input_dir.c_str(), options.block_size,
              options.block_size);

  DeviceBuffers buffers;
  int processed = 0;
  int failed = 0;
  double total_kernel_ms = 0.0;
  const auto wall_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < images.size(); ++i) {
    ImageResult result;
    if (!ProcessImage(images[i], options, &buffers, &result)) {
      std::printf("[%zu/%zu] FAILED to read %s\n", i + 1, images.size(),
                  images[i].string().c_str());
      ++failed;
      continue;
    }
    ++processed;
    total_kernel_ms += result.kernel_ms;
    std::printf("[%zu/%zu] %s %dx%d kernel %.4f ms\n", i + 1, images.size(),
                images[i].filename().string().c_str(), result.width,
                result.height, result.kernel_ms);
  }

  const std::chrono::duration<double> wall =
      std::chrono::steady_clock::now() - wall_start;
  std::printf("Processed %d images (%d failed)\n", processed, failed);
  std::printf("Total kernel time: %.3f ms\n", total_kernel_ms);
  std::printf("Total wall time (including file I/O): %.3f s\n", wall.count());
  if (processed > 0 && wall.count() > 0.0) {
    std::printf("Throughput: %.2f images/s\n", processed / wall.count());
  }
  std::printf("Output written to %s\n", options.output_dir.c_str());
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
