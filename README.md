# Batch Sobel Edge Detection with CUDA

Author: Vignesh J

## Overview
This project applies Sobel edge detection to a whole folder of images on the
GPU. Each image is loaded as grayscale, and a CUDA kernel computes the
horizontal and vertical gradients with 3x3 Sobel filters (one thread per
pixel). The gradient magnitude is saved as an edge map. The program logs the
kernel time of every image and the total throughput, so it can run on
thousands of images.

## Code Organization
`bin/` built executable (`bin/sobel_batch`).
`data/` a few sample images. The large dataset is not stored in the repo.
`lib/` stb_image and stb_image_write (single-header image libraries,
downloaded by `make`).
`src/sobel_batch.cu` the CUDA program.
`Makefile` builds the project.
`run.sh` runs the program with default arguments and saves a log.
`INSTALL` install and build instructions.

## Build
    make ARCH=sm_75

Use ARCH=sm_60 for a P100 or sm_80 for an A100.

## Run
    ./bin/sobel_batch --input_dir data --output_dir output --block_size 16 --save_gray

Or use the helper script (run chmod +x run.sh once):

    ./run.sh <input_dir> <output_dir> <block_size> <max_images>

| Argument | Meaning | Default |
|---|---|---|
| --input_dir | Folder of images (searched recursively) | data |
| --output_dir | Folder for results | output |
| --block_size | Block is N x N threads (1-32) | 16 |
| --max_images | Limit the number of images, 0 = all | 0 |
| --save_gray | Also save the grayscale input as name_gray.png | off |

## Output
For each input image name.ext the program writes name_sobel.png (the edge
map, the "after" image) and, with --save_gray, name_gray.png (the grayscale
input, the "before" image). The log lists the kernel time per image and a
summary: images processed, total kernel time, total wall time and throughput.

## Lessons Learned
For small images the file reading and writing dominates the run time, so the
GPU speedup is visible mainly in the kernel timings.
