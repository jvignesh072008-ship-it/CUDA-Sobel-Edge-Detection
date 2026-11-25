#!/bin/bash
# Usage: ./run.sh [input_dir] [output_dir] [block_size] [max_images]
INPUT_DIR=${1:-data}
OUTPUT_DIR=${2:-output}
BLOCK_SIZE=${3:-16}
MAX_IMAGES=${4:-0}

mkdir -p "$OUTPUT_DIR"
./bin/sobel_batch --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" \
  --block_size "$BLOCK_SIZE" --max_images "$MAX_IMAGES" --save_gray \
  2>&1 | tee "$OUTPUT_DIR/run_log.txt"
