#!/bin/bash

# Corpus encoding script using Qwen3-Embedding with vLLM

echo "Starting corpus encoding with Qwen3-Embedding-0.6B (vLLM version)..."

# Set paths - can use HuggingFace model name or local path
MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
# Or use local path:
# MODEL_PATH="/projects/beyd/models/Qwen_Qwen3-Embedding-0.6B"

CORPUS_FILE="data/dpr/psgs_w100.tsv"
OUTPUT_DIR="qwen3_vllm/encode_result"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if corpus file exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo "Error: Corpus file does not exist: $CORPUS_FILE"
    exit 1
fi

# Check if vLLM is installed
python -c "import vllm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: vLLM not installed. Please install first:"
    echo "pip install vllm>=0.8.5"
    exit 1
fi

echo "Checking vLLM version..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo "Testing encoding functionality first..."
python encode_qwen3_vllm.py \
    --model_path "$MODEL_PATH" \
    --test_only

if [ $? -ne 0 ]; then
    echo "Test failed, please check model and environment"
    exit 1
fi

echo "Test successful, starting full encoding..."

# Get GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

# Set tensor_parallel_size based on GPU count
if [ "$GPU_COUNT" -gt 1 ]; then
    TENSOR_PARALLEL_SIZE=$GPU_COUNT
    echo "Using $TENSOR_PARALLEL_SIZE GPUs for tensor parallelism"
else
    TENSOR_PARALLEL_SIZE=1
    echo "Using single GPU"
fi

# Run encoding
python encode_qwen3_vllm.py \
    --model_path "$MODEL_PATH" \
    --corpus_file "$CORPUS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --chunk_size 40000 \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization 0.8

# Check if encoding succeeded
if [ $? -eq 0 ]; then
    echo "Qwen3-Embedding encoding completed!"
    echo "Output saved in: $OUTPUT_DIR"
else
    echo "Error occurred during encoding!"
    exit 1
fi
