# QuCo-RAG

[![arXiv](https://img.shields.io/badge/arXiv-2512.19134-b31b1b.svg)](https://arxiv.org/abs/2512.19134)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation**
> 
> Dehai Min, Kailin Zhang, Tongtong Wu, Lu Cheng
>
> [[Paper]](https://arxiv.org/abs/2512.19134) [[PDF]](https://arxiv.org/pdf/2512.19134)

## Overview

**QuCo-RAG** is a dynamic Retrieval-Augmented Generation method that determines **when to retrieve** based on objective statistics from pre-training data, rather than relying on model-internal signals (e.g., logits, entropy) which are often unreliable due to LLM miscalibration.

### Key Features

- 🎯 **Corpus-Grounded Uncertainty Quantification**: Uses Infini-gram to query entity frequencies and co-occurrences in a 4-trillion-token corpus
- ⚡ **Two-Stage Retrieval Triggering**: 
  - *Before generation*: Identifies low-frequency entities indicating long-tail knowledge gaps
  - *During generation*: Verifies entity co-occurrence to detect hallucination risk
- 🔄 **Model-Agnostic**: Works with OLMo, Llama, Qwen, and even GPT models
- 📈 **Strong Performance**: Achieves EM gains of 5-12 points over SOTA baselines on multi-hop QA benchmarks

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [Wikipedia Dump & BM25 Index](#wikipedia-dump--bm25-index)
  - [Datasets](#datasets)
- [Running Experiments](#running-experiments)
  - [Run QuCo-RAG](#run-quco-rag)
  - [Run Baselines](#run-baselines)
- [Evaluation](#evaluation)
- [Available Configurations](#available-configurations)
- [Important Notes](#important-notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

```bash
# Clone the repository
git clone https://github.com/ZhishanQ/QuCo-RAG.git
cd QuCo-RAG

# If you already cloned the repository, pull the latest updates
git pull origin main

# Create and activate conda environment
conda create -n quco-rag python=3.9
conda activate quco-rag

# Install PyTorch with CUDA support (required first)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Download spaCy English language model
python -m spacy download en_core_web_sm
```

## Data Preparation

### Prerequisites

Before setting up the BM25 index, you need to download two things:

**1. Download Elasticsearch 7.17.9**

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
cd ..
```

**2. Download Wikipedia Dump**

Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR):

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr && gzip -d psgs_w100.tsv.gz && popd
```

### Wikipedia BM25 Index

Choose **ONE** of the following options to set up the BM25 index. You only need to do this **ONCE**.

**Option 1: Build index from scratch** (~2-3 hours)

```bash
# Start Elasticsearch
cd data/elasticsearch-7.17.9
nohup bin/elasticsearch &
cd ../..

# Wait for ES to start, then verify it's running
curl localhost:9200

# Build the index
python tools/prep_elastic_index_with_progress.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

**Option 2: Download pre-built index (Recommended)**

We provide a pre-built BM25 index (~10GB) on HuggingFace for quick setup:

```bash
bash Start_Elasticsearch_from_hf.sh
```

The script will:
1. Ask you to confirm the configuration (ES directory and URL)
2. Check if index already exists (skip download if yes)
3. Download the pre-built index from [🤗 ZhishanQ/QuCo-RAG-es-data-archive](https://huggingface.co/datasets/ZhishanQ/QuCo-RAG-es-data-archive)
4. Start Elasticsearch and verify the index

> **For HPC users**: Use `bash Start_Elasticsearch_from_hf_HPC.sh` for better I/O performance with local SSD storage.  
> ⚠️ **Warning**: HPC mode stores data in `/tmp` which will be **deleted when the job ends**. You maybe need to re-download for each new job.

### Elasticsearch Lifecycle

**Understanding the workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  FIRST TIME SETUP (do once)                                      │
│  ─────────────────────────────                                   │
│  Option 1: Build index from scratch                              │
│    → python tools/prep_elastic_index_with_progress.py            │
│                                                                  │
│  Option 2: Download pre-built index                              │
│    → bash Start_Elasticsearch_from_hf.sh                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  RUNNING EXPERIMENTS                                             │
│  ───────────────────                                             │
│  1. Start ES service:  bash start_es.sh                          │
│  2. Run experiments:   python main_quco.py -c ...                │
│  3. Stop ES service:   pkill -f elasticsearch                    │
│                                                                  │
│  The index is persistent - no need to rebuild/re-download!       │
│  (Exception: HPC mode stores in /tmp, needs re-download per job period) │
└─────────────────────────────────────────────────────────────────┘
```

**Available scripts:**
| Script | Purpose | When to use |
|--------|---------|-------------|
| `Start_Elasticsearch_from_hf.sh` | Download pre-built index + start ES | First time setup (downloads index once) |
| `Start_Elasticsearch_from_hf_HPC.sh` | Same as above, but uses local SSD | First time on HPC (re-download each job) |
| `start_es.sh` | Start ES with existing index | Before experiments if ES was stopped or after reboot |

**Stop Elasticsearch when your experiment done:**
```bash
# Elasticsearch consumes memory even when idle
pkill -f elasticsearch
```

### Datasets

**2WikiMultihopQA**

Download from [Dropbox](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1), unzip, and move to `data/2wikimultihopqa`. (You can just keep two files: `dev.json` and `id_aliases.json`.)

**HotpotQA**

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

## Running Experiments

### Verify Elasticsearch

**🚨 Make sure Elasticsearch is running before running RAG experiments.**

```bash
# The script automatically checks if ES is running and starts it if needed
bash start_es.sh
```

The script will automatically:
- Check if Elasticsearch is already running
- If yes: Display status and exit
- If no: Start Elasticsearch and verify the index

You can also manually check the status:

```bash
# Check if Elasticsearch is running
curl -X GET "localhost:9200/"

# Check cluster health
curl -X GET "localhost:9200/_cluster/health?pretty"

# Check all indices
curl -X GET "localhost:9200/_cat/indices?v"

# Check wiki index document count
curl -X GET "localhost:9200/wiki/_count?pretty"
```

**Expected output when Elasticsearch is running successfully:**

```bash
# Cluster health should show "green" status
{
  "cluster_name" : "elasticsearch",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 1,
  "number_of_data_nodes" : 1,
  "active_primary_shards" : 4,
  "active_shards" : 4,
  "relocating_shards" : 0,
  "initializing_shards" : 0,
  "unassigned_shards" : 0,
  "delayed_unassigned_shards" : 0,
  "number_of_pending_tasks" : 0,
  "number_of_in_flight_fetch" : 0,
  "task_max_waiting_in_queue_millis" : 0,
  "active_shards_percent_as_number" : 100.0
}

# Indices status should show wiki index with ~21M documents
health status index            uuid                   pri rep docs.count docs.deleted store.size pri.store.size
green  open   wiki             LqVIlRacS6C2S7CyrIfw7g   1   0   21015324            0     11.3gb         11.3gb

# Wiki index count should be approximately 21 million
{"count":21015324,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}}
```

If Elasticsearch is running correctly, you should see cluster status as "green" or "yellow", and the wiki index should contain approximately 21 million documents.

### Run QuCo-RAG

All the configuration files of our experiments are in the `config` folder. You can run QuCo-RAG with them directly. For example, if you want to run QuCo-RAG on 2WikiMultihopQA with OLMo-2-7B:

```bash
cd src
python main_quco.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/QuCo-RAG.json
```

If you don't have model weights locally, the above command will download them from Hugging Face Hub first.

If you see log messages like below, it means QuCo-RAG is running successfully:

```
2025-12-24 00:01:55 - __main__ - INFO - Namespace(model_name_or_path='allenai/OLMo-2-1124-7B-Instruct', method='QuCo-RAG', dataset='2wikimultihopqa', ...)
2025-12-24 00:01:55 - __main__ - INFO - ==================== config ********************
2025-12-24 00:01:55 - __main__ - INFO - config path: ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/QuCo-RAG.json
2025-12-24 00:01:55 - __main__ - INFO - ==================== output dir ********************
2025-12-24 00:01:55 - __main__ - INFO - output dir: ../result/QuCo-RAG_OLMo-2-1124-7B-Instruct_2wikimultihopqa/1
2025-12-24 00:01:55 - data - INFO - Loading WikiMultiHopQA from ../data/2wikimultihopqa (split: dev)
2025-12-24 00:01:57 - __main__ - INFO - sample 1000 data points from the beginning
2025-12-24 00:01:57 - __main__ - INFO - data size: 1000
2025-12-24 00:01:57 - generate_quco - INFO - Loading model 'allenai/OLMo-2-1124-7B-Instruct' with device_map: auto
Loading checkpoint shards: 100%|██████████| 3/3 [00:10<00:00,  3.36s/it]
2025-12-24 00:02:10 - root - INFO - Activating Elasticsearch....
2025-12-24 00:02:10 - root - INFO - Elastic Search Credentials: {'hostname': 'localhost', 'index_name': 'wiki', ...}
2025-12-24 00:02:10 - generate_quco - INFO - Using local entity extraction model: ZhishanQ/QuCo-extractor-0.5B
2025-12-24 00:04:05 - generate_quco - INFO - Local entity extraction model loaded successfully
2025-12-24 00:04:05 - generate_quco - INFO - Using prompt template key: new_prompt_v3
2025-12-24 00:04:05 - __main__ - INFO - model type: <class 'generate_quco.QuCo_RAG'>
2025-12-24 00:04:05 - __main__ - INFO - start inference
  0%|          | 0/1000 [00:00<?, ?it/s]
  0%|          | 1/1000 [00:24<6:54:28, 24.89s/it]
```

The output will be saved in the `result/` folder. You can change the output folder by modifying the `output_dir` parameter in the configuration file.

### Run Baselines

We also provide implementations of baseline methods for comparison. You can run them using the corresponding configuration files:

```bash
cd src

# Single Retrieval RAG (SR-RAG)
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/SR-RAG.json

# Fix-Length RAG (FL-RAG)
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/FL-RAG.json

# Fix-Sentence RAG (FS-RAG)
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/FS-RAG.json

# FLARE
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/FLARE.json

# DRAGIN
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/DRAGIN.json

# Without RAG (wo-RAG)
python main_baseline.py -c ../config/OLMo-2-1124-7B-Instruct/2WikiMultihopQA/wo-RAG.json
```

## Evaluation

Upon completion of the program, you will find a folder named with a numerical identifier within your designated output directory. This identifier corresponds to the sequential order of runs within that folder, allowing for easy organization of multiple executions.

To evaluate the results, you can use the `evaluate.py` script in the `src` folder. Assume the output folder is `result/QuCo-RAG_OLMo-2-1124-7B-Instruct_2wikimultihopqa/1`, you can run:

```bash
python evaluate.py --dir ../result/QuCo-RAG_OLMo-2-1124-7B-Instruct_2wikimultihopqa/1
```

After the evaluation, you will see the evaluation results in the program output and in the output directory:
```
result/QuCo-RAG_OLMo-2-1124-7B-Instruct_2wikimultihopqa/1/
├── config.json          # Configuration used for this run
├── output.txt           # Raw predictions with statistics
├── result.tsv           # EM and F1 scores
├── details.txt          # Per-sample evaluation details
└── retrieved_docs.json  # Retrieved documents (useful for debugging)
```

## Available Configurations

We provide configuration files for the following models and datasets:

**Models:**
- `OLMo-2-1124-7B-Instruct`
- `OLMo-2-1124-13B-Instruct`
- `OLMo-2-0325-32B-Instruct`
- `Meta-Llama-3-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen2.5-32B-Instruct`

**Datasets:**
- `2WikiMultihopQA`
- `HotpotQA`

All configurations are in the `config/` folder.

## Important Notes

1. **GPU Requirements**: Our experiments can be conducted on NVIDIA GPUs like A40/A100/H100/H200. Make sure you have sufficient GPU memory (at least ~36GB for 7B models).

2. **Elasticsearch Management**: Always ensure Elasticsearch is running before starting experiments. Use the commands in the "Verify Elasticsearch" section to verify. ES consumes memory even when idle, so stop it when not in use.
   
   To stop Elasticsearch when you're done:
   ```bash
   # Find Elasticsearch process
   ps aux | grep elasticsearch
   
   # Kill the process (replace <PID> with the actual process ID)
   kill <PID>
   
   # Or force kill if needed
   kill -9 <PID>
   
   # Alternative: kill all Elasticsearch processes at once
   pkill -f elasticsearch
   ```

3. **Reproducibility**: To reproduce our reported results, please use the exact configuration files provided in the `config` folder without modifications. Make sure to use our knowledge triple extractor from [🤗 ZhishanQ/QuCo-extractor-0.5B](https://huggingface.co/ZhishanQ/QuCo-extractor-0.5B).

## Optional: Alternative Retriever

We provide scripts for encoding corpus with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) using vLLM for faster encoding:

```bash
# Install vLLM if not already installed
pip install vllm>=0.8.5

# Encode corpus
cd tools
bash encode_qwen3_vllm.sh
```

The scripts are located in `tools/`:
- `encode_qwen3_vllm.sh` - Shell script to run encoding
- `encode_qwen3_vllm.py` - Python script for vLLM-based encoding

> **Note**: This is optional. The default BM25 retriever works well for most cases.

## Citation

If you find this work useful, please cite our paper.

## Acknowledgements

We thank the authors of the following projects for their excellent work:
- [DRAGIN](https://github.com/oneal2000/DRAGIN)
- [ETC](https://github.com/pkuserc/ETC)
- [Infini-gram](https://infini-gram.io/)