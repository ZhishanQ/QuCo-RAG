# QuCo-RAG

## Overview

## Run QuCo-RAG Experiments

### Prerequisites

#### Step 1: Environment Setup


#### Step 2: Wikipedia dump download & BM25 Indexing


Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

**Index Method 1:** the Wikipedia dump using BM25:


Use Elasticsearch to index the Wikipedia dump:

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python tools/prep_elastic_index_with_progress.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

When you run Elasticsearch successfully, you should see the cluster status by running:
```bash
curl -X GET "localhost:9200/_cluster/health?pretty"
```

The indexing process will display progress logs and a progress bar. Upon completion, you'll see a success rate report.

**Index Method 2 (Recommended):** use pre-built BM25 index:

We provide a pre-built BM25 index that you can download directly from our HuggingFace. This method is faster as it skips the indexing process.

First, download and extract Elasticsearch:
```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd ..
```

Then run the startup script which will automatically download the pre-built index from HuggingFace and start Elasticsearch:
```bash
bash Start_Elasticsearch_from_hf.sh
```

The script will:
1. Check if Elasticsearch is already running
2. Download the pre-built index (~10GB) from [🤗 ZhishanQ/QuCo-RAG-es-data-archive](https://huggingface.co/datasets/ZhishanQ/QuCo-RAG-es-data-archive)
3. Extract and start Elasticsearch with the index
4. Verify the index is working correctly

> **For HPC users:** If you are running on an HPC cluster with fast local SSD storage at `/tmp`, use the optimized script for better I/O performance:
> ```bash
> bash Start_Elasticsearch_from_hf_HPC.sh
> ```
> Note: Data stored in `/tmp` will be lost when the job ends.


#### Step 3: Download Dataset

For 2WikiMultihopQA:

Download the [2WikiMultihop](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.


For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### Run
All the configuration files of our experiments are in the `config` folder.  
You can run QuCo-RAG with them directly. For example, if you want to run QuCo-RAG on 2WikiMultihopQA with OLMo-2-7B, first, you need to get into `src` folder, then run:

```bash
python main.py -c 
```


If you don't have model weight locally, the above command will download it from Hugging Face Hub first.  


**We're preparing our code and readme and will release them soon. Stay tuned for more updates!**