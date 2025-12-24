#!/bin/bash
# Download pre-built BM25 index from HuggingFace and start Elasticsearch
# This script only needs to run ONCE to download and set up the index.
# After that, use start_es.sh to start Elasticsearch service.

echo "=========================================="
echo "QuCo-RAG: Download Pre-built Index"
echo "=========================================="

# Set path variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_DIR="$SCRIPT_DIR/data/elasticsearch-7.17.9"
DATA_DIR="$ES_DIR/data"
LOGS_DIR="$ES_DIR/logs"
HF_REPO="ZhishanQ/QuCo-RAG-es-data-archive"
HF_FILE="es-data-archive.tar.gz"

echo ""
echo "Configuration:"
echo "  Elasticsearch directory: $ES_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Logs directory: $LOGS_DIR"
echo "  Elasticsearch URL: http://localhost:9200"
echo ""

# Ask user to confirm configuration
read -p "Is this configuration correct? [Y/n] " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted. Please modify the script if needed."
    exit 0
fi

# Check if Elasticsearch is already running
echo ""
echo "Checking for existing Elasticsearch instance..."
if curl -s localhost:9200 >/dev/null 2>&1; then
    echo ""
    echo "=========================================="
    echo "✓ Elasticsearch is already running!"
    echo "=========================================="
    
    echo ""
    echo "=== Elasticsearch Status ==="
    curl -X GET "localhost:9200/" 2>/dev/null
    echo ""
    
    echo "=== Cluster Health ==="
    curl -X GET "localhost:9200/_cluster/health?pretty" 2>/dev/null
    
    echo "=== Indices Status ==="
    curl -s 'localhost:9200/_cat/indices?v' 2>/dev/null
    
    echo ""
    echo "=== Wiki Index Count ==="
    curl -s 'localhost:9200/wiki/_count' 2>/dev/null
    echo ""
    
    echo "=========================================="
    echo "Using existing Elasticsearch instance."
    echo "=========================================="
    exit 0
fi

echo "No running Elasticsearch instance found."

# Ensure ES directory exists
if [ ! -d "$ES_DIR" ]; then
    echo ""
    echo "❌ Error: Elasticsearch directory not found at $ES_DIR"
    echo ""
    echo "Please download Elasticsearch first:"
    echo "  cd data"
    echo "  wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz"
    echo "  tar zxvf elasticsearch-7.17.9.tar.gz"
    echo "  cd .."
    exit 1
fi

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOGS_DIR"

# Check if index data already exists
if [ -d "$DATA_DIR/nodes" ]; then
    echo ""
    echo "=========================================="
    echo "✓ Index data already exists at $DATA_DIR"
    echo "=========================================="
    echo ""
    echo "Skipping download. Starting Elasticsearch with existing data..."
else
    # Download archive from HuggingFace
    echo ""
    echo "=========================================="
    echo "Downloading pre-built index from HuggingFace..."
    echo "=========================================="
    echo "Repository: $HF_REPO"
    echo "File: $HF_FILE (~10GB)"
    echo "Destination: $ES_DIR"
    echo ""

    # Check if archive already exists
    cd "$ES_DIR"
    if [ -f "$HF_FILE" ]; then
        echo "Archive file already exists: $HF_FILE"
        echo "Archive size: $(ls -lh $HF_FILE | awk '{print $5}')"
        read -p "Use existing archive? [Y/n] " use_existing
        use_existing=${use_existing:-Y}
        
        if [[ ! "$use_existing" =~ ^[Yy]$ ]]; then
            echo "Removing existing archive and re-downloading..."
            rm -f "$HF_FILE"
        fi
    fi

    # Download if archive doesn't exist
    if [ ! -f "$HF_FILE" ]; then
        huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir . --repo-type dataset

        if [ $? -eq 0 ] && [ -f "$HF_FILE" ]; then
            echo "✓ Archive downloaded successfully."
            echo "Archive size: $(ls -lh $HF_FILE | awk '{print $5}')"
        else
            echo ""
            echo "❌ Failed to download archive from HuggingFace."
            echo ""
            echo "Please check:"
            echo "1. You are logged in: huggingface-cli login"
            echo "2. You have access to the repository: $HF_REPO"
            exit 1
        fi
    fi

    # Extract archive
    echo ""
    echo "Extracting archive to $DATA_DIR..."
    echo "This may take a few minutes..."
    tar -xzf "$HF_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Index data extracted successfully."
        echo "Data directory size: $(du -sh "$DATA_DIR" | cut -f1)"
        echo ""
        echo "NOTE: Archive file kept at $ES_DIR/$HF_FILE"
        echo "      You can delete it manually to save space: rm $ES_DIR/$HF_FILE"
    else
        echo "❌ Failed to extract archive. Exiting."
        exit 1
    fi
fi

# Start Elasticsearch
echo ""
echo "=========================================="
echo "Starting Elasticsearch..."
echo "=========================================="
echo "Data: $DATA_DIR"
echo "Logs: $LOGS_DIR"

cd "$ES_DIR"
nohup bin/elasticsearch \
    -E path.data="$DATA_DIR" \
    -E path.logs="$LOGS_DIR" > "$LOGS_DIR/startup.log" 2>&1 &

ES_PID=$!
echo "Elasticsearch started with PID: $ES_PID"

# Wait for ES to start
echo ""
echo "Waiting for Elasticsearch to start..."
for i in {1..120}; do 
    if curl -s localhost:9200 >/dev/null 2>&1; then
        echo "✓ Elasticsearch is up and running!"
        break
    fi
    if [ $((i % 20)) -eq 0 ]; then
        echo "Still waiting... ($i/120 x 5 seconds)"
    fi
    sleep 5
done

# Check if ES started successfully
if ! curl -s localhost:9200 >/dev/null 2>&1; then 
    echo ""
    echo "❌ Elasticsearch failed to start after 10 minutes"
    echo ""
    echo "=== Startup log ==="
    tail -50 "$LOGS_DIR/startup.log"
    exit 1
fi

echo ""
echo "=== Elasticsearch Status ==="
curl -X GET "localhost:9200/" 2>/dev/null
echo ""

# Fix replica settings for wiki index (single node doesn't need replicas)
sleep 2
echo "=== Fixing replica settings for single node ==="
curl -X PUT "localhost:9200/wiki/_settings" -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 0}}' 2>/dev/null
echo ""

# Wait for shards to be reassigned
echo "Waiting for shards to be assigned..."
sleep 5

echo ""
echo "=== Cluster Health ==="
curl -X GET "localhost:9200/_cluster/health?pretty" 2>/dev/null

echo ""
echo "=== Indices Status ==="
curl -s 'localhost:9200/_cat/indices?v' 2>/dev/null

echo ""
echo "=== Wiki Index Count ==="
curl -s 'localhost:9200/wiki/_count' 2>/dev/null
echo ""

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Elasticsearch is now running at: http://localhost:9200"
echo "Wiki index contains ~21M documents."
echo ""
echo "IMPORTANT:"
echo "  - To stop Elasticsearch: pkill -f elasticsearch"
echo "  - To restart later: bash start_es.sh"
echo "  - The index is persistent. You only need to download it ONCE."
echo "=========================================="