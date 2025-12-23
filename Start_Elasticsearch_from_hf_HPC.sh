#!/bin/bash
# HPC version: Uses local SSD (/tmp) for better I/O performance
echo "Starting Elasticsearch with HPC local storage optimization..."

# Check if Elasticsearch is already running
echo "Checking for existing Elasticsearch instance..."
if curl -s localhost:9200 >/dev/null 2>&1; then
    echo "=========================================="
    echo "✓ Elasticsearch is already running!"
    echo "=========================================="
    
    # Show current ES status
    echo "=== Current Elasticsearch Status ==="
    curl -X GET "localhost:9200/" 2>/dev/null
    echo ""
    
    echo "=== Cluster Health ==="
    curl -X GET "localhost:9200/_cluster/health?pretty" 2>/dev/null
    
    echo "=== Indices Status ==="
    curl -s 'localhost:9200/_cat/indices?v' 2>/dev/null
    
    echo "=== Wiki Index Count ==="
    curl -s 'localhost:9200/wiki/_count' 2>/dev/null
    echo ""
    
    echo "=========================================="
    echo "Using existing Elasticsearch instance. No need to start a new one."
    echo "=========================================="
    exit 0
fi

echo "No running Elasticsearch instance found. Starting new instance..."

# Set path variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_DIR="$SCRIPT_DIR/data/elasticsearch-7.17.9"
# HPC: Use local SSD for data storage to improve I/O performance
LOCAL_BASE_DIR="/tmp/$USER/es-$$"
LOCAL_DATA_DIR="$LOCAL_BASE_DIR/data"
LOGS_DIR="$ES_DIR/logs"
LOCAL_ARCHIVE="$LOCAL_BASE_DIR/es-archive.tar.gz"
HF_REPO="ZhishanQ/QuCo-RAG-es-data-archive"
HF_FILE="es-data-archive.tar.gz"

# Ensure ES directory exists
if [ ! -d "$ES_DIR" ]; then
    echo "Error: Elasticsearch directory not found at $ES_DIR"
    echo "Please download Elasticsearch first. See README for instructions."
    exit 1
fi

# Create necessary directories
mkdir -p "$LOCAL_BASE_DIR"
mkdir -p "$LOGS_DIR"

echo "HPC Mode: Using local SSD for data storage"
echo "Local data directory: $LOCAL_DATA_DIR"

# Download archive from HuggingFace
echo "Downloading archive from HuggingFace..."
echo "Repository: $HF_REPO"
echo "File: $HF_FILE"

# Use huggingface-cli to download file to local temp directory
cd "$LOCAL_BASE_DIR"
huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir . --repo-type dataset

if [ $? -eq 0 ]; then
    # File is in current directory after download
    if [ -f "$HF_FILE" ]; then
        mv "$HF_FILE" "$LOCAL_ARCHIVE"
        echo "Archive downloaded successfully."
        echo "Archive size: $(ls -lh $LOCAL_ARCHIVE | awk '{print $5}')"
    else
        echo "Error: Downloaded file not found at expected location."
        exit 1
    fi
else
    echo "Failed to download archive from HuggingFace."
    echo "Please check:"
    echo "1. You are logged in: huggingface-cli login"
    echo "2. You have access to the repository: $HF_REPO"
    exit 1
fi

# Extract to local SSD directory
echo "Extracting archive to local SSD..."
tar -xzf "$LOCAL_ARCHIVE"

if [ $? -eq 0 ]; then
    echo "Index data extracted successfully to local SSD."
    echo "Local data directory size:"
    du -sh "$LOCAL_DATA_DIR"
    
    # Clean up archive
    rm -f "$LOCAL_ARCHIVE"
    echo "Archive cleaned up."
else
    echo "Failed to extract archive. Exiting."
    exit 1
fi

# Start Elasticsearch with local SSD storage
cd "$ES_DIR"
echo "Starting Elasticsearch with HPC optimized storage:"
echo "Data (local SSD): $LOCAL_DATA_DIR"
echo "Logs (network): $LOGS_DIR"

nohup bin/elasticsearch \
    -E path.data="$LOCAL_DATA_DIR" \
    -E path.logs="$LOGS_DIR" > "$LOGS_DIR/startup.log" 2>&1 &

ES_PID=$!
echo "Elasticsearch started with PID: $ES_PID"

# Wait for ES to start (HPC environment may need more time)
echo "Waiting for Elasticsearch to start..."
for i in {1..240}; do 
    if curl -s localhost:9200 >/dev/null 2>&1; then
        echo "ES up and running!"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "Still waiting... ($i/240 x 5 seconds)"
    fi
    sleep 5
done

# Check if ES started successfully
if ! curl -s localhost:9200 >/dev/null 2>&1; then 
    echo "ES failed to start after 20 minutes"
    echo "=== Startup log ==="
    cat "$LOGS_DIR/startup.log"
    exit 1
fi

echo "ES is running successfully with HPC local storage"

sleep 3
# Check ES status and indices
echo "=== Elasticsearch Status ==="
curl -X GET "localhost:9200/" 2>/dev/null

sleep 3
# Fix replica settings for wiki index (single node doesn't need replicas)
echo -e "\n=== Fixing replica settings for single node ==="
curl -X PUT "localhost:9200/wiki/_settings" -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 0}}' 2>/dev/null
echo ""

# Wait for shards to be reassigned
echo "Waiting for shards to be assigned..."
sleep 5

echo -e "\n=== Cluster Health ==="
curl -X GET "localhost:9200/_cluster/health?pretty" 2>/dev/null

sleep 3
echo -e "\n=== Indices Status ==="
curl -s 'localhost:9200/_cat/indices?v' 2>/dev/null

sleep 3
echo -e "\n=== Wiki Index Count ==="
curl -s 'localhost:9200/wiki/_count' 2>/dev/null

echo ""
echo "=========================================="
echo "NOTE: Data is stored on local SSD at $LOCAL_DATA_DIR"
echo "This data will be lost when the job ends."
echo "=========================================="
