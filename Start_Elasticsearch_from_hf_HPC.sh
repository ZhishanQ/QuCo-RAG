#!/bin/bash
# HPC version: Download pre-built index and use local SSD (/tmp) for better I/O performance
#
# ⚠️  WARNING: This script stores index data in /tmp which is LOCAL to the compute node.
#     The data will be DELETED when the job ends or the node is released.
#     You need to re-download the index for each new job.
#
# For persistent storage, use Start_Elasticsearch_from_hf.sh instead.

echo "=========================================="
echo "QuCo-RAG: HPC Elasticsearch Setup"
echo "=========================================="
echo ""
echo "⚠️  WARNING: HPC LOCAL STORAGE MODE"
echo "   Index data will be stored in /tmp (local SSD)"
echo "   Data will be LOST when job ends!"
echo "=========================================="
echo ""

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

echo "Configuration:"
echo "  Elasticsearch directory: $ES_DIR"
echo "  Local data directory (SSD): $LOCAL_DATA_DIR"
echo "  Logs directory: $LOGS_DIR"
echo "  Elasticsearch URL: http://localhost:9200"
echo ""

# Ask user to confirm configuration
echo "⚠️  The index data will be stored at: $LOCAL_DATA_DIR"
echo "⚠️  This data will be DELETED when the HPC job ends."
echo ""
read -p "Continue with HPC local storage mode? [Y/n] " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Aborted."
    echo "Consider using Start_Elasticsearch_from_hf.sh for persistent storage."
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
mkdir -p "$LOCAL_BASE_DIR"
mkdir -p "$LOGS_DIR"

echo ""
echo "HPC Mode: Using local SSD for index data"
echo "Local data directory: $LOCAL_DATA_DIR"

# Check if index data already exists in local storage
if [ -d "$LOCAL_DATA_DIR/nodes" ]; then
    echo ""
    echo "=========================================="
    echo "✓ Index data already exists at $LOCAL_DATA_DIR"
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
    echo "Destination: $LOCAL_BASE_DIR (local SSD)"
    echo ""

    cd "$LOCAL_BASE_DIR"
    
    # Check if archive already exists
    if [ -f "$LOCAL_ARCHIVE" ]; then
        echo "Archive file already exists: $LOCAL_ARCHIVE"
        echo "Archive size: $(ls -lh $LOCAL_ARCHIVE | awk '{print $5}')"
        read -p "Use existing archive? [Y/n] " use_existing
        use_existing=${use_existing:-Y}
        
        if [[ ! "$use_existing" =~ ^[Yy]$ ]]; then
            echo "Removing existing archive and re-downloading..."
            rm -f "$LOCAL_ARCHIVE"
        fi
    fi

    # Download if archive doesn't exist
    if [ ! -f "$LOCAL_ARCHIVE" ]; then
        huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir . --repo-type dataset

        if [ $? -eq 0 ] && [ -f "$HF_FILE" ]; then
            mv "$HF_FILE" "$LOCAL_ARCHIVE"
            echo "✓ Archive downloaded successfully."
            echo "Archive size: $(ls -lh $LOCAL_ARCHIVE | awk '{print $5}')"
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

    # Extract to local SSD directory
    echo ""
    echo "Extracting archive to local SSD..."
    echo "This may take a few minutes..."
    tar -xzf "$LOCAL_ARCHIVE"

    if [ $? -eq 0 ]; then
        echo "✓ Index data extracted successfully to local SSD."
        echo "Local data directory size: $(du -sh "$LOCAL_DATA_DIR" | cut -f1)"
        echo ""
        echo "NOTE: Archive file kept at $LOCAL_ARCHIVE"
        echo "      It will be deleted when the job ends anyway."
    else
        echo "❌ Failed to extract archive. Exiting."
        exit 1
    fi
fi

# Start Elasticsearch with local SSD storage
echo ""
echo "=========================================="
echo "Starting Elasticsearch with HPC optimized storage..."
echo "=========================================="
echo "Data (local SSD): $LOCAL_DATA_DIR"
echo "Logs (network): $LOGS_DIR"

cd "$ES_DIR"
nohup bin/elasticsearch \
    -E path.data="$LOCAL_DATA_DIR" \
    -E path.logs="$LOGS_DIR" > "$LOGS_DIR/startup.log" 2>&1 &

ES_PID=$!
echo "Elasticsearch started with PID: $ES_PID"

# Wait for ES to start (HPC environment may need more time)
echo ""
echo "Waiting for Elasticsearch to start..."
for i in {1..240}; do 
    if curl -s localhost:9200 >/dev/null 2>&1; then
        echo "✓ Elasticsearch is up and running!"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "Still waiting... ($i/240 x 5 seconds)"
    fi
    sleep 5
done

# Check if ES started successfully
if ! curl -s localhost:9200 >/dev/null 2>&1; then 
    echo ""
    echo "❌ Elasticsearch failed to start after 20 minutes"
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
echo "✓ HPC Setup Complete!"
echo "=========================================="
echo ""
echo "Elasticsearch is now running at: http://localhost:9200"
echo "Wiki index contains ~21M documents."
echo ""
echo "⚠️  IMPORTANT (HPC MODE):"
echo "   - Index data is stored at: $LOCAL_DATA_DIR"
echo "   - This data is on LOCAL SSD and will be DELETED when the job ends."
echo "   - For the next HPC job, you need to run this script again."
echo ""
echo "   - To stop Elasticsearch: pkill -f elasticsearch"
echo "=========================================="
