#!/bin/bash
# Simple script to start Elasticsearch with existing data
# Use this script after you have already built or downloaded the index

echo "=========================================="
echo "QuCo-RAG Elasticsearch Startup Script"
echo "=========================================="

# Set path variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_DIR="$SCRIPT_DIR/data/elasticsearch-7.17.9"
DATA_DIR="$ES_DIR/data"
LOGS_DIR="$ES_DIR/logs"

echo ""
echo "Configuration:"
echo "  Elasticsearch directory: $ES_DIR"
echo "  Data directory: $DATA_DIR"
echo "  Logs directory: $LOGS_DIR"
echo ""

# Check if Elasticsearch directory exists
if [ ! -d "$ES_DIR" ]; then
    echo "❌ Error: Elasticsearch directory not found at $ES_DIR"
    echo ""
    echo "Please download Elasticsearch first:"
    echo "  cd data"
    echo "  wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz"
    echo "  tar zxvf elasticsearch-7.17.9.tar.gz"
    echo "  cd .."
    exit 1
fi

# Check if index data exists
if [ ! -d "$DATA_DIR/nodes" ]; then
    echo "❌ Error: Index data not found at $DATA_DIR"
    echo ""
    echo "Please build or download the index first. See README for instructions:"
    echo "  Option 1: Build from scratch with prep_elastic_index_with_progress.py"
    echo "  Option 2: Download pre-built index with Start_Elasticsearch_from_hf.sh"
    exit 1
fi

# Check if Elasticsearch is already running
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

# Ask user to confirm before starting
echo ""
echo "Ready to start Elasticsearch."
echo "  Host: localhost"
echo "  Port: 9200"
echo "  Data: $DATA_DIR"
echo ""
read -p "Continue? [Y/n] " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create logs directory if not exists
mkdir -p "$LOGS_DIR"

# Start Elasticsearch
cd "$ES_DIR"
echo ""
echo "Starting Elasticsearch..."

nohup bin/elasticsearch \
    -E path.data="$DATA_DIR" \
    -E path.logs="$LOGS_DIR" > "$LOGS_DIR/startup.log" 2>&1 &

ES_PID=$!
echo "Elasticsearch started with PID: $ES_PID"

# Wait for ES to start
echo "Waiting for Elasticsearch to start..."
for i in {1..60}; do 
    if curl -s localhost:9200 >/dev/null 2>&1; then
        echo "✓ Elasticsearch is up and running!"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still waiting... ($i/60 x 2 seconds)"
    fi
    sleep 2
done

# Check if ES started successfully
if ! curl -s localhost:9200 >/dev/null 2>&1; then 
    echo "❌ Elasticsearch failed to start after 2 minutes"
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
echo "✓ Elasticsearch is ready!"
echo "  URL: http://localhost:9200"
echo "  To stop: pkill -f elasticsearch"
echo "=========================================="
