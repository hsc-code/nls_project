#!/bin/bash
# Download models for offline use
# Run this script once to download all required models

set -e

echo "========================================"
echo "  Voice Assistant - Model Downloader"
echo "========================================"
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create directories
WHISPER_DATA_DIR="$PROJECT_ROOT/faster-whisper-data"
PIPER_DATA_DIR="$PROJECT_ROOT/piper-data"
OLLAMA_DATA_DIR="$PROJECT_ROOT/ollama-data"

echo "Creating data directories..."
mkdir -p "$WHISPER_DATA_DIR"
mkdir -p "$PIPER_DATA_DIR"
mkdir -p "$OLLAMA_DATA_DIR"

echo ""
echo "Step 1: Starting containers to download models..."
echo "This may take a few minutes on first run."
echo ""

cd "$PROJECT_ROOT"
docker compose up -d

echo ""
echo "Step 2: Waiting for Whisper model to download..."

# Wait for faster-whisper to be ready
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker logs faster-whisper 2>&1 | grep -q "Ready"; then
        echo "  Whisper model ready!"
        break
    fi
    echo "  Waiting... (${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  Warning: Timeout waiting for Whisper. Check logs with: docker logs faster-whisper"
fi

echo ""
echo "Step 3: Waiting for Piper model to download..."

# Wait for piper to be ready
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker logs piper 2>&1 | grep -q "Ready"; then
        echo "  Piper model ready!"
        break
    fi
    echo "  Waiting... (${WAITED}s)"
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  Warning: Timeout waiting for Piper. Check logs with: docker logs piper"
fi

echo ""
echo "Step 4: Pulling Ollama LLM model (qwen2.5:0.5b)..."
echo "This may take a few minutes depending on your connection."

# Wait for Ollama to be ready
WAITED=0
MAX_WAIT=60
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Ollama server ready!"
        break
    fi
    echo "  Waiting for Ollama server... (${WAITED}s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

# Pull the model
echo "  Pulling model..."
docker exec ollama ollama pull qwen2.5:1.5b

if [ $? -eq 0 ]; then
    echo "  LLM model ready!"
else
    echo "  Warning: Failed to pull model. You can manually run: docker exec ollama ollama pull qwen2.5:1.5b"
fi

echo ""
echo "========================================"
echo "  Download Complete!"
echo "========================================"
echo ""
echo "Models are cached in:"
echo "  - $WHISPER_DATA_DIR (Whisper STT)"
echo "  - $PIPER_DATA_DIR (Piper TTS)"
echo "  - $OLLAMA_DATA_DIR (Ollama LLM)"
echo ""
echo "The app will now work offline."
echo "Access it at: http://localhost:8000"
echo ""
