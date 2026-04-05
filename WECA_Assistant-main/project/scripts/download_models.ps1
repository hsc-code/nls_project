# Download models for offline use
# Run this script once to download all required models

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Voice Assistant - Model Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create directories
$projectRoot = Split-Path -Parent $PSScriptRoot
$whisperDataDir = Join-Path $projectRoot "faster-whisper-data"
$piperDataDir = Join-Path $projectRoot "piper-data"
$ollamaDataDir = Join-Path $projectRoot "ollama-data"

Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $whisperDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $piperDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ollamaDataDir | Out-Null

Write-Host ""
Write-Host "Step 1: Starting containers to download models..." -ForegroundColor Green
Write-Host "This may take a few minutes on first run." -ForegroundColor Gray
Write-Host ""

# Start containers (they will download models on first run)
Set-Location $projectRoot
docker compose up -d

Write-Host ""
Write-Host "Step 2: Waiting for Whisper model to download..." -ForegroundColor Green

# Wait for faster-whisper to be ready
$maxWait = 300  # 5 minutes
$waited = 0
while ($waited -lt $maxWait) {
    $logs = docker logs faster-whisper 2>&1 | Select-String "Ready"
    if ($logs) {
        Write-Host "  Whisper model ready!" -ForegroundColor Green
        break
    }
    Write-Host "  Waiting... ($waited s)" -ForegroundColor Gray
    Start-Sleep -Seconds 10
    $waited += 10
}

if ($waited -ge $maxWait) {
    Write-Host "  Warning: Timeout waiting for Whisper. Check logs with: docker logs faster-whisper" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 3: Waiting for Piper model to download..." -ForegroundColor Green

# Wait for piper to be ready
$waited = 0
while ($waited -lt $maxWait) {
    $logs = docker logs piper 2>&1 | Select-String "Ready"
    if ($logs) {
        Write-Host "  Piper model ready!" -ForegroundColor Green
        break
    }
    Write-Host "  Waiting... ($waited s)" -ForegroundColor Gray
    Start-Sleep -Seconds 10
    $waited += 10
}

if ($waited -ge $maxWait) {
    Write-Host "  Warning: Timeout waiting for Piper. Check logs with: docker logs piper" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 4: Pulling Ollama LLM model (qwen2.5:0.5b)..." -ForegroundColor Green
Write-Host "This may take a few minutes depending on your connection." -ForegroundColor Gray

# Wait for Ollama to be ready first
$waited = 0
$maxWait = 60
while ($waited -lt $maxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "  Ollama server ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Server not ready yet
    }
    Write-Host "  Waiting for Ollama server... ($waited s)" -ForegroundColor Gray
    Start-Sleep -Seconds 5
    $waited += 5
}

# Pull the model
Write-Host "  Pulling model..." -ForegroundColor Yellow
docker exec ollama ollama pull qwen2.5:0.5b

if ($LASTEXITCODE -eq 0) {
    Write-Host "  LLM model ready!" -ForegroundColor Green
} else {
    Write-Host "  Warning: Failed to pull model. You can manually run: docker exec ollama ollama pull qwen2.5:0.5b" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Download Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models are cached in:" -ForegroundColor White
Write-Host "  - $whisperDataDir (Whisper STT)" -ForegroundColor Gray
Write-Host "  - $piperDataDir (Piper TTS)" -ForegroundColor Gray
Write-Host "  - $ollamaDataDir (Ollama LLM)" -ForegroundColor Gray
Write-Host ""
Write-Host "The app will now work offline." -ForegroundColor Green
Write-Host "Access it at: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""
