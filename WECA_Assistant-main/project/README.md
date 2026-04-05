# Speech-to-Text API

A FastAPI application that provides speech-to-text transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) via the [Wyoming protocol](https://github.com/rhasspy/wyoming).

## Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Client    │──HTTP──▶│    STT API       │──TCP───▶│ faster-whisper  │
│             │◀────────│  (FastAPI)       │◀────────│ (Wyoming)       │
└─────────────┘         └──────────────────┘         └─────────────────┘
     Audio file           Port 8000                    Port 10300
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

This starts:
- **faster-whisper**: Wyoming protocol server for speech recognition (port 10300)
- **stt-api**: FastAPI application for HTTP access (port 8000)

### Manual Development Setup

1. Start the faster-whisper container:
   ```bash
   docker-compose up -d faster-whisper
   ```

2. Install Python dependencies:
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   # From the app directory
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Usage

### Transcribe Audio

**Endpoint:** `POST /api/v1/transcribe`

**Parameters:**
- `audio` (required): Audio file to transcribe
- `language` (optional): Language code (default: "en")

**Supported Formats:** WAV, MP3, OGG, FLAC, WebM, MP4, M4A

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -F "audio=@recording.wav" \
  -F "language=en"
```

**Example using Python:**

```python
import requests

with open("recording.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/transcribe",
        files={"audio": f},
        data={"language": "en"}
    )
    print(response.json())
```

**Response:**

```json
{
  "text": "Hello, how are you today?",
  "language": "en",
  "success": true,
  "error": null
}
```

### Health Check

**Endpoint:** `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "whisper_server": true
}
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

Environment variables for the STT API:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_WHISPER_HOST` | `faster-whisper` | Hostname of the Wyoming server |
| `STT_WHISPER_PORT` | `10300` | Port of the Wyoming server |
| `STT_DEFAULT_LANGUAGE` | `en` | Default language for transcription |
| `STT_MAX_AUDIO_SIZE_MB` | `25` | Maximum audio file size in MB |
| `STT_AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate for conversion |

## Whisper Models

The faster-whisper container supports various model sizes. Configure via the `WHISPER_MODEL` environment variable:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny-int8` | ~40MB | Fastest | Lower |
| `base-int8` | ~75MB | Fast | Good |
| `small-int8` | ~250MB | Medium | Better |
| `medium-int8` | ~750MB | Slower | High |
| `large-v3` | ~3GB | Slowest | Highest |

## Development

### Project Structure

```
project/
├── docker-compose.yml      # Docker services configuration
├── README.md
├── app/
│   ├── Dockerfile          # FastAPI container
│   ├── requirements.txt    # Python dependencies
│   ├── __init__.py
│   ├── main.py             # FastAPI application entry
│   ├── config.py           # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py       # API endpoints
│   │   └── schemas.py      # Pydantic models
│   └── services/
│       ├── __init__.py
│       └── transcription_service.py  # Wyoming client
└── faster-whisper-data/    # Model cache (gitignore this)
```

### Running Tests

```bash
cd app
pytest
```

## Troubleshooting

### "Cannot connect to Wyoming server"

1. Ensure faster-whisper container is running:
   ```bash
   docker-compose ps
   ```

2. Check if the model is downloaded (first run takes time):
   ```bash
   docker-compose logs faster-whisper
   ```

3. Verify the port is accessible:
   ```bash
   nc -zv localhost 10300
   ```

### Audio conversion errors

Ensure `ffmpeg` is installed (included in Docker image, required for local dev):
```bash
# Ubuntu/Debian
apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```
