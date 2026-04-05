# Get static ffmpeg binary (no apt-get needed)
FROM mwader/static-ffmpeg:7.1 AS ffmpeg

FROM python:3.11-slim

WORKDIR /code

# Copy static ffmpeg and ffprobe binaries from builder stage
COPY --from=ffmpeg /ffmpeg /usr/local/bin/
COPY --from=ffmpeg /ffprobe /usr/local/bin/

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code as 'app' package
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run with hot reload enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/code/app"]