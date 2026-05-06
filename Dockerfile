# ==========================================
# DetSEG — Docker Image for Hugging Face Spaces
# ==========================================
FROM python:3.10-slim

# Install system dependencies (OpenCV, ffmpeg for video)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer-cached for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY website/ ./website/
COPY models/ ./models/
COPY configs/ ./configs/
COPY src/ ./src/

# Hugging Face Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Run with gunicorn for production stability
# --timeout 300 allows long video processing requests
# --workers 1 since models are loaded into memory per worker
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "website.server:app"]
