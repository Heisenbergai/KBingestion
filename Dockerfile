FROM python:3.12-slim

# FFmpeg for video assembly + system libs for Pillow (image rendering)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render passes PORT as an environment variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
