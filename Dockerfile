FROM python:3.12-slim

# ffmpeg + fonts are required by the pipeline (video assembly, subtitles, thumbnails).
# drawtext (branding overlay) needs a full FFmpeg build, so verify it at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -filters | grep -q drawtext

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
