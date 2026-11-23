FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fontconfig fonts-noto-core \
    && rm -rf /var/lib/apt/lists/* && fc-cache -f
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m app && mkdir -p /tmp/tarjimli && chown -R app /app /tmp/tarjimli
USER app
ENV PORT=10000 PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
