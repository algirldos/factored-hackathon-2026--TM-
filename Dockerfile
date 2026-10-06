# Customer web chat (web_app.py) for Render or any container host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dependencies first, so code changes do not reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Unprivileged user; DuckDB keeps its MotherDuck extension in the user's home
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=8)"

CMD ["python", "web_app.py"]
