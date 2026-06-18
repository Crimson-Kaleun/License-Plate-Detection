FROM ultralytics/ultralytics:latest

WORKDIR /app

# Cистемные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Файлы зависимостей
COPY pyproject.toml poetry.lock* /app/

# Poetry и зависимости
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Исходный код
COPY src/ /app/src/
COPY data/weights/ /app/data/weights/
COPY README.md /app/

# Папка для логов
RUN mkdir -p /app/data/logs

# Переменные окружения
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/data/weights/best.pt

# Команда по умолчанию (будет переопределена в docker-compose)
CMD ["python", "-m", "src.traffic_detector.cli", "--help"]