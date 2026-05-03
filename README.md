# Traffic Detector - Система обнаружения номерных знаков

**Разработчик:** Бондарев Алексей

## Описание проекта

Система автоматического обнаружения номерных знаков, автомобилей и пешеходов на основе архитектуры YOLOv8. Разработана для использования в системах интеллектуального мониторинга транспорта.

### Возможности
- 🔍 Детекция номерных знаков, автомобилей и пешеходов
- 🎥 Обработка видеофайлов
- 📷 Работа с веб-камерой в реальном времени
- 🐳 Поддержка Docker контейнеризации
- 📊 Логирование всех операций

### Использование
- 📷 Работа с одним фото: Добавить в корневой папке файл и выполнить команду poetry run python test_model.py название_файла
- 📷 Работа с группой фото: Положить фото  по пути data/tests и выполнить poetry run python test_model.py --all
- 🎥 Работа с видео: poetry run python -m src.traffic_detector.cli --mode video --input road.mp4 --output result.mp4. Входное (road.mp4) и выходное (result.mp4) название можно заменить на свои. 
- 🎥 Работа с веб-камерой в реальном времени: poetry run python -m src.traffic_detector.cli --mode cam --camera 0
- С указанием модели: poetry run python -m src.traffic_detector.cli --mode video --input test.mp4 --model data/weights/best_improved.pt --confidence 0.3

### Как тренировать свою модель:
- 🎥 poetry run python split_data.py  
- 📷 Соберите фотографии сделайте разметку через LabelImg (или аналогичной программой) в формате YOLO. 0 - plate, 1 - person, 2 - car.
- 🐳 В корне проекта создадите папку Work_Nums и положите туда фото и разметку
- 📊 Введите команду poetry run python split_data.py. Это разделит данные на тестируемые и валидационные и распределит в понятные для YOLO папки
- 🔍 В командной строке по адресу проекта: poetry run python train.py --model train

### Дополнительно:
- Проверка модели на наличие CUDA - poetry run python check_gpu.py
- Эмпирический анализ датасета (/data/dataset) - analyze_dataset.py

### Software Engineer Part

| Баллы | Критерий | Статус | Комментарий|
|-------|----------|--------|-------------|
| 15 | Свой датасет | ✅ | 109 изображений, размеченных с помощью LabelImg. Включают в себя фото, сделанные самостоятельно на дорогах, скриншоты с домовых камер и изображения с открытых датасетов|
| 15 | Use cases (video + stream) | ✅ | `--mode video` и `--mode cam` |
| 5 | README demos (GIF) | ✅ | Демонстрация работы (ссылка на видео/GIF) |
| 5 | Code quality | ✅ | ООП, docstring |
| 5 | model.py | ✅ | `My_LicensePlate_Model` в `src/traffic_detector/model_impl.py` с методом `detect_plates()` |
| 2 | Logging | ✅ | Логгер, `./data/log_file.log`, обработка ошибок |
| 3 | Git workflow | ✅ | Публичный репозиторий, ветки `dev`/`main`|


## Требования

### Минимальные системные требования
- Python 3.10+
- NVIDIA GPU с драйверами CUDA (рекомендуется)
- 8GB RAM
- 5GB свободного места на диске

### Установка зависимостей
```bash
# Установка Poetry (если не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей проекта
poetry install