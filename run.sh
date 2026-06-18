#!/bin/bash
# run.sh - Скрипт для быстрого запуска приложения

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}License Plate Detection System${NC}"
echo -e "${GREEN}=================================${NC}"

# Проверка аргументов
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Использование:${NC}"
    echo "  ./run.sh video <input_video> [output_video] - Обработка видео"
    echo "  ./run.sh cam [camera_id] - Запуск веб-камеры"
    echo "  ./run.sh docker-video <input_video> - Обработка видео в Docker"
    echo "  ./run.sh docker-cam - Запуск веб-камеры в Docker"
    echo "  ./run.sh train - Запуск обучения"
    echo "  ./run.sh validate - Валидация модели"
    exit 1
fi

MODE=$1

case $MODE in
    video)
        if [ -z "$2" ]; then
            echo -e "${RED}Ошибка: Укажите путь к видео${NC}"
            exit 1
        fi
        INPUT_VIDEO=$2
        OUTPUT_VIDEO=${3:-"output_$(basename $INPUT_VIDEO)"}
        
        echo -e "${GREEN}Обработка видео: $INPUT_VIDEO${NC}"
        poetry run python -m src.traffic_detector.cli \
            --mode video \
            --input "$INPUT_VIDEO" \
            --output "$OUTPUT_VIDEO" \
            --model data/weights/best.pt \
            --confidence 0.5
        ;;
    
    cam)
        CAMERA_ID=${2:-0}
        echo -e "${GREEN}Запуск веб-камеры (ID: $CAMERA_ID)${NC}"
        poetry run python -m src.traffic_detector.cli \
            --mode cam \
            --camera $CAMERA_ID \
            --model data/weights/best.pt \
            --confidence 0.5
        ;;
    
    docker-video)
        if [ -z "$2" ]; then
            echo -e "${RED}Ошибка: Укажите путь к видео${NC}"
            exit 1
        fi
        INPUT_VIDEO=$2
        VIDEO_DIR=$(dirname "$INPUT_VIDEO")
        VIDEO_NAME=$(basename "$INPUT_VIDEO")
        
        echo -e "${GREEN}Запуск Docker для обработки видео: $INPUT_VIDEO${NC}"
        docker-compose run --rm video-processor \
            python -m src.traffic_detector.cli \
            --mode video \
            --input "/app/videos/$VIDEO_NAME" \
            --output "/app/output/result_$VIDEO_NAME" \
            --model /app/data/weights/best.pt \
            --confidence 0.5
        ;;
    
    docker-cam)
        echo -e "${GREEN}Запуск Docker для веб-камеры${NC}"
        echo -e "${YELLOW}Внимание: Для работы GUI可能需要 дополнительная настройка X11${NC}"
        docker-compose run --rm webcam-processor
        ;;
    
    train)
        echo -e "${GREEN}Запуск обучения модели${NC}"
        poetry run python train.py --mode train
        ;;
    
    validate)
        echo -e "${GREEN}Валидация модели${NC}"
        poetry run python train.py --mode validate
        ;;
    
    *)
        echo -e "${RED}Неизвестный режим: $MODE${NC}"
        exit 1
        ;;
esac