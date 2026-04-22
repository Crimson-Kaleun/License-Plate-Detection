#!/usr/bin/env python3
"""
train.py - Скрипт для обучения модели YOLO на кастомном датасете
"""
import os
from pathlib import Path
import yaml
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_data_yaml():
    """Создает файл data.yaml если его нет"""
    data_config = {
        'path': './data/dataset',  # Путь к датасету
        'train': 'train/images',   # Относительный путь к тренировочным картинкам
        'val': 'val/images',       # Относительный путь к валидационным картинкам
        'nc': 3,                   # Количество классов (plate, person, car)
        'names': ['plate', 'person', 'car']  # Имена классов
    }
    
    yaml_path = Path('data/dataset/data.yaml')
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"✅ Создан файл конфигурации: {yaml_path}")
    return yaml_path

def check_dataset():
    """Проверяет, что датасет готов к обучению"""
    train_images = Path('data/dataset/train/images')
    train_labels = Path('data/dataset/train/labels')
    val_images = Path('data/dataset/val/images')
    val_labels = Path('data/dataset/val/labels')
    
    issues = []
    
    # Проверяем наличие папок
    for path in [train_images, train_labels, val_images, val_labels]:
        if not path.exists():
            issues.append(f"❌ Папка не найдена: {path}")
    
    # Проверяем количество файлов
    num_train_images = len(list(train_images.glob('*.*'))) if train_images.exists() else 0
    num_train_labels = len(list(train_labels.glob('*.txt'))) if train_labels.exists() else 0
    num_val_images = len(list(val_images.glob('*.*'))) if val_images.exists() else 0
    num_val_labels = len(list(val_labels.glob('*.txt'))) if val_labels.exists() else 0
    
    logger.info(f"📊 Статистика датасета:")
    logger.info(f"   Train images: {num_train_images}")
    logger.info(f"   Train labels: {num_train_labels}")
    logger.info(f"   Val images: {num_val_images}")
    logger.info(f"   Val labels: {num_val_labels}")
    
    if num_train_images == 0:
        issues.append("❌ Нет тренировочных изображений!")
    if num_train_labels == 0:
        issues.append("❌ Нет разметки для тренировочных изображений!")
    if num_val_images == 0:
        issues.append("❌ Нет валидационных изображений!")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return False
    
    logger.info("✅ Датасет готов к обучению!")
    return True

def train_model():
    """
    Основная функция обучения модели
    """
    logger.info("🚀 Начинаем обучение модели YOLO...")
    
    # Проверяем наличие Ultralytics
    try:
        from ultralytics import YOLO
        logger.info("✅ Ultralytics YOLO загружен")
    except ImportError:
        logger.error("❌ Ultralytics не установлен! Установи: poetry add ultralytics")
        return
    
    # Проверяем CUDA
    import torch
    if torch.cuda.is_available():
        device = 0  # Используем GPU
        logger.info(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        logger.warning("⚠️ CUDA не найдена! Обучение будет на CPU (очень медленно)")
    
    # Создаем data.yaml если нужно
    data_yaml = create_data_yaml()
    
    # Проверяем датасет
    if not check_dataset():
        logger.error("❌ Проблемы с датасетом. Исправьте ошибки и запустите снова.")
        return
    
    # Загружаем предобученную модель
    logger.info("📥 Загрузка предобученной модели YOLOv8n...")
    model = YOLO('yolov8n.pt')  # nano версия (самая быстрая)
    # Для лучшей точности можно использовать 'yolov8s.pt' или 'yolov8m.pt'
    
    
    # Параметры обучения
    training_params = {
        'data': str(data_yaml),      # Путь к data.yaml
        'epochs': 60,                # Количество эпох (можно начать с 50)
        'imgsz': 640,                # Размер входного изображения
        'batch': 8,                 # Размер батча (уменьши если не хватает памяти)
        'device': device,            # 'cpu' или 0 для GPU. Или переменная device
        'workers': 1,                # Количество потоков для загрузки данных
        'patience': 15,              # Ранняя остановка если нет улучшений
        'save': True,                # Сохранять чекпоинты
        'save_period': 10,           # Сохранять каждые 10 эпох
        'pretrained': True,          # Использовать предобученные веса
        'optimizer': 'auto',         # Автоматический выбор оптимизатора
        'lr0': 0.01,                 # Начальная скорость обучения
        'lrf': 0.01,                 # Финальная скорость обучения
        'momentum': 0.937,           # Моментум для SGD
        'weight_decay': 0.0005,      # Регуляризация
        'warmup_epochs': 3,          # Разогрев обучения
        'warmup_momentum': 0.8,      # Начальный моментум
        'warmup_bias_lr': 0.1,       # Начальная скорость для bias
        'box': 7.5,                  # Вес loss для box
        'cls': 0.5,                  # Вес loss для класса
        'dfl': 1.5,                  # Вес loss для dfl
        'hsv_h': 0.015,              # Аугментация: Hue
        'hsv_s': 0.7,                # Аугментация: Saturation
        'hsv_v': 0.4,                # Аугментация: Value
        'degrees': 0.0,              # Аугментация: поворот
        'translate': 0.1,            # Аугментация: сдвиг
        'scale': 0.5,                # Аугментация: масштаб
        'shear': 0.0,                # Аугментация: наклон
        'perspective': 0.0,          # Аугментация: перспектива
        'flipud': 0.0,               # Аугментация: вертикальное отражение
        'fliplr': 0.5,               # Аугментация: горизонтальное отражение
        'mosaic': 1.0,               # Аугментация: мозаика
        'mixup': 0.0,                # Аугментация: смешивание
        'copy_paste': 0.0,           # Аугментация: копирование
    }

    # Параметры обучения
    training_params0 = {
        'data': str(data_yaml),      # Путь к data.yaml
        'epochs': 30,                # Количество эпох (можно начать с 50)
        'imgsz': 640,                # Размер входного изображения
        'batch': 16,                 # Размер батча (уменьши если не хватает памяти)
        'device': device,            # 'cpu' или 0 для GPU. Или переменная device
        'workers': 8,                # Количество потоков для загрузки данных
        'patience': 15,              # Ранняя остановка если нет улучшений
        'save': True,                # Сохранять чекпоинты
        'save_period': 10,           # Сохранять каждые 10 эпох
        'pretrained': True,          # Использовать предобученные веса
        'optimizer': 'auto'         # Автоматический выбор оптимизатора
    }
    
    logger.info("🏋️ Начинаем обучение...")
    logger.info(f"   Эпох: {training_params['epochs']}")
    logger.info(f"   Размер батча: {training_params['batch']}")
    logger.info(f"   Размер изображений: {training_params['imgsz']}")
    
    try:
        # Запускаем обучение
        results = model.train(**training_params)
        
        logger.info("✅ Обучение завершено!")
        
        results_dir = results.save_dir
        # Путь к лучшей модели
        best_model_path = Path(f"{results_dir}/weights/best.pt")
        if best_model_path.exists():
            logger.info(f"🏆 Лучшая модель сохранена: {best_model_path}")
            
            # Копируем лучшую модель в data/weights/
            weights_dir = Path('data/weights')
            weights_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = weights_dir / 'best.pt'
            
            import shutil
            shutil.copy(best_model_path, final_model_path)
            logger.info(f"📦 Модель скопирована в: {final_model_path}")
        else:
            logger.warning("⚠️ Файл с лучшей моделью не найден!")
        
        # Выводим метрики
        if hasattr(results, 'results') and results.results:
            logger.info("\n📊 Итоговые метрики:")
            if 'metrics/mAP50(B)' in results.results:
                mAP50 = results.results['metrics/mAP50(B)']
                logger.info(f"   mAP@0.5: {mAP50:.4f}")
            if 'metrics/mAP50-95(B)' in results.results:
                mAP5095 = results.results['metrics/mAP50-95(B)']
                logger.info(f"   mAP@0.5:0.95: {mAP5095:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время обучения: {e}")
        raise

def validate_model(model_path: str = None):
    """
    Валидация обученной модели на тестовых данных
    """
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = Path('data/weights/best.pt')
        if not model_path.exists():
            logger.error(f"Модель не найдена: {model_path}")
            return
    
    logger.info(f"🔍 Валидация модели: {model_path}")
    
    # Загружаем модель
    model = YOLO(str(model_path))
    
    # Запускаем валидацию
    results = model.val(
        data='data/dataset/data.yaml',
        device=0,  # Используем GPU
        conf=0.5,  # Порог уверенности
        iou=0.45,  # Порог IoU
        batch=16
    )
    
    logger.info("📊 Результаты валидации:")
    logger.info(f"   mAP50: {results.box.map50:.4f}")
    logger.info(f"   mAP50-95: {results.box.map:.4f}")
    logger.info(f"   Precision: {results.box.mp:.4f}")
    logger.info(f"   Recall: {results.box.mr:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение детектора номеров')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'validate'],
                       help='Режим: train (обучение) или validate (валидация)')
    parser.add_argument('--model', type=str, default=None,
                       help='Путь к модели для валидации')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    else:
        validate_model(args.model)