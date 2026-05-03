# train_improved.py
from ultralytics import YOLO
import torch
from pathlib import Path

def train_improved():
    """
    Улучшенное обучение с фокусом на номера
    """
    print("🚀 ЗАПУСК УЛУЧШЕННОГО ОБУЧЕНИЯ")
    print("="*50)
    
    # Проверяем GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Загружаем модель побольше (small вместо nano для лучшей точности)
    # YOLOv8s лучше находит маленькие объекты
    model = YOLO('yolov8s.pt')  # s - small, точнее чем n - nano
    
    # Улучшенные параметры обучения
    results = model.train(
        data='data/dataset/data.yaml',
        epochs=150,                 # Больше эпох
        imgsz=640,                 # Размер изображения
        batch=16,                  # Размер батча (уменьши если не хватает памяти)
        device=device,
        workers=8,
        
        # Оптимизация для маленьких объектов
        patience=20,               # Терпение при отсутствии улучшений
        
        # Аугментации для номеров
        mosaic=1.0,               # Обязательно для маленьких объектов
        scale=0.5,                # Масштабирование
        fliplr=0.5,              # Отражение по горизонтали
        copy_paste=0.3,          # Копирование объектов (полезно для номеров)
        
        # Улучшенные параметры оптимизатора
        lr0=0.01,                 # Начальная скорость обучения
        lrf=0.01,                # Конечная скорость обучения
        momentum=0.937,          # Моментум
        weight_decay=0.0005,     # Регуляризация
        
        # Веса потерь (увеличиваем важность номеров)
        box=7.5,                 # Потеря позиционирования
        cls=0.5,                # Потеря классификации
        dfl=1.5,                # Потеря распределения
        
        # Сохранение
        save=True,
        save_period=10,
        project='runs/train',
        name='improved_training',
        exist_ok=True,
        
        # Визуализация
        plots=True,
    )
    
    print("\n✅ Обучение завершено!")
    print("📁 Результаты в: runs/train/improved_training/")
    
    # Копируем лучшую модель
    best_model = Path('runs/train/improved_training/weights/best.pt')
    if best_model.exists():
        Path('data/weights').mkdir(exist_ok=True)
        import shutil
        shutil.copy(best_model, 'data/weights/best_improved.pt')
        print("📦 Модель сохранена в: data/weights/best_improved.pt")
    
    return results

if __name__ == "__main__":
    train_improved()