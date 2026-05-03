# analyze_dataset.py
import os
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

def analyze_dataset():
    """Анализирует датасет и выдает рекомендации"""
    
    # Проверяем количество примеров
    train_images = list(Path('data/dataset/train/images').glob('*.*'))
    train_labels = list(Path('data/dataset/train/labels').glob('*.txt'))
    
    print("📊 АНАЛИЗ ДАТАСЕТА")
    print("="*50)
    print(f"Тренировочных изображений: {len(train_images)}")
    print(f"Файлов разметки: {len(train_labels)}")
    
    # Считаем объекты
    class_counts = {0: 0, 1: 0, 2: 0}  # plate, person, car
    small_objects = 0
    total_objects = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
                    # Проверяем размер объекта
                    width = float(parts[3])
                    height = float(parts[4])
                    if width * height < 0.01:  # Меньше 1% от изображения
                        small_objects += 1
                    total_objects += 1
    
    print(f"\n📦 Распределение объектов:")
    print(f"   Номера (plate): {class_counts.get(0, 0)}")
    print(f"   Люди (person): {class_counts.get(1, 0)}")
    print(f"   Машины (car): {class_counts.get(2, 0)}")
    print(f"\n📏 Маленькие объекты (<1% площади): {small_objects}/{total_objects}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if class_counts.get(0, 0) < 500:
        print("   ⚠️ Мало примеров номеров! Нужно добавить еще 200-300 размеченных номеров")
    if small_objects / total_objects > 0.3:
        print("   ⚠️ Много маленьких объектов. Увеличь разрешение или используй mosaic аугментацию")
    
    return class_counts

if __name__ == "__main__":
    analyze_dataset()