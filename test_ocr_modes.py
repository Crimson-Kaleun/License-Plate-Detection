#!/usr/bin/env python3
"""
test_ocr_modes.py - Сравнение распознавания на разных языках
"""
import cv2
import numpy as np
from src.traffic_detector.plate_reader import PlateReader

def test_language_comparison():
    """
    Сравнивает распознавание на латинице и кириллице
    """
    # Создаем тестовые изображения
    test_plates = {
        'latin': "ABC123",      # Латиница
        'cyrillic': "А123ВС77", # Кириллица
        'mixed': "A123BC78"     # Смешанный
    }
    
    # Создаем изображения с текстом
    images = {}
    for name, text in test_plates.items():
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        images[name] = img
    
    # Тестируем с разными языками
    for lang in ['en', 'ru', 'en,ru']:
        print(f"\n🔤 Язык: {lang}")
        print("-" * 40)
        
        reader = PlateReader(language=lang, use_gpu=False, improve_image=False)
        
        for name, img in images.items():
            result = reader.read_plate(img, filter_plate_format=False)
            print(f"  {name}: '{result['text']}' (conf: {result['confidence']:.2f})")

if __name__ == "__main__":
    test_language_comparison()