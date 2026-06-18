#!/usr/bin/env python3
"""
test_ocr.py - Тестирование распознавания номеров
"""
import cv2
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from src.traffic_detector.model_impl import My_LicensePlate_Model
from src.traffic_detector.plate_reader import PlateReader

def test_ocr_on_image(image_path: str, model_path: str = "data/weights/best.pt"):
    """
    Тестирует OCR на одном изображении
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Загружаем детектор с OCR
    detector = My_LicensePlate_Model(
        model_path=model_path,
        confidence=0.5,
        device='cuda',  # или 'cpu'
        enable_ocr=True,
        ocr_language='en,ru',  # Поддерживаем оба языка
        use_ocr_cache=True,
        improve_plate_image=True
    )
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Не удалось загрузить: {image_path}")
        return
    
    # Детекция с распознаванием
    detections = detector.detect_plates(image)
    
    # Рисуем результаты
    result_image = detector.draw_boxes(image.copy(), detections)
    
    # Выводим результаты
    print("\n📸 РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ")
    print("=" * 60)
    
    for i, det in enumerate(detections, 1):
        print(f"\nНомер {i}:")
        print(f"  Позиция: {det['bbox']}")
        print(f"  Уверенность детекции: {det['confidence']:.2f}")
        
        if 'plate_text' in det and det['plate_text']:
            print(f"  Текст: {det['plate_text']}")
            print(f"  Уверенность распознавания: {det['plate_confidence']:.2f}")
            
            # Показываем предобработанное изображение
            if det.get('plate_preprocessed') is not None:
                processed = det['plate_preprocessed']
                #cv2.imshow(f"Processed Plate {i}", processed)
                img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.title('Мое изображение')
                plt.show()
        else:
            print("  Текст не распознан")
    
    # Сохраняем и показываем результат
    output_path = f"ocr_result_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\n✅ Результат сохранен: {output_path}")
    
    try:
        cv2.imshow('Detection with OCR', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        img_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Detection with OCR')
        plt.show()

def test_plate_preprocessing():
    """
    Тестирует различные методы предобработки номеров
    """
    from src.traffic_detector.plate_reader import PlateReader
    
    reader = PlateReader(language='en', use_gpu=False)
    
    # Тестовое изображение
    test_image_path = "test_plate.jpg"
    if not Path(test_image_path).exists():
        print("⚠️ Тестовое изображение не найдено")
        return
    
    plate = cv2.imread(test_image_path)
    
    methods = [
        ("Оригинал", False, False, False),
        ("CLAHE", True, False, False),
        ("Bilateral", False, True, False),
        ("CLAHE + Bilateral", True, True, False),
        ("CLAHE + Bilateral + Adaptive", True, True, True),
    ]
    
    for name, use_clahe, use_bilateral, use_adaptive in methods:
        processed = reader.preprocess_plate(
            plate,
            use_clahe=use_clahe,
            use_bilateral=use_bilateral,
            use_adaptive_threshold=use_adaptive
        )
        
                # Показывать изображение (если есть GUI)
        try:
            cv2.imshow(name, processed)
            print(f"Показано: {name}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print(f"Могло быть показано (cv2.imshow не работает): {name}")
            img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(name)
            plt.show()
            continue
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование OCR')
    parser.add_argument('--image', type=str, default='test_car.jpg',
                       help='Путь к тестовому изображению')
    parser.add_argument('--mode', type=str, default='detect',
                       choices=['detect', 'preprocess'],
                       help='Режим тестирования')
    
    args = parser.parse_args()
    
    if args.mode == 'detect':
        test_ocr_on_image(args.image)
    else:
        test_plate_preprocessing()