# test_model.py
import cv2
from pathlib import Path
from src.traffic_detector.model_impl import My_LicensePlate_Model

def test_on_image(model_path: str, image_path: str):
    """
    Тестирует модель на одном изображении
    """
    # Загружаем модель
    detector = My_LicensePlate_Model(model_path=model_path, confidence=0.5)
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return
    
    # Детектируем объекты
    plates = detector.detect_plates(image)
    all_objects = detector.detect_all(image)
    
    print(f"\n📸 Результаты для {Path(image_path).name}:")
    print(f"   Найдено номеров: {len(plates)}")
    print(f"   Найдено всего объектов: {len(all_objects)}")
    
    for i, plate in enumerate(plates):
        print(f"   Номер {i+1}: уверенность {plate['confidence']:.2f}")
    
    # Рисуем рамки
    result_image = detector.draw_boxes(image.copy(), all_objects)
    
    # Сохраняем результат
    output_path = f"result_{Path(image_path).name}"
    cv2.imwrite(output_path, result_image)
    print(f"✅ Результат сохранен в: {output_path}")
    
    # Показываем изображение (если есть GUI)
    try:
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    # Путь к обученной модели
    model_path = "data/weights/best.pt"
    
    # Путь к тестовому изображению (замени на свой)
    test_image = "test_image.jpg"  # Положи какую-нибудь фотку с машиной
    
    if Path(test_image).exists():
        test_on_image(model_path, test_image)
    else:
        print(f"❌ Тестовое изображение не найдено: {test_image}")
        print("Положите фотографию с машиной в папку проекта и назовите test_image.jpg")