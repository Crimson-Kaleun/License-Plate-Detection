# test_model.py
import cv2
import sys
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


def test_on_folder(model_path: str, input_folder: str):
    """
    Обрабатывает все изображения в папке и сохраняет результаты в папку results
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ Папка не найдена: {input_folder}")
        return
    
    # Создаем папку для результатов
    output_folder = input_path / "results"
    output_folder.mkdir(exist_ok=True)
    
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Собираем все изображения
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ Изображения не найдены в папке: {input_folder}")
        return
    
    print(f"🔍 Найдено изображений: {len(image_files)}")
    
    # Загружаем модель один раз для всех изображений
    print("📦 Загрузка модели...")
    detector = My_LicensePlate_Model(model_path=model_path, confidence=0.5)
    
    # Обрабатываем каждое изображение
    for i, image_file in enumerate(image_files, 1):
        print(f"\n📸 [{i}/{len(image_files)}] Обработка: {image_file.name}")
        
        # Загружаем изображение
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"   ⚠️ Не удалось загрузить: {image_file.name}")
            continue
        
        # Детектируем объекты
        plates = detector.detect_plates(image)
        all_objects = detector.detect_all(image)
        
        print(f"   Найдено номеров: {len(plates)}")
        print(f"   Найдено всего объектов: {len(all_objects)}")
        
        # Рисуем рамки
        result_image = detector.draw_boxes(image.copy(), all_objects)
        
        # Сохраняем результат
        output_path = output_folder / f"result_{image_file.name}"
        cv2.imwrite(str(output_path), result_image)
        print(f"   ✅ Сохранено: {output_path}")
    
    print(f"\n🎉 Обработка завершена! Результаты сохранены в: {output_folder}")


if __name__ == "__main__":
    # Путь к обученной модели
    model_path = "data/weights/best.pt"
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]
        
        # Если передан путь к папке
        if Path(input_arg).is_dir():
            print(f"📁 Режим обработки папки: {input_arg}")
            test_on_folder(model_path, input_arg)
        # Если передан путь к файлу
        elif Path(input_arg).is_file():
            print(f"🖼️ Режим обработки одного файла: {input_arg}")
            test_on_image(model_path, input_arg)
        else:
            print(f"❌ Указанный путь не существует: {input_arg}")
    else:
        # По умолчанию проверяем папку data/test
        default_folder = "data/test"
        
        if Path(default_folder).exists():
            print(f"📁 Автоматический режим обработки папки: {default_folder}")
            test_on_folder(model_path, default_folder)
        else:
            # Если папки нет, ищем test_image.jpg (старое поведение)
            test_image = "test_image.jpg"
            
            if Path(test_image).exists():
                test_on_image(model_path, test_image)
            else:
                print("❌ Использование:")
                print("   python test_model.py <путь_к_изображению>")
                print("   python test_model.py <путь_к_папке>")
                print(f"   Или положите изображения в папку {default_folder}/")