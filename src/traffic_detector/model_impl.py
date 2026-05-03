"""
model_impl.py - основной класс для детекции объектов
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2

# Пробуем импортировать YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO не установлен. Установи: poetry add  ultralytics")


class My_LicensePlate_Model:
    """
    Класс для поиска номеров, машин и людей на фото/видео
    
    Как использовать:
        1. detector = My_LicensePlate_Model('путь_к_модели.pt')
        2. результат = detector.detect_plates(кадр)
    """
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        """
        Инициализация детектора
        
        Args:
            model_path: путь к файлу .pt (если None - используем yolov8n.pt)
            confidence: порог уверенности (0.5 = 50%)
        """
        # Настраиваем логирование
        self.logger = logging.getLogger(__name__)
        
        # Проверяем наличие YOLO
        if not YOLO_AVAILABLE:
            raise ImportError("Установите ultralytics: poetry add ultralytics")
        
        # Загружаем модель
        if model_path is None or not Path(model_path).exists():
            self.logger.warning(f"Модель не найдена, загружаем предобученную yolov8n.pt")
            model_path = 'yolov8n.pt'
        
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.logger.info(f"Модель загружена: {model_path}")
        
        # Соответствие цифр и названий классов
        self.class_names = {
            0: 'plate',    # номер
            1: 'person',   # человек
            2: 'car'       # машина
        }
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """
        Находит только номера на кадре
        
        Args:
            frame: картинка (numpy array в формате BGR)
            
        Returns:
            Список словарей. Каждый словарь содержит:
                - 'bbox': [x1, y1, x2, y2] координаты рамки
                - 'confidence': уверенность (0-1)
        """
        # Получаем все объекты
        all_objects = self._detect_all(frame)
        
        # Оставляем только номера
        plates = [obj for obj in all_objects if obj['class'] == 'plate']
        
        self.logger.debug(f"Найдено номеров: {len(plates)}")
        return plates
    
    def detect_all(self, frame: np.ndarray) -> List[Dict]:
        """
        Находит все объекты (номера, машины, людей)
        
        Returns:
            Список всех найденных объектов
        """
        return self._detect_all(frame)
    
    def _detect_all(self, frame: np.ndarray) -> List[Dict]:
        """
        Внутренний метод для поиска всех объектов
        """
        if frame is None:
            self.logger.warning("Пустой кадр")
            return []
        
        try:
            # Запускаем YOLO
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            detections = []
            
            # Обрабатываем результаты
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Координаты рамки
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Уверенность
                    conf = float(box.conf[0])
                    
                    # Класс объекта (0, 1 или 2)
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name,
                        'class_id': class_id
                    })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Ошибка при детекции: {e}")
            return []
    
    def draw_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Рисует рамки на кадре
        
        Args:
            frame: исходное изображение
            detections: результат работы detect_plates или detect_all
            
        Returns:
            Изображение с нарисованными рамками
        """
        # Цвета для разных объектов (BGR формат)
        colors = {
            'plate': (0, 255, 0),    # зеленый
            'car': (255, 0, 0),      # синий
            'person': (0, 0, 255)    # красный
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            
            # Выбираем цвет
            color = colors.get(class_name, (255, 255, 255))
            
            # Рисуем рамку
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Добавляем текст
            text = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


# Простой тест (запусти, чтобы проверить)
if __name__ == "__main__":
    print("Создаем детектор...")
    detector = My_LicensePlate_Model()
    
    # Проверяем, что класс работает
    print("✅ Класс успешно создан!")
    print("Для использования:")
    print("1. Обучи модель или скачай веса")
    print("2. detector = My_LicensePlate_Model('путь_к_модели.pt')")
    print("3. plates = detector.detect_plates(изображение)")