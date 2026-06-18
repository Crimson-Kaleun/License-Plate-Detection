"""
model_impl.py - основной класс для детекции и распознавания объектов
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import cv2

from ultralytics import YOLO
from src.traffic_detector.plate_reader import PlateReader, PlateReaderWithCache


class My_LicensePlate_Model:
    """
    Класс для детекции объектов и распознавания номеров
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path] = "data/weights/best.pt",
        confidence: float = 0.5,
        device: str = "cuda",
        enable_ocr: bool = True,
        ocr_language: str = 'en',  # 'en' или 'ru' или 'en,ru'
        use_ocr_cache: bool = True,
        improve_plate_image: bool = True
    ):
        """
        Инициализация детектора
        
        Args:
            model_path: путь к файлу весов модели (.pt)
            confidence: порог уверенности (0-1)
            device: устройство ('cpu' или 'cuda')
            enable_ocr: включить ли распознавание текста
            ocr_language: язык для распознавания ('en', 'ru', 'en,ru')
            use_ocr_cache: использовать кеш для OCR
            improve_plate_image: улучшать ли изображение номера
        """
        # Настройка логгера
        self.logger = logging.getLogger(__name__)
        
        # Загрузка модели YOLO
        try:
            self.model = YOLO(str(model_path))
            self.logger.info(f"✅ Модель YOLO загружена из {model_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
        
        self.confidence = confidence
        self.device = device
        self.enable_ocr = enable_ocr
        
        # Классы для детекции
        self.class_names = {0: 'plate', 1: 'person', 2: 'car'}
        
        # Инициализация OCR
        self.plate_reader = None
        if enable_ocr:
            try:
                reader_class = PlateReaderWithCache if use_ocr_cache else PlateReader
                self.plate_reader = reader_class(
                    language=ocr_language,
                    use_gpu=(device == 'cuda' or device == 0),
                    improve_image=improve_plate_image
                )
                self.logger.info(f"✅ OCR инициализирован (язык: {ocr_language})")
            except Exception as e:
                self.logger.warning(f"⚠️ OCR не доступен: {e}")
                self.enable_ocr = False
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция номерных знаков с распознаванием текста
        
        Returns:
            Список словарей с информацией о номерах:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': уверенность детекции
                - 'class': 'plate'
                - 'plate_text': распознанный текст
                - 'plate_confidence': уверенность распознавания
        """
        detections = self._detect_objects(frame, class_filter='plate')
        
        # Если включен OCR - распознаем текст на каждом номере
        if self.enable_ocr and self.plate_reader and detections:
            for det in detections:
                plate_result = self.plate_reader.read_plate_from_bbox(
                    frame,
                    det['bbox']
                )
                det['plate_text'] = plate_result.get('text', '')
                det['plate_confidence'] = plate_result.get('confidence', 0.0)
                det['plate_preprocessed'] = plate_result.get('preprocessed')
        
        return detections
    
    def detect_all(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция всех объектов
        """
        return self._detect_objects(frame, class_filter=None)
    
    def _detect_objects(
        self, 
        frame: np.ndarray, 
        class_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Внутренний метод для детекции объектов
        """
        if frame is None:
            self.logger.warning("Получен пустой кадр")
            return []
        
        try:
            results = self.model(
                frame, 
                conf=self.confidence,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    if class_filter and class_name != class_filter:
                        continue
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id
                    })
            
            self.logger.debug(f"Найдено объектов: {len(detections)}")
            return detections
            
        except Exception as e:
            self.logger.error(f"Ошибка при детекции: {e}")
            return []
    
    def draw_boxes(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        show_text: bool = True,
        show_bbox: bool = True
    ) -> np.ndarray:
        """
        Рисует рамки и текст на кадре
        """
        colors = {
            'plate': (0, 255, 0),    # зеленый
            'car': (255, 0, 0),      # синий
            'person': (0, 0, 255)    # красный
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Рисуем рамку
            if show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Подготовка текста
            label = f"{class_name}: {confidence:.2f}"
            if class_name == 'plate' and 'plate_text' in det and det['plate_text']:
                label += f" [{det['plate_text']}]"
            
            # Рисуем текст
            if show_text:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_size[1] - 5), 
                    (x1 + label_size[0], y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (188, 188, 188), 
                    1
                )
        
        return frame