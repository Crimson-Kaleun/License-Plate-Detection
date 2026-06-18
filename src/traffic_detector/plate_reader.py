"""
plate_reader.py - Модуль для распознавания текста на номерных знаках
Поддерживает: латиницу, кириллицу, улучшение изображения
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
import easyocr
from pathlib import Path

class PlateReader:
    """
    Класс для распознавания текста на номерных знаках
    """
    
    def __init__(
        self,
        language: str = 'ru',  # 'en' для латиницы, 'ru' для кириллицы
        use_gpu: bool = True,
        improve_image: bool = False
    ):
        """
        Инициализация ридера номеров
        
        Args:
            language: язык ('en' - латиница, 'ru' - кириллица, 'en,ru' - оба)
            use_gpu: использовать ли GPU для распознавания
            improve_image: улучшать ли изображение перед распознаванием
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.use_gpu = use_gpu
        self.improve_image = improve_image
        
        # Настройка языков для EasyOCR
        self.language_map = {
            'en': ['en'],           # Английский (латиница)
            'ru': ['ru'],           # Русский (кириллица)
            'en,ru': ['en', 'ru'],  # Оба языка
            'all': ['en', 'ru']     # Все доступные
        }
        
        languages = self.language_map.get(language, ['en'])
        self.logger.info(f"Инициализация EasyOCR для языков: {languages}")
        
        try:
            self.reader = easyocr.Reader(
                lang_list=languages,
                gpu=use_gpu,
                model_storage_directory='data/models/easyocr'  # Папка для моделей
            )
            self.logger.info("✅ EasyOCR успешно инициализирован")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации EasyOCR: {e}")
            raise
        
        # Регулярные выражения для фильтрации номеров
        # Российские номера: 1-3 буквы, 2-3 цифры, 2 буквы, 2-3 цифры
        self.ru_plate_pattern = r'^[А-ЯA-Z]{1,3}[0-9]{2,3}[А-ЯA-Z]{2}[0-9]{2,3}$'
        # Европейские номера: 1-2 буквы, 1-4 цифры, 1-2 буквы
        self.eu_plate_pattern = r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$'
        
    def preprocess_plate(
        self,
        plate_image: np.ndarray,
        use_clahe: bool = True,
        use_bilateral: bool = True,
        use_adaptive_threshold: bool = False
    ) -> np.ndarray:
        """
        Предобработка изображения номера для улучшения распознавания
        
        Args:
            plate_image: изображение номерного знака (BGR)
            use_clahe: использовать CLAHE (выравнивание гистограммы)
            use_bilateral: использовать билатеральный фильтр
            use_adaptive_threshold: использовать адаптивный порог
            
        Returns:
            улучшенное изображение (в оттенках серого)
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image
        
        # Конвертируем в grayscale если еще не в нем
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # 1. CLAHE - улучшение контрастности
        if use_clahe:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                self.logger.debug("CLAHE применен")
            except Exception as e:
                self.logger.debug(f"CLAHE не применен: {e}")
        
        # 2. Билатеральный фильтр - шумоподавление с сохранением границ
        if use_bilateral:
            try:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                self.logger.debug("Билатеральный фильтр применен")
            except Exception as e:
                self.logger.debug(f"Билатеральный фильтр не применен: {e}")
        
        # 3. Адаптивный порог - для улучшения контраста текста
        if use_adaptive_threshold:
            try:
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                self.logger.debug("Адаптивный порог применен")
            except Exception as e:
                self.logger.debug(f"Адаптивный порог не применен: {e}")
        
        # 4. Дополнительно: убираем шум маленькими объектами
        if use_bilateral:
            kernel = np.ones((3, 3), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        return gray
    
    def read_plate(
        self,
        plate_image: np.ndarray,
        confidence_threshold: float = 0.3,
        filter_plate_format: bool = True
    ) -> Dict:
        """
        Распознавание текста на номерном знаке
        
        Args:
            plate_image: изображение номера (BGR)
            confidence_threshold: минимальная уверенность (0-1)
            filter_plate_format: фильтровать ли результат по формату номера
            
        Returns:
            Словарь с результатами:
                - 'text': распознанный текст
                - 'confidence': средняя уверенность
                - 'all_results': все результаты распознавания
                - 'preprocessed': улучшенное изображение
        """
        result = {
            'text': '',
            'confidence': 0.0,
            'all_results': [],
            'preprocessed': None
        }
        
        if plate_image is None or plate_image.size == 0:
            self.logger.warning("Пустое изображение номера")
            return result
        
        try:
            # Предобработка изображения
            if self.improve_image:
                processed_img = self.preprocess_plate(
                    plate_image,
                    use_clahe=True,
                    use_bilateral=False,
                    use_adaptive_threshold=False
                )
            else:
                # Просто конвертация в grayscale
                processed_img = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            result['preprocessed'] = processed_img
            
            # Распознавание
            self.logger.debug("Запуск OCR")
            ocr_results = self.reader.readtext(
                processed_img,
                detail=1,  # Возвращать детали с координатами и уверенностью
                paragraph=False,
                min_size=10,
                text_threshold=0.3
            )
            
            if not ocr_results:
                self.logger.debug("Текст не найден")
                return result
            
            # Фильтруем по уверенности
            filtered_results = []
            for bbox, text, confidence in ocr_results:
                if confidence >= confidence_threshold:
                    # Очищаем текст от пробелов и спецсимволов
                    clean_text = ''.join(c for c in text if c.isalnum() or c in '-.')
                    if clean_text:
                        filtered_results.append({
                            'text': clean_text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            # Сортируем по уверенности
            filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Объединяем все найденные тексты
            if filtered_results:
                # Берем самый уверенный результат
                best_result = filtered_results[0]
                result['text'] = best_result['text']
                result['confidence'] = best_result['confidence']
                result['all_results'] = filtered_results
                
                # Проверяем формат номера
                if filter_plate_format:
                    if not self._is_valid_plate(result['text']):
                        self.logger.debug(f"Неверный формат номера: {result['text']}")
                        # Пробуем найти среди всех результатов
                        for res in filtered_results[1:]:
                            if self._is_valid_plate(res['text']):
                                result['text'] = res['text']
                                result['confidence'] = res['confidence']
                                break
            
            self.logger.debug(f"Распознано: {result['text']} (conf: {result['confidence']:.2f})")
            
        except Exception as e:
            self.logger.error(f"Ошибка распознавания: {e}")
        
        return result
    
    def _is_valid_plate(self, text: str) -> bool:
        """
        Проверяет, соответствует ли текст формату номера
        
        Args:
            text: распознанный текст
            
        Returns:
            True если соответствует формату номера
        """
        import re
        
        # Удаляем пробелы и специальные символы
        clean_text = ''.join(c for c in text.upper() if c.isalnum())
        
        # Проверка российского номера
        ru_pattern = r'^[А-ЯA-Z]{1,3}[0-9]{2,3}[А-ЯA-Z]{2}[0-9]{2,3}$'
        # Проверка европейского номера
        eu_pattern = r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$'
        
        if re.match(ru_pattern, clean_text) or re.match(eu_pattern, clean_text):
            return True
        return False
    
    def read_plate_from_bbox(
        self,
        frame: np.ndarray,
        bbox: List[int],
        **kwargs
    ) -> Dict:
        """
        Вырезает номер по координатам и распознает
        
        Args:
            frame: исходный кадр
            bbox: [x1, y1, x2, y2] координаты рамки
            **kwargs: параметры для read_plate
            
        Returns:
            Результат распознавания
        """
        x1, y1, x2, y2 = bbox
        
        # Добавляем отступ для лучшего распознавания
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Вырезаем область номера
        plate_roi = frame[y1:y2, x1:x2]
        
        if plate_roi.size == 0:
            return {'text': '', 'confidence': 0.0}
        
        return self.read_plate(plate_roi, **kwargs)

class PlateReaderWithCache(PlateReader):
    """
    Расширенная версия PlateReader с кешированием результатов
    """
    
    def __init__(self, *args, cache_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size
    
    def read_plate(
        self,
        plate_image: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Распознавание с кешированием
        """
        # Создаем хеш изображения для кеша
        if plate_image is not None:
            import hashlib
            img_hash = hashlib.md5(plate_image.tobytes()).hexdigest()
            
            # Проверяем кеш
            if img_hash in self.cache:
                self.logger.debug(f"Результат из кеша: {self.cache[img_hash]['text']}")
                return self.cache[img_hash]
        
        # Если нет в кеше - распознаем
        result = super().read_plate(plate_image, **kwargs)
        
        # Сохраняем в кеш
        if plate_image is not None and result['text']:
            img_hash = hashlib.md5(plate_image.tobytes()).hexdigest()
            self.cache[img_hash] = result
            
            # Ограничиваем размер кеша
            if len(self.cache) > self.cache_size:
                # Удаляем самый старый элемент
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return result