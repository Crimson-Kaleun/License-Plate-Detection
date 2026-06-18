#!/usr/bin/env python3
"""
CLI приложение для детекции номеров
Поддерживает: видеофайлы и веб-камеру
"""
import argparse
import cv2
import logging
from pathlib import Path
from datetime import datetime
import sys

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traffic_detector.model_impl import My_LicensePlate_Model
from src.traffic_detector.logger_config import setup_logger

class LicensePlateApp:
    def __init__(self, model_path: str, confidence: float = 0.5,
             enable_ocr: bool = False, ocr_lang: str = 'en',
             improve: bool = True):
        """
        Инициализация приложения
        """
        self.logger = setup_logger()
        self.logger.info(f"Запуск приложения с моделью: {model_path}")
        
        try:
            self.detector = My_LicensePlate_Model(
                model_path=model_path,
                confidence=confidence,
                device='cuda',  # или 'cpu'
                enable_ocr=enable_ocr,
                ocr_language=ocr_lang,
                improve_plate_image=improve
            )
            self.logger.info("Детектор успешно загружен")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def process_video(self, input_path: str, output_path: str = None):
        """
        Обработка видеофайла
        
        Args:
            input_path: путь к входному видео
            output_path: путь для сохранения результата
        """
        self.logger.info(f"Обработка видео: {input_path}")
        
        # Открываем видео
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Не удалось открыть видео: {input_path}")
            return
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Подготавливаем выходной файл
        if output_path is None:
            output_path = f"output_{Path(input_path).stem}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_plates = 0
        
        self.logger.info(f"Начинаем обработку. FPS: {fps}, Размер: {width}x{height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Детекция номеров
                plates = self.detector.detect_plates(frame)
                total_plates += len(plates)
                
                # Рисуем рамки
                frame = self.detector.draw_boxes(frame, plates)
                
                # Добавляем информацию на кадр
                cv2.putText(frame, f"Plates: {len(plates)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Сохраняем кадр
                out.write(frame)
                
                # Логируем каждые 100 кадров
                if frame_count % 100 == 0:
                    self.logger.info(f"Обработано кадров: {frame_count}, найдено номеров: {total_plates}")
        
        finally:
            cap.release()
            out.release()
        
        self.logger.info(f"✅ Обработка завершена!")
        self.logger.info(f"   Всего кадров: {frame_count}")
        self.logger.info(f"   Всего найдено номеров: {total_plates}")
        self.logger.info(f"   Результат сохранен: {output_path}")
    
    def process_webcam(self, camera_id: int = 0):
        """
        Обработка потока с веб-камеры в реальном времени
        
        Args:
            camera_id: ID веб-камеры (обычно 0)
        """
        self.logger.info(f"Запуск веб-камеры (ID: {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"Не удалось открыть веб-камеру {camera_id}")
            return
        
        self.logger.info("Нажмите 'q' для выхода, 's' для сохранения скриншота")
        
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Не удалось получить кадр с камеры")
                    break
                
                frame_count += 1
                
                # Детекция (каждый кадр или каждый 2-й для производительности)
                if frame_count % 2 == 0:  # Обрабатываем каждый 2-й кадр для скорости
                    plates = self.detector.detect_plates(frame)
                    
                    # Рисуем рамки
                    frame = self.detector.draw_boxes(frame, plates)
                    
                    # Добавляем информацию
                    cv2.putText(frame, f"Plates: {len(plates)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Логируем каждые 100 кадров
                    if frame_count % 100 == 0:
                        self.logger.info(f"Кадров обработано: {frame_count}, найдено номеров: {len(plates)}")
                
                # Показываем кадр
                cv2.imshow('License Plate Detection', frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Выход по запросу пользователя")
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    self.logger.info(f"Скриншот сохранен: {screenshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        self.logger.info(f"✅ Работа с камерой завершена. Всего кадров: {frame_count}")

def main():
    """Главная функция для CLI"""
    parser = argparse.ArgumentParser(description='License Plate Detection System')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['video', 'cam'],
                       help='Режим работы: video (обработка видео) или cam (веб-камера)')
    
    parser.add_argument('--input', type=str, default=None,
                       help='Путь к входному видео (для режима video)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения результата (для режима video)')
    
    parser.add_argument('--model', type=str, default='data/weights/best.pt',
                       help='Путь к модели (по умолчанию: data/weights/best.pt)')
    
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Порог уверенности (0-1), по умолчанию: 0.5')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='ID веб-камеры (для режима cam), по умолчанию: 0')
    
    parser.add_argument('--ocr', action='store_true',
                   help='Включить распознавание текста на номерах')
    parser.add_argument('--ocr-lang', type=str, default='en,ru',
                    choices=['en', 'ru', 'en,ru'],
                    help='Язык для распознавания (en - латиница, ru - кириллица)')
    parser.add_argument('--improve', action='store_true',
                    help='Улучшать изображение номера перед распознаванием')
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if args.mode == 'video' and not args.input:
        parser.error("Для режима video необходимо указать --input")
    
    # Создаем приложение
    try:
        app = LicensePlateApp(model_path=args.model, confidence=args.confidence)
        
        if args.mode == 'video':
            app.process_video(args.input, args.output)
        elif args.mode == 'cam':
            app.process_webcam(args.camera)
            
    except Exception as e:
        logging.error(f"Ошибка приложения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()