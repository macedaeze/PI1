"""
Detector de Botellas usando YOLOv11
====================================

Este módulo implementa un detector de botellas en tiempo real usando YOLOv11.
Incluye métricas de rendimiento, configuración flexible y logging.

Autor: Usuario
Fecha: 2025
"""

import cv2
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionConfig:
    """Configuración para el detector de botellas."""
    model_path: str = "yolo11n.pt"
    camera_index: int = 0
    confidence_threshold: float = 0.25
    target_classes: List[int] = None  # [39] para botellas
    window_name: str = "Detector de Botellas"
    fps_update_interval: int = 10
    
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = [39]  # Clase 39 = botella en COCO dataset


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del detector."""
    fps: float = 0.0
    inference_time_ms: float = 0.0
    frame_count: int = 0
    total_detections: int = 0
    resolution: Tuple[int, int] = (0, 0)
    
    def reset_fps_counter(self):
        """Resetea el contador de FPS."""
        self.frame_count = 0


class FPSCalculator:
    """Calculadora de FPS optimizada."""
    
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
    
    def update(self) -> float:
        """Actualiza el cálculo de FPS y retorna el valor actual."""
        self.frame_count += 1
        
        if self.frame_count >= self.update_interval:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.current_fps = self.frame_count / elapsed
            
            # Reset para el próximo cálculo
            self.frame_count = 0
            self.start_time = current_time
        
        return self.current_fps


class DetectionVisualizer:
    """Maneja la visualización de detecciones y métricas."""
    
    @staticmethod
    def draw_metrics(frame: np.ndarray, metrics: PerformanceMetrics) -> np.ndarray:
        """Dibuja las métricas de rendimiento en el frame."""
        height, width = frame.shape[:2]
        
        # Información principal
        main_info = (
            f"FPS: {metrics.fps:5.1f} | "
            f"Resolución: {width}x{height} | "
            f"Inferencia: {metrics.inference_time_ms:5.1f}ms"
        )
        
        # Información adicional
        detection_info = f"Detecciones: {metrics.total_detections}"
        
        # Configuración del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Colores
        bg_color = (0, 0, 0)  # Negro para fondo
        text_color = (0, 255, 0)  # Verde para texto
        
        # Dibujar fondo para mejor legibilidad
        DetectionVisualizer._draw_text_with_background(
            frame, main_info, (10, 25), font, font_scale, 
            text_color, bg_color, thickness
        )
        
        DetectionVisualizer._draw_text_with_background(
            frame, detection_info, (10, 55), font, font_scale, 
            text_color, bg_color, thickness
        )
        
        return frame
    
    @staticmethod
    def _draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int],
                                 font: int, font_scale: float, text_color: Tuple[int, int, int],
                                 bg_color: Tuple[int, int, int], thickness: int):
        """Dibuja texto con fondo para mejor legibilidad."""
        # Obtener dimensiones del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        # Dibujar rectángulo de fondo
        cv2.rectangle(frame, 
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + baseline + 5),
                     bg_color, -1)
        
        # Dibujar texto
        cv2.putText(frame, text, position, font, font_scale, text_color, thickness)


class BottleDetector:
    """Detector principal de botellas usando YOLO."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps_calculator = FPSCalculator(config.fps_update_interval)
        self.metrics = PerformanceMetrics()
        self.is_running = False
        
        # Configurar logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Inicializa el modelo YOLO y la cámara."""
        try:
            # Cargar modelo YOLO
            self.logger.info(f"Cargando modelo YOLO: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            
            # Inicializar cámara
            self.logger.info(f"Inicializando cámara: {self.config.camera_index}")
            self.cap = cv2.VideoCapture(self.config.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo abrir la cámara")
            
            # Configurar resolución si es posible
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.logger.info("Inicialización completada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error durante la inicialización: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Ejecuta la detección en un frame."""
        if self.model is None:
            raise RuntimeError("Modelo no inicializado")
        
        # Realizar inferencia
        start_time = time.time()
        results = self.model(
            frame,
            classes=self.config.target_classes,
            conf=self.config.confidence_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # Actualizar métricas
        self.metrics.inference_time_ms = inference_time * 1000
        
        # Obtener frame anotado
        annotated_frame = results[0].plot()
        
        # Contar detecciones
        detections_count = len(results[0].boxes) if results[0].boxes is not None else 0
        self.metrics.total_detections = detections_count
        
        return annotated_frame, detections_count
    
    def update_metrics(self, frame: np.ndarray):
        """Actualiza las métricas de rendimiento."""
        height, width = frame.shape[:2]
        self.metrics.resolution = (width, height)
        self.metrics.fps = self.fps_calculator.update()
    
    def run(self):
        """Ejecuta el bucle principal de detección."""
        if not self.initialize():
            return
        
        self.is_running = True
        self.logger.info("Iniciando detección de botellas...")
        self.logger.info("Presiona 'ESC' para salir, 'q' para salir, 's' para screenshot")
        
        screenshot_counter = 0
        
        try:
            while self.is_running:
                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("No se pudo capturar frame de la cámara")
                    break
                
                # Detectar objetos
                annotated_frame, detections = self.detect_objects(frame)
                
                # Actualizar métricas
                self.update_metrics(annotated_frame)
                
                # Dibujar métricas
                display_frame = DetectionVisualizer.draw_metrics(
                    annotated_frame, self.metrics
                )
                
                # Mostrar frame
                cv2.imshow(self.config.window_name, display_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC o 'q' para salir
                    break
                elif key == ord('s'):  # 's' para screenshot
                    screenshot_path = f"screenshot_{screenshot_counter:03d}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    self.logger.info(f"Screenshot guardado: {screenshot_path}")
                    screenshot_counter += 1
                elif key == ord('r'):  # 'r' para resetear métricas
                    self.metrics = PerformanceMetrics()
                    self.fps_calculator = FPSCalculator(self.config.fps_update_interval)
                    self.logger.info("Métricas reseteadas")
                
        except KeyboardInterrupt:
            self.logger.info("Interrupción por teclado recibida")
        except Exception as e:
            self.logger.error(f"Error durante la ejecución: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos y cierra ventanas."""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.logger.info("Recursos liberados correctamente")


def create_argument_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Detector de botellas en tiempo real usando YOLOv11"
    )
    
    parser.add_argument(
        "--model", "-m", 
        default="yolo11n.pt",
        help="Ruta al modelo YOLO (default: yolo11n.pt)"
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int, default=0,
        help="Índice de la cámara (default: 0)"
    )
    
    parser.add_argument(
        "--confidence", "-conf",
        type=float, default=0.25,
        help="Umbral de confianza para detecciones (default: 0.25)"
    )
    
    parser.add_argument(
        "--classes",
        nargs="+", type=int, default=[39],
        help="Clases a detectar (default: 39 para botellas)"
    )
    
    return parser


def main():
    """Función principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Crear configuración
    config = DetectionConfig(
        model_path=args.model,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        target_classes=args.classes
    )
    
    # Crear y ejecutar detector
    detector = BottleDetector(config)
    detector.run()


if __name__ == "__main__":
    main()