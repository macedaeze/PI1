"""
Detector de Botellas Simple usando YOLOv11
Este script utiliza el modelo YOLOv11 para detectar botellas en tiempo real desde una cámara.
"""

import cv2
import time
import logging
from ultralytics import YOLO


class SimpleBottleDetector:
    
    def __init__(self, model_path="yolo11n.pt", camera_index=0):
        self.model_path = model_path
        self.camera_index = camera_index
        self.model = None
        self.cap = None
        
        # Inicializo las variables para calcular FPS
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # Configuro logging simple
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self):
        # Inicializa la cámara y maneja posibles errores de backend en Windows, esto lo agregamos por un error que tuvimos en las pruebas.
        self.logger.info("Inicializando cámara...")
    
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    # Probar capturar un frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.logger.info(f"Cámara inicializada correctamente")
                        return True
                self.cap.release()
            except:
                continue
        
        # Si falló, probar sin especificar backend
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.logger.info("Cámara inicializada correctamente")
                    return True
        except:
            pass
        
        self.logger.error("No se pudo inicializar la cámara")
        return False
    
    def initialize(self):
        # Inicializo el modelo y la cámara
        try:
            # Cargar modelo YOLO
            self.logger.info(f"Cargando modelo: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Inicializar cámara
            if not self.initialize_camera():
                return False
             # DIAGNÓSTICO: Ver resolución actual
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Resolución actual: {int(actual_width)}x{int(actual_height)}")
            self.logger.info(f"FPS configurado: {actual_fps}")
            
            # Configurar resolución deseada
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)  # pide MJPEG
            self.cap.set(cv2.CAP_PROP_FPS, 15)              # objetivo lógico
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            
            # Verificar si se aplicó
            new_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            new_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            self.logger.info(f"Resolución después de configurar: {int(new_width)}x{int(new_height)}")
    
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en inicialización: {e}")
            return False
    
    def update_fps(self):
        """Actualiza el cálculo de FPS."""
        self.frame_count += 1
        if self.frame_count >= 10:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
    
    def draw_info(self, frame, inference_time, total_time):
        """Dibuja información en el frame."""
        height, width = frame.shape[:2]
        
        # Información principal
        info_text = f"FPS: {self.fps:.1f} | Inferencia: {inference_time*1000:.1f}ms | Tiempo Total: {total_time*1000:.1f}ms"
        
        # Fondo negro para el texto
        (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
        
        # Texto verde
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Ejecuta el detector."""
        if not self.initialize():
            return
        
        self.logger.info("Detector iniciado. Presiona 'q' o ESC para salir")
        
        try:
            while True:

                # INICIO del tiempo total
                total_start_time = time.time()

                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Error capturando frame")
                    break
                
                # Detectar botellas
                #El tiempo de inferencia es el tiempo entre
                start_time = time.time()
                results = self.model(frame, classes=[39], conf=0.25, verbose=False)
                inference_time = time.time() - start_time
                
                # Dibujar detecciones
                annotated_frame = results[0].plot()
                
                # FIN del tiempo total
                total_time = time.time() - total_start_time

                # Actualizar FPS y dibujar info
                self.update_fps()
                display_frame = self.draw_info(annotated_frame, inference_time, total_time)
                
                cv2.namedWindow("Detector de Botellas", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detector de Botellas", 800, 600)  # tamaño inicial

                # Mostrar frame
                cv2.imshow("Detector de Botellas", display_frame)
                
                # Salir con 'q' o ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Interrumpido por el usuario")
        except Exception as e:
            self.logger.error(f"Error durante ejecución: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Recursos liberados")


def main():
    """Función principal."""
    import sys
    
    # Parámetros simples
    model_path = "yolo11n.pt"
    camera_index = 0
    
    # Si se pasan argumentos, usarlos
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except:
            print("Uso: python detector.py [índice_cámara]")
            print("Ejemplo: python detector.py 1")
            return
    
    # Crear y ejecutar detector
    detector = SimpleBottleDetector(model_path, camera_index)
    detector.run()


if __name__ == "__main__":
    main()