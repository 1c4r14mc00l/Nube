import threading
import queue
import time
import cv2
from datetime import datetime
from kafka import KafkaProducer
import json
import base64
from ultralytics import YOLO
import torch
import logging
import os
from typing import List, Dict, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('YOLOProducer')

# Constantes de configuración
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
FPS_LIMIT = int(os.getenv('FPS_LIMIT', '10'))
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'broker:29092')

# Leer secrets de Docker
def read_secret(secret_path: str) -> str:
    """Lee un secret de Docker con manejo robusto de errores"""
    if not secret_path:
        logger.error("Ruta de secret no proporcionada")
        return ""
    
    try:
        with open(secret_path, 'r') as f:
            secret = f.read().strip()
            if not secret:
                logger.error(f"Secret {secret_path} está vacío")
            return secret
    except IOError as e:
        logger.error(f"Error leyendo secret {secret_path}: {e}")
        return ""

RTSP_USERNAME = read_secret(os.getenv('RTSP_USERNAME_FILE', ''))
RTSP_PASSWORD = read_secret(os.getenv('RTSP_PASSWORD_FILE', ''))

# Configuración de streams RTSP (ahora con credenciales seguras)
STREAMS_CONFIG = [
    {
        "url": f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@129.222.172.5:10554/Streaming/Channels/102",
        "topic": "transmision_cam1"
    },
    {
        "url": f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@129.222.172.5:1500/Streaming/Channels/102",
        "topic": "transmision_cam2"
    },
    {
        "url": f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@129.222.172.5:2525/Streaming/Channels/102", 
        "topic": "transmision_cam3"
    }
]

class YOLODetector:
    def __init__(self, model_path: str = 'best.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Inicializando modelo YOLO en {self.device}")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            # Warmup GPU
            self.model.predict(torch.zeros(1, 3, 640, 640).to(self.device))
            logger.info("Modelo YOLO cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def detect_batch(self, frames: List) -> List[Dict[str, Any]]:
        """Procesa un lote de frames"""
        try:
            results = self.model(frames, verbose=False)
            return results
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.warning("OOM: Liberando memoria GPU")
                torch.cuda.empty_cache()
            raise

class KafkaStreamProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=16384,   #Aprender a calcular esto
            linger_ms=100,      #y esto
            compression_type='gzip'
        )
        logger.info(f"Productor Kafka conectado a {KAFKA_BROKER}")

    def send_detection(self, topic: str, data: Dict):   #Revisar qué es DICT
        try:
            self.producer.send(topic, value=data)   #Confirmar topics
        except Exception as e:
            logger.error(f"Error enviando a Kafka: {e}")

def imagen_a_base64(image) -> str:
    """Optimización: Reduce calidad JPEG para menor tamaño"""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')

def procesar_stream(rtsp_url: str, nombre_topico: str, detector: YOLODetector):
    """Maneja el stream de video para una cámara"""
    cam_id = next((k for k, v in TOPICO_A_CAM_ID.items() if v == nombre_topico), 0)
    kafka_producer = KafkaStreamProducer()
    cap = cv2.VideoCapture(rtsp_url)
    frame_queue = queue.Queue(maxsize=1)
    ultima_deteccion = {"cam_id": cam_id, "timestamp": None, "detecciones": []}
    frame_interval = 1.0 / FPS_LIMIT

    def leer_frames():
        """Hilo para lectura eficiente de frames"""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Error leyendo stream: {rtsp_url}")
                break
            
            frame_resized = cv2.resize(frame, (640, 640))
            if frame_queue.full():
                frame_queue.get()  # Descarta el frame más antiguo
            frame_queue.put(frame_resized)

    def procesar_frame():
        """Hilo principal de procesamiento"""
        nonlocal ultima_deteccion
        frames_batch = []
        timestamps_batch = []
        
        while cap.isOpened():
            start_time = time.time()
            
            if not frame_queue.empty():
                frame = frame_queue.get()
                timestamp = datetime.now().isoformat()
                
                # Acumular para batch processing
                frames_batch.append(frame)
                timestamps_batch.append(timestamp)
                
                # Procesar cuando se llena el batch
                if len(frames_batch) == BATCH_SIZE:
                    try:
                        results = detector.detect_batch(frames_batch)
                        
                        for i, result in enumerate(results):
                            detecciones = [{
                                "clase": int(box.cls.cpu().numpy()),
                                "confianza": float(box.conf.cpu().numpy()),
                                "coordenadas": box.xyxy.cpu().numpy().tolist()
                            } for box in result.boxes]
                            
                            ultima_deteccion.update({
                                "timestamp": timestamps_batch[i],
                                "detecciones": detecciones
                            })
                            
                            frame_data = {
                                "cam_id": cam_id,
                                "timestamp": timestamps_batch[i],
                                "frame_base64": imagen_a_base64(result.plot())
                            }
                            kafka_producer.send_detection(nombre_topico, frame_data)
                            
                        frames_batch.clear()
                        timestamps_batch.clear()
                        
                    except Exception as e:
                        logger.error(f"Error procesando batch: {e}")
                        frames_batch.clear()
                        timestamps_batch.clear()
                        continue

                # Control FPS
                elapsed = time.time() - start_time
                time.sleep(max(0, frame_interval - elapsed))

    def enviar_detecciones():
        """Hilo para envío periódico de detecciones"""
        while cap.isOpened():
            if ultima_deteccion["timestamp"]:
                kafka_producer.send_detection("detecciones", ultima_deteccion)
            time.sleep(2)

    # Configuración de hilos
    hilos = [
        threading.Thread(target=leer_frames, daemon=True),
        threading.Thread(target=procesar_frame, daemon=True),
        threading.Thread(target=enviar_detecciones, daemon=True)
    ]
    
    for hilo in hilos:
        hilo.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Deteniendo procesamiento...")
    finally:
        cap.release()
        kafka_producer.producer.close()

if __name__ == '__main__':
    # Validar credenciales
    if not RTSP_USERNAME or not RTSP_PASSWORD:
        logger.error("No se encontraron credenciales RTSP en los secrets!")
        exit(1)

    detector = YOLODetector('best.pt')
    
    # Un hilo por stream
    hilos_streams = []
    for stream in STREAMS_CONFIG:
        hilo = threading.Thread(
            target=procesar_stream,
            args=(stream["url"], stream["topic"], detector),
            daemon=True
        )
        hilo.start()
        hilos_streams.append(hilo)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Apagando el sistema...")