import cv2
import threading
import queue
import time
from yolo_onnx import YOLOv8
from byte_tracker import BYTETracker
import random
import numpy as np

BUFFER_SIZE = 10000
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
stop_event = threading.Event()  # Evento para indicar que se debe detener el programa

def capture_frames():
    cap = cv2.VideoCapture("rtsp://admin:ctucl2021@@192.168.0.101:554/cam/realmonitor?channel=1&subtype=0")
    while cap.isOpened() and not stop_event.is_set():  # Verificamos el evento de parada
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
            print(frame_queue.qsize())
        else:
            print("Buffer lleno. Descartando frame...")
    cap.release()

def process_frames():
    class ByteTrackConfig:
        def __init__(self):
            self.track_thresh = 0.5  
            self.track_buffer = 30  
            self.match_thresh = 0.8 
            self.mot20 = False
    tracker_cfg = ByteTrackConfig()
    tracker = BYTETracker(tracker_cfg, frame_rate=30)
    model_path = "models/best_consorcio_test1.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
    contador_objetos = 0
    linea_y = 600 
    last_ids = set()
    random_read = True
    range_random = 20
    aux_rand = range_random
    i_r = 0
    while not frame_queue.empty() or not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            if random_read:
                boxes, scores, _ = yolov8_detector(frame)            
                if len(scores) > 0:
                    dets = []
                    i = 0
                    for result in boxes:
                        x1, y1, x2, y2 = result
                        conf = scores[i]
                        dets.append([x1, y1, x2, y2, conf])
                        i += 1
                    dets = np.array(dets, dtype=np.float32)
                    online_targets = tracker.update(dets, frame.shape, frame.shape)
                    for target in online_targets:
                        tlwh = target.tlwh
                        track_id = target.track_id
                        x1, y1, w, h = tlwh
                        x2, y2 = int(x1 + w), int(y1 + h)
                        cx, cy = int(x1 + w / 2), int(y1 + h / 2)
                        cx, cy = int(x1 + w / 2), int(y1 + h / 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        if linea_y - 5 < cy < linea_y + 15 and track_id not in last_ids:
                            last_ids.add(track_id)
                            contador_objetos += 1

                    cv2.line(frame, (0, linea_y), (frame.shape[1], linea_y), (0, 0, 255), 2)
                    cv2.rectangle(frame, (40, 200), (550, 320), (0, 0, 0), -1)
                    cv2.putText(frame, f'Pasajeros: {contador_objetos}', (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                else:
                    random_read = False
                    aux_rand = random.randint(10,range_random)
            else:
                if i_r >= aux_rand:
                    random_read = True
                    i_r = 0
                else:
                    i_r +=1
            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Detected Objects", frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            time.sleep(0.01)

    cv2.destroyAllWindows()
    stop_event.set() 


capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()


capture_thread.join()
process_thread.join()
