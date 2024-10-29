import cv2
from yolo_onnx import YOLOv8
from byte_tracker import BYTETracker
import numpy as np

class ByteTrackConfig:
    def __init__(self):
        self.track_thresh = 0.5  
        self.track_buffer = 30  
        self.match_thresh = 0.8 
        self.mot20 = False
tracker_cfg = ByteTrackConfig()
tracker = BYTETracker(tracker_cfg, frame_rate=30)

video_path = 'video/test1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video.")
    exit()
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
contador_objetos = 0
linea_y = 900 
last_ids = []
while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break
    try:
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue
    boxes, scores, class_ids = yolov8_detector(frame)
    dets = []
    i = 0
    for result in boxes:
        x1, y1, x2, y2 =result
        conf = scores[i]
        dets.append([x1, y1, x2, y2, conf])
        i+=1
    dets = np.array(dets, dtype=np.float32)
    if len(dets) >0:
        online_targets = tracker.update(dets, frame.shape,frame.shape)
        for target in online_targets:
            tlwh = target.tlwh 
            track_id = target.track_id
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if cy > linea_y - 5 and cy < linea_y + 15:
                if track_id in last_ids:
                    pass
                else:
                    last_ids.append(track_id)
                    contador_objetos += 1


    cv2.line(frame, (0, linea_y), (frame.shape[1], linea_y), (0, 0, 255), 2)
    cv2.rectangle(frame, (40, 200), (550, 320), (0, 0, 0), -1)
    cv2.putText(frame, f'Pasajeros: {contador_objetos}', (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Detected Objects", frame_resized)

