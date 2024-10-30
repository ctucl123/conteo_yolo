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
last_ids = set()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes, scores, _ = yolov8_detector(frame)
    
    dets = []
    i = 0
    for result in boxes:
        x1, y1, x2, y2 =result
        conf = scores[i]
        dets.append([x1, y1, x2, y2, conf])
        i+=1
    dets = np.array(dets, dtype=np.float32)
    print(dets)
    if len(dets) >0:
        online_targets = tracker.update(dets, frame.shape,frame.shape)
        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            x1, y1, w, h = tlwh
            x2, y2 = int(x1 + w), int(y1 + h)
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            if linea_y - 5 < cy < linea_y + 15 and track_id not in last_ids:
                last_ids.add(track_id)
                contador_objetos += 1
                print(contador_objetos)

