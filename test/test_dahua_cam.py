import cv2
cap = cv2.VideoCapture('rtsp://admin:ctucl2021@@192.168.0.100:554/cam/realmonitor?channel=1&subtype=0')
while True:
    ret, img = cap.read()
    if ret == True:
        cv2.imshow('video output', img)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
