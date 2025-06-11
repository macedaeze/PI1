from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")        # descarga automática la 1.ª vez
cap = cv2.VideoCapture(0)         # índice 0 = cámara integrada

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, classes=[39])  # 39 = 'bottle' en COCO
    cv2.imshow("Botellas", results[0].plot())
    if cv2.waitKey(1) & 0xFF == 27:       # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
