import cv2
import sys
sys.path.append('../..')
import config

cap = cv2.VideoCapture(config.VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % 30 == 0:
        filename = f'frame_{count}.jpg'
        cv2.imwrite(f'path/to/output/{filename}', frame)

    count += 1

cap.release()