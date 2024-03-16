# module as an example of how should we apply frame difference
import cv2
import numpy as np
cap = cv2.VideoCapture('yolo/video.mp4')
ret, prev_frame = cap.read()


threshold = 10000

while True:
    ret, current_frame = cap.read()
    if not ret:
        break
    frame_diff = cv2.absdiff(current_frame, prev_frame)
    diff_sum = np.sum(frame_diff)


    if diff_sum > threshold:
        prev_frame = current_frame

    cv2.imshow('Frame', current_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break
cap.release()
cv2.destroyAllWindows()
