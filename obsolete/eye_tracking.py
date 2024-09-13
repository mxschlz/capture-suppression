import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 2)
    gaze.refresh(frame)
    new_frame = gaze.annotated_frame()
    text = (f"Vertical: {round(gaze.vertical_ratio(), 2)} \n"
            f"Horizontal: {round(gaze.horizontal_ratio(), 2)}")

    cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
    cv2.imshow("Demo", new_frame)

    if cv2.waitKey(1) == 27:
        break
