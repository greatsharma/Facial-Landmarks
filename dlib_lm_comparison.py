import cv2
import time
import dlib
import numpy as np


face_detector = dlib.get_frontal_face_detector()

custom_lm1 = dlib.shape_predictor("models/custom1.dat")
custom_lm2 = dlib.shape_predictor("models/custom3.dat")
custom_lm3 = dlib.shape_predictor("models/custom4.dat")
default_lm = dlib.shape_predictor("models/default.dat")

color_palette = [(0,0,0), (94,70,9), (217,217,24)]


def draw_text_with_backgroud(img, text, x, y, font_scale, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                            background=(0,0,0), foreground=(255,255,255), box_coords_1=(-5,5), box_coords_2=(5,-5)):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((x+box_coords_1[0], y+box_coords_1[1]), (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]))
    cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, fontScale=font_scale, color=foreground, thickness=thickness)


def annotate_landmarks(model, frame, gray_frame, face):
    lm = model(gray_frame, face)
    for i in range(68):
        cv2.circle(frame, (lm.part(i).x, lm.part(i).y), 2, color_palette[0],-1,)


vidcap = cv2.VideoCapture(0)
frame_count = 0
tt_default = 0
tt_custom1 = 0
tt_custom2 = 0
tt_custom3 = 0

cv2.namedWindow("facial landmarks detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("facial landmarks detector", 1400, 700)

while True:
    status, frame_1 = vidcap.read()
    if not status:
        break

    frame_count += 1

    frame_1 = cv2.flip(frame_1, 1, 0)
    frame_2 = frame_1.copy()
    frame_3 = frame_1.copy()
    frame_4 = frame_1.copy()

    gray_frame = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame)
    for face in faces:
        tik = time.time()
        annotate_landmarks(default_lm, frame_1, gray_frame, face)
        tt_default += time.time() - tik

        tik = time.time()
        annotate_landmarks(custom_lm1, frame_2, gray_frame, face)
        tt_custom1 += time.time() - tik

        tik = time.time()
        annotate_landmarks(custom_lm2, frame_3, gray_frame, face)
        tt_custom2 += time.time() - tik

        tik = time.time()
        annotate_landmarks(custom_lm3, frame_4, gray_frame, face)
        tt_custom3 += time.time() - tik

    text = f"default landmark predictor (99.7 MB),  avg_time: {round(tt_default/frame_count, 6)}"
    draw_text_with_backgroud(frame_1, text, 25, 25, font_scale=0.6, thickness=2)

    text = f"custom landmark predictor (6 MB),  avg_time: {round(tt_custom1/frame_count, 6)}"
    draw_text_with_backgroud(frame_2, text, 25, 25, font_scale=0.6, thickness=2)

    text = f"custom landmark predictor (265.4 MB),  avg_time: {round(tt_custom2/frame_count, 6)}"
    draw_text_with_backgroud(frame_3, text, 25, 25, font_scale=0.6, thickness=2)

    text = f"custom landmark predictor (331.9 MB),  avg_time: {round(tt_custom3/frame_count, 6)}"
    draw_text_with_backgroud(frame_4, text, 25, 25, font_scale=0.6, thickness=2)

    top = np.hstack((frame_1, frame_2))
    bottom = np.hstack((frame_3, frame_4))
    combined = np.vstack((top, bottom))
    cv2.imshow("facial landmarks detector", combined)

    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()
vidcap.release()
