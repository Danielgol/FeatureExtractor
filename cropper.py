import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def crop_face(image):

  with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      return np.zeros((224,224,3), np.uint8)

    annotated_image = image.copy()
    for detection in results.detections:
      ymin = math.floor(detection.location_data.relative_bounding_box.ymin * 224)
      xmin = math.floor(detection.location_data.relative_bounding_box.xmin * 224)
      width = math.floor(detection.location_data.relative_bounding_box.width * 224)
      height = math.floor(detection.location_data.relative_bounding_box.height * 224)

      crop_img = annotated_image[ymin:ymin+height, xmin:xmin+width]
      return crop_img