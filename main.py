import mediapipe
import cv2


cap = cv2.VideoCapture(0)

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_drawing = mediapipe.solutions.drawing_utils
mp_face_mesh = mediapipe.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh()
drawing = mp_drawing.DrawingSpec()
GREEN_COLOR = (0, 128, 0)

def detectHumanHeadFaceMesh(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results = face_mesh.process(frame_rgb)
    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(frame_rgb, detection)

        for face_landmarks in results.multi_face_landmarks:
             mp_drawing.draw_landmarks(frame_rgb, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,None, mp_drawing_styles.get_default_face_mesh_contours_style())
             mp_drawing.draw_landmarks(frame_rgb, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())

    cv2.imshow('mediapipe', cv2.flip(frame_rgb, 1))

while True:
    ret, frame = cap.read()
    #cv2.imshow('DetectHuman', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

    #uncomment later
    #if(detectHumanHead(frame)):
     #   print("Face is Show")
   # else:
    #    print("Face is not shown")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detectHumanHeadFaceMesh(frame_rgb)



cap.release()
cv2.destroyAllWindows()