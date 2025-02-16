import mediapipe
import cv2
from mediapipe.python.solutions.hands import Hands
from mediapipe.tasks.python.vision.face_stylizer import FaceStylizer

cap = cv2.VideoCapture(0)

mp_face_detection = mediapipe.solutions.face_detection
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_drawing = mediapipe.solutions.drawing_utils
mp_face_mesh = mediapipe.solutions.face_mesh
#mp_face_stylizer = mediapipe.tasks.vision.face_stylizer

mp_hands = mediapipe.solutions.hands

face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()
drawing = mp_drawing.DrawingSpec()
#face_stylizer = mp_face_stylizer.FaceStylizer()

def detectHumanFaceMesh(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)
    results = face_mesh.process(frame_rgb)
    #results_face_stylizer = face_stylizer._process_video_data('')
    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(frame_rgb, detection)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
             mp_drawing.draw_landmarks(frame_rgb,
                                       face_landmarks,
                                       mp_face_mesh.FACEMESH_CONTOURS,
                                       None,
                                       mp_drawing_styles.get_default_face_mesh_contours_style())
             mp_drawing.draw_landmarks(frame_rgb,
                                       face_landmarks,
                                       mp_face_mesh.FACEMESH_TESSELATION,
                                       None,
                                       mp_drawing_styles.get_default_face_mesh_tesselation_style())
    cv2.imshow('mediapipe', cv2.flip(frame_rgb, 1))

def detectHumanHands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_rgb,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      None,
                                      mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('Hands', cv2.flip(frame_rgb, 1))



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
    detectHumanFaceMesh(frame_rgb)
    detectHumanHands(frame_rgb)




cap.release()
cv2.destroyAllWindows()
