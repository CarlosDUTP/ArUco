#   Carlos D. Gonzalez F.     20-70-5162      1IL141
import cv2
import numpy as np

#   Configuraciones de los detectores
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()

#   Procesos para los filtros
def blur_face(face):
    return cv2.GaussianBlur(face, (99, 99), 30)

def cartoonize_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(face, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        marker_id = None
        if ids is not None:
            marker_id = ids[0][0]
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected ID: {ids[0][0]}")

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            if marker_id == 341:
                frame[y:y+h, x:x+w] = blur_face(face)
            elif marker_id == 945:
                frame[y:y+h, x:x+w] = cartoonize_face(face)
            elif marker_id == 5:
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                bw_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
                frame[y:y+h, x:x+w] = bw_face

        #   Comentario Arriba Izquierda
        cv2.putText(frame, "Carlos Gonzalez  20-70-5162", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ArUco', frame)
        
        #   Guardar Imagen
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("output.jpg", frame)
        #   Salir del Codigo
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video('Video.mp4')