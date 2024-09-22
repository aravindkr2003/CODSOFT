import cv2
import dlib
import numpy as np

haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dnn_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, faces

def detect_faces_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            faces.append((startX, startY, endX - startX, endY - startY))
    return image, faces

def get_face_embedding(image, face):
    (x, y, w, h) = face
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(image, dlib_rect)
    face_embedding = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_embedding)

def recognize_face(new_image, known_faces, known_names, face):
    new_embedding = get_face_embedding(new_image, face)
    distances = [np.linalg.norm(new_embedding - known_face) for known_face in known_faces]
    min_distance = min(distances)
    if min_distance < 0.6:
        return known_names[distances.index(min_distance)]
    else:
        return "Unknown"

def main(image_path, use_haar=True):
    image = cv2.imread(image_path)
    cv2.imshow("Original Image", image)

    if use_haar:
        image_with_faces, faces = detect_faces_haar(image)
        print("Detected faces using Haar cascades.")
    else:
        image_with_faces, faces = detect_faces_dnn(image)
        print("Detected faces using DNN.")
   
    cv2.imshow("Detected Faces", image_with_faces)

    known_faces = []
    known_names = []

    if faces:
        known_face_embedding = get_face_embedding(image, faces[0])
        known_faces.append(known_face_embedding)
        known_names.append("Known Person")

    for face in faces:
        recognized_name = recognize_face(image, known_faces, known_names, face)
        print(f"Recognition result: {recognized_name}")
        (x, y, w, h) = face
        cv2.putText(image_with_faces, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Recognized Faces", image_with_faces)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "example.jpg"  
    main(image_path, use_haar=True)  