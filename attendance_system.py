import cv2
import face_recognition
from deepface import DeepFace
import gender_guesser.detector as gender_detector
import sqlite3

# Initialize gender detector
gender_detector = gender_detector.Detector()

# Connect to the database
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Create table for known faces
c.execute('''CREATE TABLE IF NOT EXISTS known_faces
             (name TEXT, encoding BLOB, emotion TEXT, age INTEGER, gender TEXT)''')

# Load known face images and their corresponding names
known_face_images = ["face1.jpg", "face2.jpg", "face3.jpg"]
known_face_names = ["John", "Jane", "Alex"]

# Initialize lists to store encodings, emotions, ages, and genders
known_face_encodings = []
known_face_emotions = []
known_face_ages = []
known_face_genders = []

# Load and encode known faces
for face_image, name in zip(known_face_images, known_face_names):
    image = face_recognition.load_image_file(face_image)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)

    # Perform emotion detection on known faces
    emotion = DeepFace.analyze(face_image, actions=['emotion'])['emotion']
    known_face_emotions.append(emotion)

    # Perform age estimation on known faces
    age = int(DeepFace.analyze(face_image, actions=['age'])['age'])
    known_face_ages.append(age)

    # Perform gender detection on known faces
    gender = gender_detector.get_gender(name.split()[0])[0]
    known_face_genders.append(gender)

    # Insert known face data into the database
    c.execute("INSERT INTO known_faces VALUES (?, ?, ?, ?, ?)",
              (name, face_encoding.tobytes(), emotion, age, gender))

# Commit the changes to the database
conn.commit()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
face_emotions = []
face_ages = []
face_genders = []

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and their encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize empty lists for names, emotions, ages, and genders
    face_names = []
    face_emotions = []
    face_ages = []
    face_genders = []

    for face_encoding in face_encodings:
        # Compare face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        emotion = "Unknown"
        age = "Unknown"
        gender = "Unknown"

        # Find the index of the matched face
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            emotion = known_face_emotions[match_index]
            age = known_face_ages[match_index]
            gender = known_face_genders[match_index]

        face_names.append(name)
        face_emotions.append(emotion)
        face_ages.append(age)
        face_genders.append(gender)

    # Display the results
    for (top, right, bottom, left), name, emotion, age, gender in zip(face_locations, face_names, face_emotions,
                                                                     face_ages, face_genders):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name, emotion, age, and gender below the face
        label = f"{name}, {emotion}, {age} years old, {gender}"
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
