import face_recognition
print("Running face recognition")
image = face_recognition.load_image_file("images/obama.jpg")
face_locations = face_recognition.face_locations(image)
print(face_locations)