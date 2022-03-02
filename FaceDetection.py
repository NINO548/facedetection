import cv2
from random import randrange
  
# Load some pre-trained data on face frontals from opencv ( haar cascade algorithm )
trained_face_data = cv2.CascadeClassifier( cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in 
img = cv2.imread(r'C:\Users\NINO\Desktop\Python\AI\FaceDetection\ironman.jpg')


# Make it black-White( convert grayscale )
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


# Draw retangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange (256),randrange(256)), 10)



#print(face_coordinates)


# Display the image and name
cv2.imshow('Clever Promgrammer Face Detector',img)


# Wait to display
cv2.waitKey()


print("Code complete")
