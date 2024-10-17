import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/vusin/OneDrive/Documents/GitHub/CrowdDetection/DSC03218-Pano.jpg')

# Load the pre-trained Haar Cascade classifier for detecting people
person_cascade = cv2.CascadeClassifier('C:/Users/vusin/OneDrive/Documents/GitHub/CrowdDetection/haarcascade_frontalface_alt.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect people in the image using the Haar Cascade classifier
people = person_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the detected people
for (x, y, w, h) in people:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Print the number of people detected
print("Number of people detected:", len(people))

# Display the image with the detected people
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
