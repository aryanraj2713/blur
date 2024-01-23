import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def plotImages(img): 
    plt.imshow(img, cmap="gray") 
    plt.axis('off') 
    plt.show()

path_to_image = str(input("Enter path to image: "))


image = cv2.imread(path_to_image) 
if image is None:
    print("Error: Unable to load the image. Please check the path.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plotImages(image)


face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
face_data = face_detect.detectMultiScale(image, 1.3, 5) 


for (x, y, w, h) in face_data: 
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    roi = image[y:y+h, x:x+w] 
    # Apply a Gaussian blur over the new rectangle area 
    roi = cv2.GaussianBlur(roi, (23, 23), 30) 
    # Impose the blurred image on the original image to get the final image 
    image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 


plotImages(image)
