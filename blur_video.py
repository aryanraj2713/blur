import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def plotImages(img): 
    plt.imshow(img, cmap="gray") 
    plt.axis('off') 
    plt.show()

# Path to the video file
path_to_video = str(input("Enter path to video: "))

# Open the video file
cap = cv2.VideoCapture(path_to_video)


if not cap.isOpened():
    print("Error: Unable to open the video file. Please check the path.")
    exit()

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video_with_blur.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through frames
while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
    face_data = face_detect.detectMultiScale(frame, 1.3, 5) 

    for (x, y, w, h) in face_data: 
        roi = frame[y:y+h, x:x+w] 
        roi = cv2.GaussianBlur(roi, (23, 23), 30) 
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved at {output_path}")
