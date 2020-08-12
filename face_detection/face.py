
import cv2
import os
import time

start = time.time()
# Start capturing video 
vid_cam = cv2.VideoCapture(0)
stop = time.time()

print(stop-start)
# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')   # use haar features
face_detector1 = cv2.CascadeClassifier('haarcascade_eye.xml')  


# Initialize sample face image


# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    faces1 = face_detector1.detectMultiScale(gray, 1.3, 5)
   
    

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
    
        # Display the video frame, with bounded rectangle on the person's face
        #cv2.imshow('frame', image_frame)

        for (sx,sy,sw,sh) in faces1:

            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (sx,sy), (sx+sw,sy+sh), (255,0,0), 2)
        
    
            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video


# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
