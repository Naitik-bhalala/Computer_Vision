import cv2 
import numpy as np 
import time 
  

print(cv2.__version__)    
  
capture_video = cv2.VideoCapture(0) 
     

time.sleep(1)  
background = 0 
  

for i in range(60): 
    ret, background = capture_video.read() 
    
  
 
  
# we are reading from video  
while (capture_video.isOpened()): 
    return_val, img = capture_video.read() 
    if not return_val : 
        break 
   
  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
   
    lower_red = np.array([0, 120, 70])        
    upper_red = np.array([10, 255, 255]) 
    mask1 = cv2.inRange(hsv, lower_red, upper_red) 
    # setting the lower and upper range for mask2  
    lower_red = np.array([170, 120, 70]) 
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red) 
    #----------------------------------------------------------------------# 
  
    # the above block of code could be replaced with 
    # some other code depending upon the color of your cloth  
    mask1 = mask1 + mask2 
  
    # Refining the mask corresponding to the detected red color 
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), 
                                         np.uint8), iterations = 2) 
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 1) 
    mask2 = cv2.bitwise_not(mask1) 
  
    # Generating the final output 
    res1 = cv2.bitwise_and(background, background, mask = mask1) 
    res2 = cv2.bitwise_and(img, img, mask = mask2) 
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) 
  
    cv2.imshow("INVISIBLE MAN", final_output) 
    k = cv2.waitKey(10) 
    if k == 27: 
        break

capture_video.release()
cv2.destroyAllWindows()