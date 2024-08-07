import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
import random
import time
# X Is the distance between hand land mark
# Y is the actual distance between my hand and camera
x = [172,151,120,95,84,76,69,61,55,52,46,44,42,43,40,37,30]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coeff = np.polyfit(x,y,2)


# Game Variable
cx,cy = 250,250
color = (255,0,255)
score = 0
target_hit = 0
start_time = time.time()
play_time = 5
def distance_measure(p1,p2):
    return int(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5)


def get_hand_distance(lm_distance):
    return np.polyval(coeff, lm_distance)
#Setup Camera

cam = cv2.VideoCapture(0)
# Hand Detector
detector = HandDetector(detectionCon=.8,maxHands=1)
while True:
    ret,frame = cam.read()
    frame = cv2.resize(frame,(0,0),None,0.75,0.75)

    if not ret:
        break

    if time.time()-start_time < play_time:
        hands,_ = detector.findHands(frame,draw=False)
        
        if hands:
            landmark_list = hands[0]["lmList"]
            x,y,w,h = hands[0]["bbox"]
            lm1 = landmark_list[5]
            lm2 = landmark_list[17]
            lm_distance = distance_measure(lm1,lm2)



            hand_distance = int(get_hand_distance(lm_distance))

            if hand_distance < 25 and x < cx < x+w and y < cy < y+h:
                target_hit = 1
            else:
                color = (255,0,255)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 255),3)
            cvzone.putTextRect(frame,f"{hand_distance} cm",(x,y-10))

        if target_hit:
            target_hit+=1
            color = (0,255,0)
            if target_hit == 3:
                cx = random.randint(50,900)
                cy = random.randint(50,500)
                score +=1
                target_hit = 0

        # Drawing The Target
        cv2.circle(frame,(cx,cy),30,color,-1)
        cv2.circle(frame,(cx,cy),10,(255, 255, 255 ),-1)
        cv2.circle(frame,(cx,cy),20,(255, 255, 255 ),2)
        cv2.circle(frame,(cx,cy),30,(50, 50, 50 ),1)

        # Drawing GUI
        cvzone.putTextRect(frame,f"Time: {play_time - (int(time.time()-start_time))}",(770,60),scale=2,offset=10)
        cvzone.putTextRect(frame,f"Score: {str(score).zfill(2)}",(35,60),scale=2,offset=10)
    else:
        cvzone.putTextRect(frame,"Game Over",(250,250),scale=5,offset=30,thickness=7)
        cvzone.putTextRect(frame,f"Score: {score}",(325,360),scale=4,offset=20,thickness=6)
        cvzone.putTextRect(frame,"Press R To Restart",(180,450),scale=4,offset=20,thickness=6)
    cv2.imshow("WebCam",frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord("r"):
        start_time = time.time()

cv2.destroyAllWindows()
cam.release()