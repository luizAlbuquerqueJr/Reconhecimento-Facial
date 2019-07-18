''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os
import time



f = open('dataset/dataSet.txt','r')
lines = f.readlines()

f.close()

face_id = 0
if(len(lines) == 0):
    print("Arquivo vazio")
else:
    lastLines = lines[-1]
    print(lastLines)
    face_id = int(lastLines.split("@")[0]) + 1





cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
# face_id = input('\n Digite seu ID:')
face_nome = input('\n Digite o nome da pessoa fotografada:  ')

f = open('dataset/dataSet.txt','a+')
f.write("\n" + str(face_id) + "@"+ face_nome)
f.close()



print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    time.sleep(1) 
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        print("Data")
        cv2.imwrite("dataset/images/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('image', img)

    k = cv2.waitKey(20) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


