import cv2

# Face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab Webcam feed
webcam = cv2.VideoCapture(0) #why_so_serious.mp4

while True:

    #Read current frame webcam
    successful_frame_read, frame = webcam.read()

    #if there is an error,abort
    if not successful_frame_read:
        break

    #change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces first
    face = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)

    #run smile detection within each of those faces
    for (x, y, w, h) in face:

        #Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        #create the face sub-image (opencv allows you to subindex like tjis. it is built on numpy. slice a n-dimensions array)
        face = frame[y:y+h, x:x+w]
        
        #grayscale the face
        frame_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #detect smiles in the face
        smile = smile_detector.detectMultiScale(frame_grayscale, 1.7, 20)
    




        #label this face as smiling
        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    #show the current frame
    cv2.imshow ('Why So Serious', frame)

    #Stop id Q key is pressed
    key = cv2.waitKey(1)
    if  key==81 or key==113:
        break
    
#clean up
webcam.release()
cv2.destroyAllWindows()

#Kaushalk844

