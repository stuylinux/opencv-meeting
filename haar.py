import cv2

cap = cv2.VideoCapture(0)

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_detected = False

while True:
    ret, frame = cap.read()

    #convert to grayscale to run face recog on
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #run for face recog
    #for evry face found it will add it to the faces array
    #multiscale use to find multiple faces
    #compares grauy to sample imf in xml
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    print("Found {0} faces!".format(len(faces)))


    # Draw a rectangle around the faces
    biggest_face_img = ""

    # system to save the biggest detected face
    biggest_face_area = 0
    for (x, y, w, h) in faces:
        area_of_face= w*h
        if area_of_face > biggest_face_area:
            biggest_face_area = area_of_face
            biggest_face_img = frame[y:y+h,x:x+h]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 255, 0),3)
        

    if biggest_face_img != "":
        cv2.imshow("Locked on biggest face" ,biggest_face_img)
    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    #displaying the curent frame with face outlined
    cv2.imshow('Face recognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

#end recording and close windows
cap.release()
cv2.destroyAllWindows()
