import cv2
import os
import numpy as np
from PIL import Image
import tkinter as tk


root = tk.Tk() # main window
#root.title(string = '') # need to fix

frame = tk.LabelFrame(
    root, # the frame is on the root (main window)
    text = 'Cocos:>',
    bg = '#f0f0f0', # color of the frame
    font = (20)
)
frame.pack(expand = True, fill = 'both')  # shiw the frame, expand = True -- maximal size
root.geometry('400x500+550+130') # size (lxw) and shift x and y

# global vars
person_id = 0
names = []
name = '' #


def introduce():

    def save_name():
        
        global name

        s_name = the_name.get()
        if s_name:
            name = s_name
            print("Name ddadadada", name)
            done = tk.Label(intro, text = 'Done!')
            done.place(x=60,y=140)
        else:
            pass  
    
    def new_person():

        intro.destroy()

        ########## Adding new person to the dataset ###############

        global person_id

        face_id = person_id

        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print("\n [INFO] Look at the camera and wait, buddy ...")

        # Initialize individual sampling face count
        count = 0
        
        while(True):

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1

                # Save photo to the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' +
                            str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('Look at the camera. Not here', img)
                cv2.moveWindow('Look at the camera. Not here', 635, 200)
            
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break
        
        cam.release()
        cv2.destroyAllWindows()
    
        person_id += 1

        ########### Model training ###############

        print ("\n [INFO] I'm trying to memorize you. Wait ...")

        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = face_detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)

            return faceSamples,ids

        
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.save('trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        #print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


    intro = tk.Tk() # window for new person (namee+photos)
    intro.title('')
    intro.geometry('300x200+600+220')
    label_name = tk.Label(intro, text = 'Name')
    label_name.place(x=60,y=55)
    the_name = tk.Entry(intro) # window to enter the word
    the_name.place(x=120,y=55) # place the window

    #the_name = input('\n Your name pleassss ==>  ')
    
    btn_save = tk.Button(intro, text = 'Save name', 
              command = save_name)
    print("Name: ", name)
    
    btn_take_photo = tk.Button(intro, text = 'Take photos',
              command = new_person)
    btn_save.place(x=48,y=110)
    btn_take_photo.place(x=170,y=110)

    

btn_intro = tk.Button(frame, text = "I'm a new user", 
                      command = introduce,
                      padx = 10, 
                      font = ('Arial', 14))
btn_intro.place(x=120, y=110)



def recognition():

    global names, name
    names.append(name)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    id = 0 # id counter initiator
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100):
                id = names[id]
                print(names)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            

            # Write the name of recognized person
            cv2.putText(
                img,
                str(id),
                (x + 5, y - 5),
                font,
                1,
                (255, 255, 255),
                2
            )
              
            
        cv2.imshow('camera', img)
        cv2.moveWindow('camera', 635, 200)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()


btn_recogn = tk.Button(frame, text = "Recognize", 
                      command = recognition,
                      padx = 10, 
                      font = ('Arial', 14))
btn_recogn.place(x=135, y=200)

root.mainloop() # show main window
