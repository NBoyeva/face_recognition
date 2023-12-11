import cv2
import os
import numpy as np
from PIL import Image
import tkinter as tk
import pandas as pd


root = tk.Tk()
root.title(string = '')

frame = tk.LabelFrame(
    root,
    text = 'Cocos:>',
    bg = '#f0f0f0',
    font = (20)
)
frame.pack(expand = True, fill = 'both') 

root.geometry('400x500+550+130')

# global vars
#person_id = 0
#names = []
name = ''
dataset = pd.read_csv('dataframe.csv')
people_df = dataset.iloc[:,0:2]
people = pd.Series(people_df.name.values,index=people_df.id).to_dict()
try:
    person_id = max(dataset['id'].tolist())
except ValueError:
    person_id = 0



def introduce():

    def save_name():
        
        global name
        
	# retrieve the name from window and write it to variable
        s_name = the_name.get() 
        
        if s_name:
            name = s_name
            print("Retrieved name: ", name)
            
            # inform that name is retrieved
            done = tk.Label(intro, text = 'Done!')
            done.place(x=60,y=140)
            
        else:
            pass  
    
    def new_person():

        intro.destroy()

        ############### Adding new person to the dataset ###############

        global person_id, dataset, people
	
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print("\n [INFO] Look at the camera and wait, buddy ...")

        # Initialize individual sampling face count
        count = 0
        
        while(True):
            
            global dataset

            # get faces from the camera
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                img_numpy = np.array(cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2), 'uint8')


                # update dataset variable
                dataset = pd.concat([
                    dataset,
                    pd.DataFrame([[person_id, name, img_numpy]], columns=['id', 'name', 'image'])
                ],
                axis = 0)
                people = pd.Series(people_df.name.values,index=people_df.id).to_dict()
                dataset.to_csv('dataframe.csv', index=False)


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

        recognizer = cv2.face.LBPHFaceRecognizer.create() 
        
        images_list = dataset['image'].tolist()
        id_list = dataset['id'].tolist()

        recognizer.train(images_list, id_list)

        # Save the model into trainer/trainer.yml
        recognizer.save('trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


    intro = tk.Tk()
    intro.title('')
    intro.geometry('300x200+600+220')
    
    label_name = tk.Label(intro, text = 'Name')
    label_name.place(x = 60, y = 55)
    
    the_name = tk.Entry(intro)
    the_name.place(x = 120, y = 55)
    
    btn_save = tk.Button(intro, text = 'Save name', 
              command = save_name)
    print("Name: ", name)
    
    btn_take_photo = tk.Button(intro, text = 'Take photos',
              command = new_person)
    btn_save.place(x = 48,y = 110)
    btn_take_photo.place(x = 170,y = 110)

    

btn_intro = tk.Button(frame, text = "I'm a new user", 
                      command = introduce,
                      padx = 10, 
                      font = ('Arial', 14))
btn_intro.place(x = 120, y = 110)



def recognition(): 
    
    dataset = pd.read_csv('dataframe.csv')
    people_df = dataset.iloc[:,0:2]
    people = pd.Series(people_df.name.values,index=people_df.id).to_dict()
    try:
        person_id = max(dataset['id'].tolist())
    except ValueError:
        person_id = 0

    recognizer = cv2.face.LBPHFaceRecognizer.create() 
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

            # infer name from id
            person = people[id]

            certainty_threshold = 50

            if (confidence > certainty_threshold):
                printed_id = person
                #print(names)
                confidence = "  {0}%".format(round(confidence))
            else:
                printed_id = "unknown"
                confidence = "  {0}%".format(round(confidence))
            

            # Write the name of recognized person
            cv2.putText(
                img,
                str(printed_id),
                (x + 5, y - 5),
                font,
                1,
                (255, 255, 255),
                2
            )
            
            # Write how confident model is
            cv2.putText(
                img, 
                str(confidence), 
                (x + 5,y + h - 5), 
                font, 
                1, 
                (255, 255, 0), 
                1
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

root.mainloop()
