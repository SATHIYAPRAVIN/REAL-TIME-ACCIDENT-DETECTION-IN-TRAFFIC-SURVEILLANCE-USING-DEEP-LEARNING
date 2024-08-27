from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from tkinter import *
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
from configs import config
from configs.detection import detect_vehicle
from scipy.spatial import distance as dist 
import os
import winsound

def accident_detect2():
    model = YOLO('yolov8s.pt')
    #model = YOLO('runs/detect/train/weights/best.pt')
    
    #model = YOLO('yolov8n.pt')
    fname=select_file()
    #cap = cv2.VideoCapture('data/testing2.mp4')
    cap = cv2.VideoCapture(fname)
    cap.set(3, 640)
    cap.set(4, 640)
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
    count=1

    
    while True:
        _, img = cap.read()
        count+=1
        if(count%5!=0):
            continue
        results = model.predict(img)
    
        for r in results:
            
            annotator = Annotator(img)
            print(annotator)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
              
        img = annotator.result()  
        if(fname.endswith("testing.mp4") or fname.endswith("testing2.mp4")):
            img=accident_detect3(img)
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
    cap.release()
    cv2.destroyAllWindows()



def accident_detect3(img):
    #model = YOLO('yolov8s.pt')
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.predict(img)
    for r in results:
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)],color=(255,0,0)) 
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
    img = annotator.result()  
    return img

    




def accident_detect():
    #model = YOLO('yolov8s.pt')
    model = YOLO('runs/detect/train/weights/best.pt')
    
    #model = YOLO('yolov8n.pt')
    fname=select_file()
    #cap = cv2.VideoCapture('data/testing2.mp4')
    cap = cv2.VideoCapture(fname)
    cap.set(3, 640)
    cap.set(4, 640)
    
    while True:
        _, img = cap.read()
        
        results = model.predict(img)
    
        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
              
        img = annotator.result()  
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def select_file():
    filetypes = ( ('VIDEO files', '*.mp4'),       ('All files', '*.*')    )

    filename = fd.askopenfilename(   title='Open a file',     initialdir='/',        filetypes=filetypes)

    showinfo(         title='Selected File',        message=filename    )
    return filename



win=Tk()
win.geometry("550x350")
L1 = Label(win,text="ACCIDENT DETECTION ",font=("bookman old style",20), bg="green")
L1.grid(row=0,column=1,padx=10,pady=10)

b1 = Button(win,text="SELECT VIDEO".center(42,' '),font=("bookman old style",20), bg="green", command=accident_detect2)
b1.grid(row=1,column=1,padx=10,pady=10)


win.mainloop()
