##################################################
##    Using of Emotion Recognition Technique 
##        to Evaluate Teaching Strategies
##################################################
## {BSD}
##################################################
## Author: {Mohamed Achraf BEN MOHAMED}
## Copyright: Copyright {year}, {project_name}
## Credits: []
## License: {}
## Version: {1}.{0}.{0}
## Mmaintainer: {Mohamed Achraf BEN MOHAMED}
## Email: {mohamedachraf@gmail.com}
## Status: {4 - Beta}
##################################################

#---------------------------------------------------
print ("*** STARTING SCRIPT EXECUTION ***")

print ("[INFO] PREVENTING WARNING MESSAGE...", end='')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print ("Done!")
#---------------------------------------------------

print ("[INFO] LOADING LIBRARIES..")
import time
print("\t\t time")
time.sleep(0.3)

from keras.preprocessing.image import img_to_array
print("\t\t Keras")
time.sleep(0.3)

import imutils
print("\t\t imutils")
time.sleep(0.3)

import cv2
print("\t\t openCV")
from keras.models import load_model
time.sleep(0.3)

import numpy as np
print("\t\t numpy")
time.sleep(0.3)

import matplotlib.pyplot as plt
print("\t\t matplotlib")
time.sleep(0.3)

from datetime import datetime
print("\t\t datetime")
time.sleep(0.3)

import arabic_reshaper
print("\t\t arabic_reshaper")
time.sleep(0.3)

from bidi.algorithm import get_display
print("\t\t bidi")
time.sleep(0.3)

import matplotlib.ticker as ticker

print ("\t Done!")

#==============================
print ("[INFO] INIT WINDOW PARAMETERS..")
# parameters
Windows_width = 350
Windows_height = 250
Frame_duration = 1
display_legend = True

print("\t\t Windows width  = ", Windows_width  )
print("\t\t Windows height = ", Windows_height )
print("\t\t Frame duration = ", Frame_duration )
print("\t\t Display legend = ", display_legend )
#==============================
def max_of_two( x, y ):
    if x > y:
        return x
    return y
def max_of_three( x, y, z ):
    return max_of_two( x, max_of_two( y, z ) )

def max_emotions( x, y, z ):
    if (max_of_two( x, max_of_two( y, z ) ) == x):
        return 1
    
    if (max_of_two( x, max_of_two( y, z ) ) == y):
        return 2
    
    if (max_of_two( x, max_of_two( y, z ) ) == z):
        return 3
#==============================

print ("[INFO] INIT COLORS CODES..", end='')
#---------------------------------------------------
#indicates color for plots RED GREEN YELLOW
#pal = ["#76b729", "#fbb901",  "#e40228"]
pal = ["#22d69c", "#999999",  "#f15a24"]
#---------------------------------------------------
# indicates list of colors for every emotions
emotions_colors = [
    (  0,   0, 255), # angry
    (130,   0,  75), # disgust 
    (255,   0,   0), # scared
    (  0, 255,   0), # happy
    (255,   0, 143), # sad
    (255, 255,   0), # surprised
    (211, 201, 206)  # neutral
]
print ("Done!")

#---------------------------------------------------
# uses style 
print ("[INFO] INIT GGPLOT STYLE..", end='')
plt.style.use('ggplot')
print ("Done!")


#---------------------------------------------------


#---------------------------------------------------
print ("[INFO] LOADING MODELS..")
# hyper-parameters for bounding boxes shape
# loading models
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'


face_detection = cv2.CascadeClassifier(detection_model_path)
print("\t\t Face Detection Model    : haarcascade_frontalface_default ")


emotion_classifier = load_model(emotion_model_path, compile=False)
print("\t\t Emotion Detection Model : _mini_XCEPTION.102-0.66.hdf5 ")

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
print ("\t Done!")

#---------------------------------------------------
print ("[INFO] INITIALISE LISTS FOR GATHERING DATA..", end='')
# initialise lists for gathering data
angry = []
disgust = []
scared = []
happy = []
sad = []
surprised = []

neutral = []
negative = []
positive = []

ph_ = []
ps_ = []
pn_ = []
psc_ = []

axes = plt.gca()
ax = plt.axes()
ax.title.set_text('First Plot')
print ("Done!")

#---------------------------------------------------
ph = 0
ps= 0
pn = 0
psc= 0
ind = 0

#---------------------------------------------------
print ("[INFO] STARTING VIDEO STREAMING..")
# starting video streaming
CAM_SOURCE = 1
camera = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_DSHOW)
print("\t\t Camera Source : ", CAM_SOURCE)
print ("\t Done!")
                
#---------------------------------------------------
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=Windows_width, height = Windows_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    # init histograms zone
    canvas = np.zeros((Windows_height , Windows_width, 3), dtype="uint8")
    
    # draw plain rectangle as background in histograms zone
    cv2.rectangle(canvas, (0,0),(Windows_width,Windows_height),(140,140,140),-1)
    
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)        
        label = EMOTIONS[preds.argmax()]
    else: continue
    
    j = 0
    line_widh = 0.35
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

               # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]
                #########################################
                #########################################  
                #---------------------------------------------------
                # recuperate emotions      
                if(ind == 0) :
                    angry.append(prob*100)                    
                    
                if(ind == 1) :
                    disgust.append(prob*100)
                    
                if(ind == 2) :
                    scared.append(prob*100)
                    
                if(ind == 3) :
                    happy.append(prob*100)
                    #---------------------------------------------------
                    # recuperate the number of collected emotions for stackplot parameters
                    #ph += 1
                    #ph_.append(ph)
                    
                    time_now = datetime.now().strftime('%H:%M:%S')
                    
                    ph_.append(time_now)
                    
                if(ind == 4) :
                    sad.append(prob*100)                   
                    
                if(ind == 5) :
                    surprised.append(prob*100)
                    
                if(ind == 6) :
                    neutral.append(prob*100)
                    
                
                if (ind == 6): # all emotions are collected
                    #---------------------------------------------------
                    # negative = sad + disgust + angry + scared
                    negative = [x + y for x, y in zip(sad, [x + y for x, y in zip(disgust, [x + y for x, y in zip(angry, scared)])])]
                    
                    #---------------------------------------------------
                    # negative = happy + surprised
                    positive = [x + y for x, y in zip(happy, surprised)]
                    #---------------------------------------------------
                    # draw plot : positive,neutral and negative
                    plt.stackplot(ph_, positive,neutral, negative,  colors=pal, alpha=0.4 )   
                    #plt.xticks(rotation=45)
                    
                   
                    #---------------------------------------------------
                    # Prevent x-tick labels from overlapping                     
                    ax.xaxis.set_major_locator(ticker.AutoLocator())
                   
                    
                    positve_label =  get_display( arabic_reshaper.reshape('إيجابي'))  
                    neutral_label =  get_display( arabic_reshaper.reshape('محايد'))
                    negative_label =  get_display( arabic_reshaper.reshape('سلبي'))                    
                    plt.plot([],[], label=positve_label, color=pal[0], marker='s', linestyle='None',markersize=10)
                    plt.plot([],[], label=neutral_label, color=pal[1], marker='s', linestyle='None',markersize=10)
                    plt.plot([],[], label=negative_label,color=pal[2], marker='s', linestyle='None',markersize=10)

                    #---------------------------------------------------
                    # Prevent legend to be displayed every iteration.
                    if(display_legend == False) :
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=3)
                        display_legend = True

                    #---------------------------------------------------
                    # Display title      
                    title_label =  get_display( arabic_reshaper.reshape('تحليل المشاعر في الوقت الحقيقي')) 
                    plt.title(title_label)
                    
                    #---------------------------------------------------
                    # Display x and y axis legend                    
                    X_R = get_display( arabic_reshaper.reshape('التوقيت الزمني'))
                    Y_R = get_display( arabic_reshaper.reshape('تحليل المشاعر'))
                    
                    
                    plt.xlabel(X_R, fontdict=None, labelpad=None)
                    plt.ylabel(Y_R, fontdict=None, labelpad=None)
                        
                    plt.ylim(0, 100) 
                    
                    #---------------------------------------------------
                    # Display rectangle with percentages                     
                    cv2.rectangle(frameClone, (fX - 118, fY- 69), (fX-20 , fY-20), (0, 255, 255),-1)
                    percentage_pos = str ("{0:.2f}".format(positive[len(positive) -1]) )+"% Positive" 
                    percentage_neu = str ("{0:.2f}".format(neutral [len(neutral ) -1]) )+"% Neutral" 
                    percentage_neg = str ("{0:.2f}".format(negative[len(negative) -1]) )+"% Negative" 
                    
                    cv2.line(frameClone, (fX - 20, fY- 20), (fX, fY), (0, 255, 255), 2)
                    
                    cv2.putText(frameClone, percentage_pos,(fX - 115, fY-55),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
                    cv2.putText(frameClone, percentage_neu,(fX - 115, fY-40),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
                    cv2.putText(frameClone, percentage_neg,(fX - 115, fY-25),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)

               
                #---------------------------------------------------
                #re initialize variable ind for next collection (emotions are collected 6 by 6)
                ind += 1
                if(ind == 7):
                    ind = 0
               
                
                #---------------------------------------------------#
                #                 Camera output                     #
                #---------------------------------------------------#
                # Draw rectangle around face and display detected emotion
                w = int(prob * 300)                
                #cv2.rectangle(canvas, (4, (j * 35) + 5), (w, (j * 35) + 35), (94.5, 35.3, 14.1), -1)             
                cv2.rectangle(canvas, (4, (j * 35) + 5), (w, (j * 35) + 35), emotions_colors[ind-1], -1)
                cv2.putText(canvas, text, (10, (j * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)          
                #cv2.putText(frameClone, label, (fX, fY - 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                #=======================
                
                #=======================
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 255), 2)
                
                
                j+=1


    plt.pause(Frame_duration)
    
    
    #--------------------------------------  
    if (max_emotions( percentage_pos, percentage_neu, percentage_neg ) == 1):
        smiley = cv2.imread('./emojis/AINC_happy.png',1) 
        
    if (max_emotions( percentage_pos, percentage_neu, percentage_neg ) == 2):
        smiley = cv2.imread('./emojis/AINC_happy.png',1) 
        
    if (max_emotions( percentage_pos, percentage_neu, percentage_neg ) == 3):
        smiley = cv2.imread('./emojis/AINC_happy.png',1) 
    
    width = Windows_width
    height = 111
    dim = (width, height)
 
    # resize image
    resized_smiley = cv2.resize(smiley, dim, interpolation = cv2.INTER_AREA)
    
    #--------------------------------------
    # separator : white line
    separator = cv2.imread('./emojis/separator.png',1) 
    dim1 = (width, 7)
    resized_separator = cv2.resize(separator, dim1, interpolation = cv2.INTER_AREA)
    
    
    #concatenete windows of camera and emotions histograms
    All_screens = np.concatenate([resized_smiley, frameClone, resized_separator, canvas], axis=0)
    #both = np.concatenate([frameClone, canvas], axis=0)
    cv2.imshow('Artificial Intelligence National Competition',All_screens)
    
    #--------------------------------------  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #--------------------------------------



camera.release()
cv2.destroyAllWindows()
