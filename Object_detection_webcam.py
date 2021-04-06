######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import win32api, win32con
import time
import _thread

from keyboard import Keyboard


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def get_x(x1,x2,class_id):
    pos_x=x1*2
    #X range : [0.0,1.0]
    pos_x = 1920 - int(1920 * (pos_x - 0) / 1.0)

    if pos_x < 0     :  pos_x = 0
    elif pos_x > 1920:  pos_x = 1920
    return pos_x

def get_y(y1,y2):
    pos_y= y1*2
    # Y range : [0.5,1.3]
    pos_y = int(1080 * (pos_y - 0.5) / 0.7)
    if pos_y < 0     :  pos_y = 0
    elif pos_y > 1080:  pos_y = 1080
    return pos_y

def move(last_pos_x, last_pos_y, pos_x, pos_y):
    diffx= pos_x - last_pos_x
    diffy= pos_y - last_pos_y
    for i in range(1,35):
        xxx=int(last_pos_x + i * diffx / 35)
        yyy=int(last_pos_y + i * diffy / 35)
        win32api.SetCursorPos((xxx,yyy))
        time.sleep(0.005)
    return 0

def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(x),int(y),0,0)

def click_up():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(x),int(y),0,0)

def volume_up(vol):
    for i in range(vol):
        Keyboard.key(Keyboard.VK_VOLUME_UP)

def volume_down(vol):
    for i in range(vol):
        Keyboard.key(Keyboard.VK_VOLUME_DOWN)

def page_up():
    Keyboard.key(Keyboard.VK_PRIOR)

def page_down():
    Keyboard.key(Keyboard.VK_NEXT)

def next_window():
    Keyboard.keyDown(Keyboard.VK_ALT)
    Keyboard.key(Keyboard.VK_TAB)
    Keyboard.keyUp(Keyboard.VK_ALT)

def desktop():
    Keyboard.keyDown(Keyboard.VK_LWIN)
    Keyboard.key(Keyboard.VK_D)
    Keyboard.keyUp(Keyboard.VK_LWIN)
def volume_mute():
    Keyboard.key(Keyboard.VK_VOLUME_MUTE)


video = cv2.VideoCapture(0)
video.set(3,1920)
video.set(4,1080)

lx=ly=x=y=0

cursor_x=cursor_y=0

up=down=0;

while True:
    s=time.time()
    clicking=False

    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    if np.squeeze(scores)[0]>0.9 :

        class_id=np.squeeze(classes)[0]
        x=get_x(np.squeeze(boxes)[0][1],np.squeeze(boxes)[0][3],class_id)
        y=get_y(np.squeeze(boxes)[0][0],np.squeeze(boxes)[0][2])

        print("("+str(x)+","+str(y)+")")

        if class_id==2:
            click()
            click_up()
            clicking=True

        elif class_id==3:
            if x<640:
                if y>540:
                    volume_down(down)
                    down+=1
                else:
                    volume_up(up)
                    up+=1
            elif x<1280:
                if y<540:
                    desktop()
                else:
                    print("2")
            else:
                if y<540:
                    page_up()
                else:
                    page_down()

        try:
            #if class_id==1 and not clicking:_thread.start_new_thread(move, (2*lx-1920, 2*ly-1080, 2*x-1920, 2*y-1080,))
            if class_id==1 and not clicking:_thread.start_new_thread(move, (lx, ly, x, y,))

        except:
            print ("Error: unable to start thread")
    else:
        up=down=0

    lx=x
    ly=y

    if cv2.waitKey(1) == ord('q'):
        break
    print(time.time()-s)
video.release()
cv2.destroyAllWindows()




