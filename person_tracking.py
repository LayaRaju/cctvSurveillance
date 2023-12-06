import cv2 # opencv for image processing
import datetime
import imutils # image processing
import numpy as np
from centroidtracker import CentroidTracker
from nms import non_max_suppression_fast
from collections import defaultdict
protopath = "MobileNetSSD_deploy.prototxt" # deep learning ---> for person detection
modelpath = "MobileNetSSD_deploy.caffemodel"# like tensorflow , caffe is deep learning cnn library
import pandas as pd
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
person_entering={}
person_in_id=[]
lock=0
time=0
lock1=False
temp=[]
tmp=[]
elapsed_dict=defaultdict(list)
temp_tup=()
object_id_list = []
dtime = dict()
dwell_time = dict()
my_dict = {"Id":[],"Time":[],"Elapsed_time":[]}
def main():
    cap = cv2.VideoCapture('test_video.mp4')
    #cap = cv2.VideoCapture(0)
    global my_dict
    result=0  
    #fps_start_time = datetime.datetime.now()
    #fps = 0
    #total_frames = 0

    while True:
        ret, frame = cap.read() # read frame by frame from video or camera
        frame = imutils.resize(frame, width=600) #image resize to width=600
        #total_frames = total_frames + 1
        area = [(27,334), (27,214), (153,214), (154,334)]
	#area_2 = [(650, 575), (862, 549), (883, 392), (725, 375)]
	#area_3 = [(648, 343), (596, 213), (437, 218), (324, 352)]
	#for area in [area_1]:
        cv2.polylines(frame, [np.array(area, np.int32)], True, (15,220,10), 2)
        #print(frame.shape)
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
                
                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            #person_in_id.append(objectId)
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cx=int((x2+x1)/2)
            cy=int((y2+y1)/2)
            #print(cx,cy)
            color = (0, 0, 255)
            cv2.circle(frame,(cx,cy),5,color,-1)
            result = cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False)
            if(result>=0):
               #lock
           
               if(objectId not in person_in_id):
                 #print("Success")
                 #person_entering[objectId]=(cx,cy)
                 person_in_id.append(objectId)
                 #print(person_in_id)
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 #object_id_list.append(objectId)
                 now=datetime.datetime.now()
                 dtime[objectId] = datetime.datetime.now()
                 dwell_time[objectId] = 0
                 lock=0
                 #tmp.append(0)
                 time = now.strftime("%y-%m-%d %H:%M:%S")
                 #print(type(objectId))
                 my_dict["Id"].append((str(objectId)))
                 my_dict["Time"].append(str(time))
               else:  
                 curr_time = datetime.datetime.now()
                 old_time = dtime[objectId]
                 time_diff = curr_time - old_time
                 dtime[objectId] = datetime.datetime.now()
                 sec = time_diff.total_seconds()
                 dwell_time[objectId] += sec
                 if(int(dwell_time[objectId])>10 and lock==0):
                    print('Anomaly')
                    lock=1
                 if(int(dwell_time[objectId])<0 and lock==1):
                    lock=0
                 elapsed_dict[objectId].append(int(dwell_time[objectId]))
                
                 tmp.append(int(dwell_time[objectId]))
                
                 #print(int(dwell_time[objectId]))
                 #print(type(objectId))
                 #temp.append(tmp)
                 #tmp.pop(0)
                 temp.insert(int(objectId),tmp[-1])
            else:
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        #fps_end_time = datetime.datetime.now()
        #time_diff = fps_end_time - fps_start_time
        #if time_diff.seconds == 0:
        #    fps = 0.0
        #else:
        #    fps = (total_frames / time_diff.seconds)

        #fps_text = "FPS: {:.2f}".format(fps)

        #cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            #print((dict(elapsed_dict)))
            mydict=dict(elapsed_dict)
            #print(mydict[1][-1])
            #print(my_dict)
            tmp_list=[mydict[x][-1] for x in mydict.keys()]
            my_dict={"Id":my_dict["Id"],"Time":my_dict["Time"],"Elapsed_time":tmp_list}
            print(my_dict)
            df=pd.DataFrame.from_dict(my_dict)
            #for k,v in dict(elapsed_dict): 
            #     dict[v]=max(dict[v])     
            df.to_csv('person_tracking.csv', index=False) 
            
            break

    cv2.destroyAllWindows()


main()


