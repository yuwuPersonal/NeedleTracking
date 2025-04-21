#%% Import modules

import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

#%% Define utility funcs for bounding box and text annotation
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))
    cv.rectangle(frame, p1, p2, (255, 0, 0))

def displayRectangle(frame, bbox):
    frame_copy = frame.copy()
    drawRectangle(frame_copy, bbox)
    frame_copy = cv.cvtColor(frame_copy, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 10))
    plt.imshow(frame_copy)

def drawText(frame, txt, loc, color=(50,170,50)):
    cv.putText(frame, txt, loc, cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#%% Create tracker instance
tracker_types = ["BOOSTING", "MIL", "KCF", "CSRT", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE"]

tracker_type = tracker_types[3]

if tracker_type == "BOOSTING":
    tracker = cv.legacy.TrackerBoosting.create()
elif tracker_type == "MIL":
    tracker = cv.legacy.TrackerMIL.create()
elif tracker_type == "KCF":
    tracker = cv.TrackerKCF.create()
elif tracker_type == "CSRT":
    tracker = cv.TrackerCSRT.create()
elif tracker_type == "TLD":
    tracker = cv.legacy.TrackerTLD.create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv.legacy.TrackerMedianFlow.create()
elif tracker_type == "GOTURN":
    tracker = cv.TrackerGOTURN.create()
else:
    tracker = cv.legacy.TrackerMOSSE.create()

#%% Read input video and setup output video
video = cv.VideoCapture("needle_sample.mp4")
ok, frame = video.read()

# exit if video is closed 
if not video.isOpened():
    print("Could not open the video")
    sys.exit()
else:
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

video_out_filename = "needle_sample_{}.mp4".format(tracker_type)
video_out = cv.VideoWriter(video_out_filename, cv.VideoWriter_fourcc(*"XVID"), 10, (width, height))

#%% Define bounding box
n_boxes = 4
bboxes = []
for n in range(n_boxes):
    box = cv.selectROI(frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    bboxes.append(box)

# bbox = (46, 162, 22, 28)
# win = cv.namedWindow("first_frame")
# bbox = cv.selectROI(win, frame)

frame_copy = frame.copy()
for bbox in bboxes:
    drawRectangle(frame_copy, bbox)
plt.imshow(frame_copy)

#%% create and initiate multitracker object
multi_tracker = cv.legacy_MultiTracker.create()

for n, bbox in enumerate(bboxes):
    multi_tracker.add(cv.legacy_TrackerCSRT.create(), frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bboxes_pred = multi_tracker.update(frame)

    if ok:
        for bbox in bboxes_pred:
            drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracker failure detected", (80, 140), (0, 0, 255))
    
    video_out.write(frame)

video.release()
video_out.release()



#%% Initialize the tracker with first frame and bbox
ok = tracker.init(frame, bbox)
print(ok)

#%% Read frame and track object
while True:
    ok, frame = video.read()
    if not ok:
        break
    
    # start the timer
    timer = cv.getTickCount()
    # update the tracker, asking it to predict current bbox
    ok, bbox = tracker.update(frame)
    # calculate the fps
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

    # draw bounding box and text
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracker failure detected", (80, 140), (0, 0, 255))

    drawText(frame, tracker_type + " Tracker", (80, 60))
    drawText(frame, "FPS: " + str(int(fps)), (80, 100))

    # write frame to the output video
    video_out.write(frame)

video.release()
video_out.release()





# %%
