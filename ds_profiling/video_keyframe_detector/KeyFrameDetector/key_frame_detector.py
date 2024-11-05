import os
import cv2
import csv
import numpy as np
import time
import peakutils
from .utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    
    keyframePath = os.path.join(dest, 'keyFrames')
    imageGridsPath = os.path.join(dest, 'imageGrids')
    outPath = os.path.join(dest, 'outFile')
    path2file = os.path.join(outPath, 'output.txt')
    prepare_dirs(keyframePath, imageGridsPath, outPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    cap.release()
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)
    
    ##plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(indices, lstfrm, lstdiffMag, dest)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.\n'
        if(verbose):
            print(log_message)
        with open(path2file, 'a') as outFile:
            # writer = csv.writer(csvFile)
            outFile.write(log_message)
            # csvFile.close()
        cnt +=1

    cv2.destroyAllWindows()
