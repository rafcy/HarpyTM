# import the necessary packages
from threading import Thread
import sys
import cv2
import time
from  src.detector import *
# import the Queue class from Python 3
from queue import Queue
# Segmentation imports


class VideoStream:
    def __init__(self, video, transform=None, queue_size=1024,detectNet=None,resize=True,size=(512,512),frameid=0,cars=None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = video
        self.stopped = False
        self.transform = transform
        self.detectNet=detectNet
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.resize=resize
        self.size=size
        self.frameid = frameid
        self.mask = None
        self.cars = cars
    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True
                else:
                    test = time.time()

                    image_cp = frame.copy()
                    frame = cv2.resize(frame, (self.size[0], self.size[1])) if self.resize else frame
                    frame2   = frame.copy()

                    dets = self.detectNet.detect(frame2)

                    delay = time.time()-test
                    # print("Detection time: %.4f"%(delay))
                    if self.transform:
                        frame = self.transform(frame)

                    # print(self.mask.dtype)
                    self.frameid += 1

                    self.Q.put((frame,dets,delay))


            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self,tracks):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        if tracks is not None:
            self.cars = tracks
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
