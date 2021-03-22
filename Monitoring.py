""""
Traffic Monitoring using Darknet Yolo and  Opencv library.
Project's Website - https://www.kios.ucy.ac.cy/harpydata/
Developed by,
Rafael Makrigiorgis , Software Engineer,
KIOS Research and Innovation Center of Excellence,
University of Cyprus
Web: www.kios.ucy.ac.cy
Email: makrigiorgis.rafael@ucy.ac.cy
© - 2021
"""

import cv2
import os
from src.tracker import *
import time
from datetime import datetime
from src.queue_thread import *
import src.detector

# Load the configuration file
cfg = configparser.ConfigParser()
cfg.read('config.ini')

"""
## Global configarations taken from the config.ini 
"""
video_filename =  cfg.get('Video','video_filename')
print(video_filename)
export_video_path = cfg.get('Export','video_export_path')
resize = cfg.getboolean('Video','resize') # if you want to resize for better performance
save_results = cfg.getboolean('Export','save_video_results') # False if you don't want to save detection/results in video/image format
im_width = cfg.getint('Video','image_width')
im_height = cfg.getint('Video','image_height')

# you need to set the cfg/weights/class files for the DNN detections
config = cfg.get('Detector','darknet_config')
weights = cfg.get('Detector','darknet_weights')
cfg_size = (cfg.getint('Detector','cfg_size_width'),cfg.getint('Detector','cfg_size_height'))
classes_file = cfg.get('Detector','classes_file')
iou_thresh = cfg.getfloat('Detector','iou_thresh')
conf_thresh = cfg.getfloat('Detector','conf_thresh')
nms_thresh = cfg.getfloat('Detector','nms_thresh')
use_gpu = cfg.getboolean('Detector','use_gpu')

# tracker configs
draw_tracks = cfg.getboolean('Tracker','draw_tracks')
export_data = cfg.getboolean('Tracker','export_data')

# Initializing the Detector
detectNet = detector(weights, config, conf_thresh=conf_thresh, netsize=cfg_size, nms_thresh=nms_thresh, gpu=use_gpu, classes_file=classes_file)
print('Network initialized!')

###################################################################
fps,duration = 0,0
frame_num = 0
classes = None
vid_out = None
video = None
frameA = None
Tracking=None
num_det=0


"""
Function for  Video Detection - Traffic Monitoring
"""
def main_TM():
    global frameA
    global classes
    global frame_num
    global fps,duration
    global Tracking
    # initializing the video
    video = cv2.VideoCapture(video_filename)
    fps = video.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    minutes = int(duration / 60)
    seconds = duration % 60
    print('Total Video duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    frame_num = 0
    key = None
    vid_out = None
    sum_seconds=0
    start_time=time.time()
    print("Start Time:", time.asctime( time.localtime(time.time()) ))

    vs = VideoStream(video,detectNet=detectNet,resize=resize,size=(im_width,im_height),frameid=frame_num).start()
    time.sleep(1.0)
    key = None
    while (frame_num <= frame_count):
        if frame_num== frame_count:
            break
        if key == ord('q') or key == 27:
            vs.stop()
            break
        tmp_cars = Tracking.cars if Tracking is not None else None

        while(vs.more(tmp_cars)):
            start = time.time()
            # read image and detections from the video stream thread
            img,detections,delay = vs.read()
            image = img

            if (image is not None):

                num_det = len(detections[0])
                if frame_num == 0:
                    filename =  video_filename.split('.')[-2].split('/')[-1]
                    # Tracker initialization for the first time
                    Tracking = tracker(detections, image, draw_tracks, detectNet.classes, iou_thresh, export=export_data,
                                       filename=filename, fps=fps)
                else:
                    # updating the tracker using detections and the image frame.
                    test1 = time.time()
                    Tracking.update(detections, image)
                    # print("Tracking time: %.4f" % (time.time() - test1))

                Width = image.shape[1]
                Height = image.shape[0]
                # showing the output image
                if save_results:
                    if frame_num == 0:
                        # creating output folder if save_results is on
                        vid_fps=video.get(cv2.CAP_PROP_FPS)
                        videoname = video_filename.split('.')[-2].split('/')[-1]
                        path = export_video_path
                        try:
                            if not os.path.exists(path):
                                os.makedirs(path, 0o666)
                        except OSError:
                            print("Creation of the directory %s failed" % path)
                        else:
                            print("Successfully created the directory %s " % path)

                        # initialize output video, using  24 fps
                        now = datetime.now()
                        timetmp = now.strftime("%Y%m%d%H%M")
                        vid_out = cv2.VideoWriter(path + 'det_' + videoname +'_'+timetmp+ '.avi',
                                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), vid_fps, (Width, Height))
                    vid_out.write(image) # exporting the image frame to the video
                # displaying the resutls (cannot be shown in code ocean)
                if cfg.getboolean('Video','display_video'):
                    cv2.imshow('Detection', cv2.resize(image,(cfg.getint('Video','display_width'),cfg.getint('Video','display_height'))))  # showing a lower resolution image in order to fit the screen.

                frame_num = frame_num + 1

                # End time
                end = time.time()
                # Time elapsed
                seconds = end - start
                # Calculate frames per second
                seconds = delay if delay>seconds else seconds
                frps = 1 / seconds
                sum_seconds = sum_seconds + frps
                avg_fps = sum_seconds /frame_num
                print("Frame: %d / %d \t FPS/avg %.2f / %.2f \tTotal Dets: %d"%(frame_num,frame_count,round(frps,2),round(avg_fps,2),num_det))

                key = cv2.waitKey(1) & 0xFF
            # If 'q' or 'Esc' keys are pressed, exit the program
            if key == ord('q') or key == 27:
                break

    extract_vidimg_settings(Tracking)
    print("Turning off camera/video.")
    video.release()
    print("Ending Time:", time.asctime( time.localtime(time.time()) ))
    print("Duration:",round(time.time()-start_time),"seconds")
    if save_results:
        print("Realising video output.")
        vid_out.release()
    print("Camera/Video off.")
    print("Program ended.")
    cv2.destroyAllWindows()
    exit(0)

# this executes at the end of the execution
def extract_vidimg_settings(tracker):
    global fps,duration
    """ If exporting data is set to true, initialization of export CSVs """
    if (tracker.export):
        fullpath = cfg.get('Export', 'csv_path') + tracker.filename + '_'
        # write vehicles csv
        with open(fullpath + 'videoinfo.csv', newline='', mode='w') as csv_file:
            fieldnames = ['Name', 'extention','framerate', 'duration(s)', 'width', 'height', 'gsd(km/px)','real_width(km)','real_height(km)']
            writerVeh = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writerVeh.writeheader()

        with open(fullpath + 'videoinfo.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'extention','framerate', 'duration(s)', 'width', 'height', 'gsd(km/px)','real_width(km)','real_height(km)']
            writerVeh = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writerVeh.writerow(
                {'Name':tracker.filename,'extention': video_filename.split('.')[-1], 'framerate':fps, 'duration(s)':duration,
                 'width':tracker.cw, 'height':tracker.ch, 'gsd(km/px)':tracker.km_pixels,
                 'real_width(km)':tracker.km_pixels * tracker.cw,'real_height(km)':tracker.km_pixels * tracker.ch})

if __name__ == '__main__':
    main_TM()
