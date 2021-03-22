import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
import sys
import queue
from numpy.linalg import inv
from sklearn.neighbors import KDTree
import math
import csv
from src.detector import *
from src.kalman import *
from scipy import signal
import operator
from scipy.interpolate import *
import time

import configparser
# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

###########Classes################################
""" bounding box class """
class box_in(object):
    def __init__(self, x=None, y=None, w=None, h=None,confidence=None,label = None,direction=""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.label = label
        self.direction = direction

""" Main Class for each vehicle 
    self.box is a box_in class
"""
class bount_boxes(object):
    def __init__(self,box=None, count_id=None, frame_id=None, kalman=None, kalman_box=None, velocity=0,parked=None):
        self.box = box
        self.count_id=count_id
        self.frame_id = frame_id
        self.kalman = kalman
        self.kalman_box = kalman_box
        self.active = True
        self.velocity = velocity

class trajectories(object):
    def __init__(self,points,direction,id):
        self.points=points
        self.direction=direction
        self.id = id
"""
Tracking Class
Needs the previous boxes(cars), current detections and image
For first initialization, previous boxes should be an empty array
    Cars => array :
        box         => (x,y,w,h,confidence,label,direction)
        count_id    => int
        frame_id    => array
        kalman      => kalman class
        kalman_box  => (x,y) array
        active      => Boolean
        velocity    => Array
"""
class tracker():
    def __init__(self,  detections, image, draw_track=True,classes=[],iou_thresh=0.3,export=True,filename='',fps=0):
        self.cars = []
        self.detections = detections
        self.image = image
        self.draw_track = draw_track
        self.frame_id=0
        self.classes=classes
        self.iou_thresh=iou_thresh
        self.export=export
        self.filename=filename.split('.')[0]
        self.fps = fps
        self.ch,self.cw = self.image.shape[:2]
        self.trajectories = []
        self.km_pixels = 0


        # first time initialization
        imH,imW= self.ch,self.cw # height and width of the image

        if self.image is not None and detections is not None:
            indices, boxes, classIDs, confidences= self.detections
            index = 0
            """ If exporting data is set to true, initialization of export CSVs """
            if (self.export):
                fullpath = config.get('Export','csv_path') + self.filename + '_'
            # write vehicles csv
                with open(fullpath+'vehicles.csv', newline='', mode='w') as csv_file:
                    fieldnames = ['veh_id', 'video', 'class', 'initW', 'initH', 'initFrame']
                    writerVeh = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writerVeh.writeheader()

                with open(fullpath+'tracks.csv', newline='', mode='w') as csv_file:
                    fieldnames2 = ['veh_id', 'video', 'frame', 'bboxX', 'bboxY', 'Width', 'Height','Direction',
                                   'Velocity']
                    writerTra = csv.DictWriter(csv_file, fieldnames=fieldnames2)
                    writerTra.writeheader()

            for i in indices: # loop all detections
                i = i[0]
                box = boxes[i]
                x,y,w,h = box[0],box[1],box[2],box[3]
                temp_box,temp_frame,tmp_kmlbox,tmp_parked,tmp_velo = [],[],[],[],[]

                box_temp,classid = [round(x), round(y), round(w), round(h)],classIDs[i]
                classid = classIDs[i]
                labelz = str(self.classes[classid])
                confidence=confidences[i]

                # checking if the box is out of bounds of the image frame
                x,y,w,h=out_of_bounds((x,y,w,h),0,0,imW,imH)
                temp_box.append(box_in(round(x), round(y), round(w), round(h), confidence, labelz))

                # saving Frame number
                temp_frame.append(self.frame_id)

                # initializing the Kalman filter using the first detection
                XY = (x, y)
                tmp_kalman = KalmanFilter(XY)
                tmp_kmlbox.append(tmp_kalman.predict())
                # Initializing/saving the car
                self.cars.append(bount_boxes(temp_box, index, temp_frame, tmp_kalman, tmp_kmlbox,parked=tmp_parked))

                # Initializing the Velocity, it is set as 0 for the first time.
                tmp_velo.append(0)
                self.cars[index].velocity = tmp_velo

                if (self.export):
                # write vehicles to csv files
                    fullpath = config.get('Export', 'csv_path') + self.filename + '_'
                    with open(fullpath+'vehicles.csv', newline='', mode='a') as csv_file:
                        fieldnames = ['veh_id', 'video', 'class', 'initW', 'initH', 'initFrame']
                        writerVeh = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writerVeh.writerow(
                            {'veh_id': index, 'video': self.filename, 'class': labelz, 'initW': w, 'initH': h,
                             'initFrame': self.frame_id})

                    with open(fullpath+'tracks.csv', newline='', mode='a') as csv_file:
                        fieldnames2 = ['veh_id', 'video', 'frame', 'bboxX', 'bboxY', 'Width', 'Height', 'Direction',
                                       'Velocity']
                        writerTra = csv.DictWriter(csv_file, fieldnames=fieldnames2)
                        writerTra.writerow(
                            {'veh_id': index, 'video': self.filename, 'frame': self.frame_id, 'bboxX': x, 'bboxY': y,
                             'Width': w, 'Height': h,'Direction':'','Velocity': 0})

                if self.draw_track:
                    # Displaying the results in the image
                    self.draw_tracks(index)

                # increasing the car index ( car id for tracking it)
                index +=1
        # increasing frame number
        self.frame_id+=1

    """
        Update function of the Tracker, needs to be updated for every frame
    """
    def update(self,detections,image):
        self.dir_counters = np.zeros(9)
        self.detections = detections
        self.image = image
        indices, boxes, classIDs, confidences = self.detections
        indices_cp, boxes_cp, classIDs_cp, confidences_cp = self.detections
        imH,imW = image.shape[:2]
        adboxes,boxesXY,distance_tree,indexes,indices2=[],[],[],[],[]

        """
        Checking all previously active boxes for any match with the current detections
        """

        # KDTree helps us find the nearest vehicles of each detection
        if len(boxes) > 0:
            distance_tree = KDTree(np.asarray(boxes)[:,:2],leaf_size=5)
            indices2 = list(map(lambda t: t[0], indices))

        for car in (tmp for tmp in  self.cars if tmp.active == True): # getting only the active cars
            found = 0
            # must match it with previous cars
            box_id,box_latest ,kalm_latest= car.count_id,car.box[-1],car.kalman_box[-1]

            box_temp2=[]
            box_temp2 = [kalm_latest[0][0], kalm_latest[1][0], box_latest.w,
                         box_latest.h]
            if len(boxes) >= 5:
                distances, indexes = distance_tree.query(
                    np.asarray([box_latest.x, box_latest.y], dtype=int).reshape(-1, 2), 5)
            elif 5 > len(boxes) > 0:
                distances, indexes = distance_tree.query(
                    np.asarray([box_latest.x, box_latest.y], dtype=int).reshape(-1, 2), len(boxes))

            # print("First",distances,indexes[0])
            temp_iou = []
            scoreiou=[]
            for index in indexes[0]:
                i = indices_cp[index][0]
                # print(i)
                box = boxes_cp[i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                box_temp = [x, y, w, h]
                classid = classIDs_cp[i]
                labelz = str(self.classes[classid])
                confidence = confidences_cp[i]

                """ [ Hungarian Algorithm ]
                     Checking IOU Threshold between the current detection and the latest kalman detection of each active box.
                     If it's true, the detected object already existed before.
                """
                scoreiou = iou(box_temp2, box_temp)
                if scoreiou >= self.iou_thresh:
                    temp_iou.append((box_temp, index, scoreiou, confidence, labelz))
                    found = 1
                    # break
            # finding the iou scores for each detection


            """iou score"""
            indindex=0
            if len(temp_iou)>1:
                temp_iou = sorted(temp_iou, key=operator.itemgetter(2), reverse=True)
            if len(temp_iou)>0:
                try:
                    indices2.remove(temp_iou[0][1])
                except ValueError as e:
                    found=0
                    pass  # do nothing!
            if found == 1:      # a match is found previously
                # appending the new detection to the corresponding vehicle id
                x,y,w,h=temp_iou[0][0]
                confidence,labelz = temp_iou[0][3],self.classes[classIDs_cp[indices_cp[temp_iou[0][1]][0]]]
                temp_box,temp_frame = car.box, car.frame_id
                # checking if box is out of bounds
                x, y, w, h = out_of_bounds((x,y,w,h), 0, 0, imW, imH)
                # calculate the direction of the box
                direction = self.get_directions(car.box)
                # appending the info into the corresponding object
                temp_box.append(box_in(round(x), round(y), round(w), round(h), confidence, labelz,direction))
                temp_frame.append(self.frame_id)
                # self.cars[box_id].kalman.predict()
                cm = [[x],[y]]#np.array([[np.float32(x)], [np.float32(y)]])

                # correcting kalman using the detected coordinates
                self.cars[box_id].kalman.correct(cm)

                # doing kalman predict once again in order to be one step ahead
                xykal = self.cars[box_id].kalman.predict()
                kal_tmp = self.cars[box_id].kalman_box
                kal_tmp.append(xykal)

                self.cars[box_id].kalman_box = kal_tmp
                self.cars[box_id].box = temp_box
                self.cars[box_id].frame_id = temp_frame
                index = box_id

                # calculating and saving the velocity
                velocity = self.calc_velocity(box_id)
                tmp_velo = self.cars[index].velocity
                tmp_velo.append(velocity)
                self.cars[index].velocity = tmp_velo
                font_size = self.image.shape[0] / 3000

                """ Display the results found """
                if self.draw_track:
                    self.draw_tracks(index)
                found = 1 # indicates that a match has been found

                # exporting the results to CSV files
                if (self.export):
                    fullpath = config.get('Export', 'csv_path') + self.filename + '_'
                    with open(fullpath+'tracks.csv', newline='', mode='a') as csv_file:
                        fieldnames2 = ['veh_id', 'video', 'frame', 'bboxX', 'bboxY', 'Width', 'Height', 'Direction',
                                       'Velocity']
                        writerTra = csv.DictWriter(csv_file, fieldnames=fieldnames2)
                        writerTra.writerow(
                            {'veh_id': index, 'video': self.filename, 'frame': self.frame_id, 'bboxX': x,
                             'bboxY': y, 'Width': w, 'Height': h, 'Direction': direction, 'Velocity': velocity})
            elif found == 0:
                self.check_actives(car.count_id)
            # Line counter for each car
            """ 
                Active box did not matched with any detection
                Since it is not found in new detections, checking if it's time to set this as inactive 
            """

        font_size = self.image.shape[0] / 2500
        color = (0.0, 255.0, 255.0)

        """
            Saving all the new detections 
        """
        for i in indices2:
            # i = i[0]
            box = boxes_cp[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # self.check_random_spawn(box)
            box_temp = [round(x), round(y), round(w), round(h)]
            classid = classIDs_cp[i]
            labelz = str(self.classes[classid])
            confidence = confidences_cp[i]
            index = len(self.cars)
            temp_box = []
            temp_frame = []
            x, y, w, h = out_of_bounds((x, y, w, h), 0, 0, imW, imH)
            temp_box.append(box_in(round(x), round(y), round(w), round(h), confidence, labelz))
            temp_frame.append(self.frame_id)
            tmp_kmlbox = []
            XY = (x, y)
            tmp_kalman = KalmanFilter(XY)
            tmp_kmlbox.append(tmp_kalman.predict())
            tmp_parked = []
            self.cars.append(bount_boxes(temp_box, index, temp_frame, tmp_kalman, tmp_kmlbox))
            tmp_velo = []
            tmp_velo.append(0)
            self.cars[index].velocity = tmp_velo

            if (self.export):
                # write vehicles csv
                fullpath = config.get('Export','csv_path') + self.filename + '_'
                with open(fullpath + 'vehicles.csv', newline='', mode='a') as csv_file:
                    fieldnames = ['veh_id', 'video', 'class', 'initW', 'initH', 'initFrame']
                    writerVeh = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writerVeh.writerow(
                        {'veh_id': index, 'video': self.filename, 'class': labelz, 'initW': w, 'initH': h,
                         'initFrame': self.frame_id})

                with open(fullpath + 'tracks.csv', newline='', mode='a') as csv_file:
                    fieldnames2 = ['veh_id', 'video', 'frame', 'bboxX', 'bboxY', 'Width', 'Height', 'Direction',
                                   'Velocity']
                    writerTra = csv.DictWriter(csv_file, fieldnames=fieldnames2)
                    writerTra.writerow(
                        {'veh_id': index, 'video': self.filename, 'frame': self.frame_id, 'bboxX': x, 'bboxY': y,
                         'Width': w, 'Height': h, 'Direction': '', 'Velocity': 0})

            """ Display the results found """
            if self.draw_track:
                self.draw_tracks(index)
            adboxes.append(i)
        self.frame_id+=1

    # Function to calculate the direction that a vehicle is heading based on previous points
    def get_directions(self,box):
        direction=""
        nsize = 25
        dist = 5
        if len(box)>nsize:
            lb=len(box)-1
            if self.frame_id >= nsize and box[-nsize] is not None:  # and i == 1
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = box[-nsize].x - box[lb].x
                dY = box[-nsize].y - box[lb].y
                (dirX, dirY) = ("", "")
                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > int(box[-1].w/2):
                    dirX = "Left" if np.sign(dX) == 1 else "Right"
                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) >  int(box[-1].h/2):
                    dirY = "Up" if np.sign(dY) == 1 else "Down"
                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)
                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY
        return direction
    """    
    Euclidean distance calculation. 
    """
    def eucl_dist(self,x1,x2,y1,y2):
        result = math.sqrt(pow(abs(x2 - x1), 2.0) + pow(abs(y2 - y1), 2.0))
        return result
    """
    Using Ground Sample Distance calculation to convert the distance in kilometers / pixel.
    """
    def gsd_calc(self,x1,y1,x2,y2):
        # GSD CALCULATION
        # (FH*SH)/FL*IH)

        # *** Flight Height/Width, Sensor Height, Focal Length, Image Height/Width ***
        # Image Width
        # 1280px
        # Image Height
        # 720 px
        # Sensor Width
        # 6.17mm
        # Sensor Height
        # 4.55 mm
        # Focal Length
        # 5.67 mm or 3.57

        flightH = config.getint('Tracker','flight_height')*100 #250m = 25000
        sensorH = config.getfloat('Tracker','sensor_height')
        sensorW = config.getfloat('Tracker','sensor_width')
        focalL  = config.getfloat('Tracker','focal_length')

        imHeight,imWidth = self.image.shape[:2]
        GSDh,GSDw = (flightH*sensorH)/(focalL*imHeight),(flightH*sensorW)/(focalL*imWidth)
        pixels_km= (GSDw/100000) if GSDw > GSDh else (GSDh/100000)  # getting the worst gsd value
        self.km_pixels = pixels_km
        result = math.sqrt(pow(abs(x2 - x1) * pixels_km, 2.0) + pow(abs(y2 - y1) * pixels_km, 2.0))
        return result

    def draw_trajectory(self,boxid,image):
        # counter ids
        # [0-7] Top,Top Right, Right, Bottom Right, Bottom, Bottom Left,Left, Top Left
        #choose a color for each direction
        car = self.cars[boxid].box
        if car[-1].direction == "Right":
            color = (0.0,0.0,255.0) #red
            self.dir_counters[2] +=  1
        elif car[-1].direction == "Left":
            color = (0.0,255.0,0.0) #green
            self.dir_counters[6] +=  1
        elif car[-1].direction == "Up":
            color = (255.0,0.0,0.0) #blue
            self.dir_counters[0] +=  1
        elif car[-1].direction == "Down":
            color = (0.0, 255.0, 255.0) #yellow
            self.dir_counters[4] +=  1
        elif car[-1].direction == "Up-Left":
            color = (255.0, 255.0, 0.0)  #roz
            self.dir_counters[7] +=  1
        elif car[-1].direction == "Down-Left":
            color = (22.0, 185.0, 158.0) #purlple
            self.dir_counters[5] +=  1
        elif car[-1].direction == "Up-Right":
            color = (255.0, 0.0, 255.0) #dark green
            self.dir_counters[1] +=  1
        elif car[-1].direction == "Down-Right":
            color = (0.0, 154.0, 255.0)  # dark blue
            self.dir_counters[3] +=  1
        else:
            color = (128.0,128.0,128.0)
            self.dir_counters[8] +=  1

        lines = []

        imH,imW = self.image.shape[:2]
        x,y=[],[]
        filter = False
        if len(car)<=15:
            lines.append((int(car[0].x + (car[0].w / 2)), int((car[0].y + (car[0].h / 2)))))
            lines.append((int(car[-1].x + (car[-1].w / 2)), int((car[-1].y + (car[-1].h / 2)))))
            pts = np.array(lines, dtype=np.int32)
            if self.draw_track:
                cv2.polylines(self.image, [pts], False, color, thickness=1)
        else:
            xy = car[::15]
            linex,liney = list(map(lambda x: x.x+(x.w/2) ,xy)),list(map(lambda x: x.y+(x.h/2) ,xy))
            lines = np.column_stack((linex,liney))

            if filter:
                xnew,ynew = signal.savgol_filter(x, 15, 3, mode='nearest'),signal.savgol_filter(y, 15, 3, mode='nearest')
                newlines = np.column_stack((xnew, ynew))
                pts = np.array(newlines, dtype=np.int32)
                if self.draw_track:
                    cv2.polylines(self.image, [pts], False, color, thickness=1)
            else:
                pts = np.array(lines, dtype=np.int32)
                if self.draw_track:
                    cv2.polylines(self.image, [pts], False, color, thickness=1)

    """
     Since it is not found in new detections, checking if it's time to set this as inactive 
    """
    def check_actives(self,index):
        # Set inactive a box if it is not appeared for a few frames. Prevents false box matching.
        box = self.cars[index]
        # If the box is undetected for the past 30 frames, it is inactive.
        reset_time = config.getint('Tracker', 'reset_boxes_frames')
        if box.frame_id[-1] + reset_time <= self.frame_id:
            self.cars[box.count_id].active = False

        else:
            """ 
            Kalman Predict so the movement will still be available.
            Also helps for better tracking.
            """
            # kalman
            if len(box.box)>15 :
                x, y, w, h = box.box[-1].x, box.box[-1].y, box.box[-1].w, box.box[-1].h
                sumw,sumh = 0,0

                tmp_box = box.box[-11:]
                sumw,sumh = int(np.sum(list(map(lambda w: w.w, tmp_box)))/len(tmp_box)), \
                            int(np.sum(list(map(lambda h: h.h, tmp_box)))/len(tmp_box))
                kx, ky, kw, kh = box.kalman_box[-1][0], box.kalman_box[-1][1],sumw, sumh
                kal_tmp = self.cars[index].kalman_box
                xykal = self.cars[index].kalman.predict()
                cm = (xykal[0],xykal[1])
                self.cars[index].kalman.correct(cm)
                kal_tmp.append(xykal)
                temp_box = self.cars[index].box
                oldbox = temp_box[-1]
                imH,imW = self.image.shape[:2]
                cm = xykal

                (x,y, kw, kh),out_of = out_of_bounds_actives(( int(cm[0][0]+(kw/2)),  int(cm[1][0]+(kh/2)), kw, kh), 0, 0, imW, imH)
                if out_of == 1:
                    self.cars[box.count_id].active = False
                    return
                temp_box.append(box_in(int(x-(kw/2)),int(y-(kh/2)), kw, kh, 0, oldbox.label, oldbox.direction))
                # saving
                self.cars[index].kalman_box = kal_tmp

                #save the same frame id
                temp_frame = box.frame_id
                temp_frame.append(self.cars[index].frame_id[-1])

                #calculate velocity
                velocity = self.cars[index].velocity[-1]
                tmp_velo = self.cars[index].velocity
                tmp_velo.append(velocity)
                self.cars[index].velocity = tmp_velo
                if len(self.cars[index].box)>1:
                    if self.draw_track:
                        self.draw_trajectory(index, self.image)

    """ 
    Function used to Display info about the detected objects on the image. (eg bounding box, direction,velocity..)
    """
    def draw_tracks(self,index):

            # getting the correct object based on its index
            box = self.cars[index]
            x,y,w,h,kx,ky,kw,kh = box.box[-1].x,box.box[-1].y,box.box[-1].w,box.box[-1].h,box.kalman_box[-1][0],\
                                  box.kalman_box[-1][1],box.box[-1].w,box.box[-1].h

            index,labelz= box.count_id,box.box[-1].label

            font_size,color,colors = 0.35,(255.0, 0.0, 0.0),[]
            colors.append((255.0, 0.0, 0.0))
            colors.append((0.0, 0.0, 255.0))
            colors.append((0.0, 255.0, 0.0))
            colors.append((0.0, 255.0, 255.0))
            cidx=0 #color index
            if labelz== 'car':
                cidx=0
            elif labelz== 'bus':
                cidx=1
            elif labelz == 'truck':
                cidx = 2
            elif labelz == 'motorbike':
                cidx = 3
            imh,imw = self.image.shape[:2]

            # displaying the bounding box
            cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)),colors[cidx], 1)
            # display kalman filter
            # cv2.rectangle(self.image, (int(kx), int(ky)), (int(kx + kw), int(ky + kh)),(17.0,128.0,213.0), 1)

            # displaying the detected objects class label (eg. car, truck etc..)
            label = labelz + ' #' + str(index)
            cv2.putText(self.image, label, (x, y - int(imh*0.005)), cv2.FONT_HERSHEY_TRIPLEX, font_size, colors[cidx], 1)

            # displaying the direction of the object if it's not empty.
            if box.box[-1].direction is not "":
                #direction = box.box[-1].direction
                cv2.putText(self.image, box.box[-1].direction, (x , y - int(imh*0.015)), cv2.FONT_HERSHEY_TRIPLEX,font_size, color, 1)

            # # displaying the velocity of the object if it's not zero.
            if box.velocity[-1] is not 0:
                 cv2.putText(self.image, 'vel: '+str( box.velocity[-1]), (x , y - int(imh*0.025)), cv2.FONT_HERSHEY_TRIPLEX, font_size, (199,0,0), 1)

            if len(self.cars[index].box)>1:
                if self.draw_track:
                    self.draw_trajectory(index, self.image)


    """"
    Velocity calculation
    Calculating the gsd  between the latest 24 detections.
    Velocity = (Sum of distance) / ( Frame Difference between first and last detection / Frame Rate of the video)
    """
    def calc_velocity(self,index):

        carbox = self.cars[index].box
        car,sum_dist = self.cars[index],0
        limit = config.getint('Tracker','calc_velocity_n')

        frame_rate = self.fps*3600 # convert frame rate to hours
        if len(carbox)>= limit:
            frame_diff = self.cars[index].frame_id[-1] - self.cars[index].frame_id[-limit]
            carbox1,carbox2 = carbox[-limit:], carbox[-(limit+1):-1]
            sum = np.sum(list(map(lambda car1,car2: self.gsd_calc(car1.x,car1.y,car2.x,car2.y) ,carbox1,carbox2)))

        else:
            frame_diff = self.cars[index].frame_id[-1] - self.cars[index].frame_id[-len(carbox)]
            carbox1, carbox2 = carbox[1:],carbox[:-1]
            sum = np.sum(list(map(lambda car1, car2: self.gsd_calc(car1.x, car1.y, car2.x, car2.y), carbox1, carbox2)))
        velo = int((sum) /(frame_diff/frame_rate))
        euc_dist =9999
        #calc dist between last
        if len(carbox) >limit:
            x, y = [], []
            points = [x, y]
            carbox1 = carbox[1::limit]
            carbox2 = carbox[:-1:limit]
            euc_dist = np.mean(list(map(lambda car1, car2: self.eucl_dist(car1.x, car2.x, car1.y, car2.y), carbox1, carbox2)))

        # Set velocity as 0 if it is below 10 and below 7px distance
        if velo <= 15 and euc_dist<15 :#or self.cars[index].parked[-1].parked:
            velo = 0
        return velo




"""
    SUPPORTING FUNCTIONS
"""

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    # bbox1 = [float(x) for x in bbox1]
    # bbox2 = [float(x) for x in bbox2]
    if len(bbox1)==0 or len(bbox2)==0:
        return 0
    (x0_1, y0_1, w1_1, h1_1), (x0_2, y0_2, w1_2, h1_2) = bbox1,bbox2
    x1_1,x1_2,y1_1,y1_2 = x0_1 + w1_1,x0_2 + w1_2,y0_1 + h1_1,y0_2 + h1_2

    # get the overlap rectangle
    overlap_x0,overlap_y0,overlap_x1,overlap_y1 = max(x0_1, x0_2),max(y0_1, y0_2),min(x1_1, x1_2),min(y1_1, y1_2)
    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1, size_2 = (x1_1 - x0_1) * (y1_1 - y0_1),(x1_2 - x0_2) * (y1_2 - y0_2)
    # size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


""" 
    Checking if the box given is out of the bounds of the image.
    zx,zy must be given as the lowest values (eg. 0,0) and W,H as the maximum Width,Height of the image respectively.
"""
def out_of_bounds(box,zx,zy,W,H):
    x,y,w2,h2 = box
    oobX,oobY = 0,0

    if x < zx:
        x,oobX  = zx,1
        # oobX =1

    elif x>W:
        x,oobX = W, 1
        # oobX =1
    if y < zy:
        y,oobY = zy,1
        # oobY =1

    elif y>H:
        y,oobY = H,1
        # oobY =1

    box = x,y,w2,h2
    if zx ==0 and zy ==0 and oobX ==1 and oobY==1:
        box = 0,0,0,0

    return box

#check if a box went out of the image's bounds
def out_of_bounds_actives(box,zx,zy,W,H):
    x,y,w2,h2 = box
    oobX,oobY = 0,0
    if x < zx:
        x, oobX = zx, 1
        # oobX =1

    elif x > W:
        x, oobX = W, 1
        # oobX =1
    if y < zy:
        y, oobY = zy, 1
        # oobY =1

    elif y > H:
        y, oobY = H, 1
        # oobY =1

    out_of = 0
    box = x,y,w2,h2
    if zx ==0 and zy ==0 and oobX ==1 and oobY==1:
        box = 0,0,0,0
    if  oobX == 1 or oobY == 1:
        out_of = 1
    return box,out_of

