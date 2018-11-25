#! /usr/bin/env python3

# Copyright(c) 2017-2018 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from threading import Thread
from mvnc import mvncapi as mvnc
from video_processor import VideoProcessor
from ssd_mobilenet_processor import SsdMobileNetProcessor
import cv2
import numpy
import time
import os
import sys
from sys import argv
import serial
import RPi.GPIO as GPIO
import math

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

pin1 = 18 #right forward pin for backward
pin2 = 24 #right forward pin for forward
pin3 = 23 #right back pin for backward
pin4 = 22 #right back pin for forward

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

NETWORK_GRAPH_FILENAME = "./graph"

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

# for title bar of GUI window
cv_window_name = 'video_objects_threaded - SSD_MobileNet'

# the SsdMobileNetProcessor
obj_detector_proc = None

video_proc = None

# read video files from this directory
input_video_path = '.'

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0

stopCascade = cv2.CascadeClassifier("stopsign_classifier.xml")
crossCascade = cv2.CascadeClassifier("crossPedestrian.xml")

stop_flag = True
stop_count = 0
stop_send = 0
sign_flag = False
obstacle_flag = False
sign_start1 = cv2.getTickCount()
sign_stop1 = 0
sign_time1 = 0
obstacle = False
caffe_flag = False

center = 320.0 / 2.0
left_step = (center / (1590.0 - 1280.0)) + 0.3
right_step = (center / (1900.0 - 1590.0)) + 0.5
print(right_step)
print(left_step)

def Line_detection(frame):
    global left_lane, right_lane
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #line_roi = imutils.resize(line_roi, width=480, height = 120)
    #gray = gray[0:gray.shape[0]/2, gray.shape[1]/2:gray.shape[1]]
    lower_yellow = numpy.array([20, 100, 100], dtype = "uint8")
    upper_yellow = numpy.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 100, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    roi = frame[frame.shape[0]/2+40:frame.shape[0], 0:frame.shape[1]]
    line_roi = mask_white[mask_white.shape[0]/2+40:mask_white.shape[0]/2 + 60, 0:mask_white.shape[1]]

    gauss_gray = cv2.GaussianBlur(line_roi,(5,5), 0)
    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)
    
    contours, h = cv2.findContours(gauss_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.rectangle(frame, (x,y+frame.shape[0]/2+10), (x+w/5, y+frame.shape[1]-150), (0,255,0), 1)
        #print("Coordonata X: ", x)
        if(x < 160):
            #print("")
            #print("X1: ", x + 5)
            x1 = x + 5
            left_lane = center - x1
            right_lane = (center - left_lane) + center
            #print("Left lane: ", left_lane)
            #print("Right lane: ", right_lane)
##            print("Curba: ", interval*left_step+50)
##            print("Final: ", 1590 - (interval*left_step + 100))
##            if(interval > 40):
##                rc.servo(1590 - (interval*left_step+100))
##            else:
##                rc.servo(1590)
        elif(x > 160):
            #print("")
            #print("X2: ", x)
            x2 = x
            right_lane = x2 - center
            left_lane = (center - right_lane) + center
            #print("Right lane: ", right_lane)
            #print("Left lane: ", left_lane)

        if(left_lane < right_lane):
            curve = right_lane - left_lane
            #print("Curve: ", curve)
            #print("Curve to right: ", 1590 + right_step * curve)
            right_curve = 1590 + right_step * curve
            rc.servo(right_curve)
            
        elif(left_lane > right_lane):
            curve = right_lane - left_lane
            #print("Curve: ", curve)
            #print("Curve to left: ", 1500 + left_step * curve)
            left_curve = 1500 + left_step * curve
            rc.servo(left_curve)

        elif(left_lane == right_lane):
            print("Indrept rotile")
            rc.servo(1570)
        
    return frame, line_roi, mask_white

h1 = 15.5 - 10  # cm

def DistanceToCamera(v, h, x_shift, image):
    # camera params
    alpha = 8.0 * math.pi / 180
    v0 = 119.865631204
    ay = 332.262498472

    # compute and return the distance from the target point to the camera
    d = h / math.tan(alpha + math.atan((v - v0) / ay))
    if d > 0:
        cv2.putText(image, "%.1fcm" % d,(image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return d

class cascade_thread(Thread):
    def __init__(self, car, ser, display_image):
        Thread.__init__(self)
        self.ser = ser
        self.car = car
        self.display_image = display_image
        self.running = True

    def run(self):
        while self.running:
                gray = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                stops = stopCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(50, 50),
            ##        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
##                crosses = crossCascade.detectMultiScale(
##                    gray,
##                    scaleFactor=1.1,
##                    minNeighbors=5,
##                    minSize=(30, 30),
##            ##        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
##                )

                for (x, y, w, h) in stops:
                    cv2.rectangle(self.display_image, (x+5, y+5), (x+w-5, y+h-5), (0, 0, 255), 2)
                    cv2.putText(self.display_image, "STOP", (x,y-5), font, 0.5, (0, 0, 255), 1)
##                    print("STOP")
                    stop = "1.png"
                    stop = stop.encode("utf-8")
                    self.ser.write(stop)
                    
##                for (x, y, w, h) in crosses:
##                    cv2.rectangle(self.display_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
##                    cv2.putText(self.display_image, "STOP", (x,y-5), font, 0.5, (0, 0, 255), 1)
##                    stop = "2.png"
##                    stop = stop.encode("utf-8")
##                    self.ser.write(stop)
                cv2.imshow("Frame", self.display_image)    
                time.sleep(1)
    
    def stop(self):
        self.running = False

class bluetooth_thread(Thread):
    def __init__(self, car, ser):
        Thread.__init__(self)
        self.ser = ser
        self.car = car
        self.running = True

    def run(self):
        while self.running:
            x = self.ser.read()
            x = x.decode("utf-8")
            ##            print(x)
            if(x != " "):
            ##                print(x)
                if(x == "c"):
##                    print("center")
                    self.car.full_stop()
                    # time.sleep(.05)
                    
                elif(x == "l"):
##                    print("left")
                    self.car.steer_left()
                    # time.sleep(.05)
                    
                elif(x == "r"):
##                    print("right")
                    self.car.steer_right()
                    # time.sleep(.05)
                    
                elif(x == "f"):
##                    print("forward")
                    self.car.full_forward()
                    # speed = 1600
                    # SetSpeed(speed)
                                
                elif(x == "b"):
##                    print("backward")
                    self.car.full_backward()
                    # speed = 1350
                    # SetSpeed(speed)

                elif(x == "s"):
##                    print("stop")
                    self.car.full_stop()
                    # speed = 1500
                    # SetSpeed(speed)

                elif(x == "X"):
                    print("Print")
            time.sleep(0.1)
    
    def stop(self):
        self.running = False

class Car:
    def __init__(self):
        GPIO.setup(pin1, GPIO.OUT)
        GPIO.setup(pin2, GPIO.OUT)
        GPIO.setup(pin3, GPIO.OUT)
        GPIO.setup(pin4, GPIO.OUT)
        GPIO.output(pin1, False)
        GPIO.output(pin2, False)
        GPIO.output(pin3, False)
        GPIO.output(pin4, False)
        
    def full_forward(self):
        GPIO.output(pin2, True)
        GPIO.output(pin4, True)

    def full_stop(self):
        GPIO.output(pin1, False)
        GPIO.output(pin2, False)
        GPIO.output(pin3, False)
        GPIO.output(pin4, False)

    def full_backward(self):
        GPIO.output(pin1, True)
        GPIO.output(pin3, True)

    def steer_left(self):
        GPIO.output(pin1, True)
        #GPIO.output(pin4, True)

    def steer_right(self):
        #GPIO.output(pin2, True)
        GPIO.output(pin3, True)

def handle_keys(raw_key:int, obj_detector_proc:SsdMobileNetProcessor):
    """Handles key presses by adjusting global thresholds etc.
    :param raw_key: is the return value from cv2.waitkey
    :param obj_detector_proc: the object detector in use.
    :return: False if program should end, or True if should continue
    """
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent = obj_detector_proc.get_box_probability_threshold() * 100.0 + 5
        if (min_score_percent > 100.0): min_score_percent = 100.0
        obj_detector_proc.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent = obj_detector_proc.get_box_probability_threshold() * 100.0 - 5
        if (min_score_percent < 0.0): min_score_percent = 0.0
        obj_detector_proc.set_box_probability_threshold(min_score_percent/100.0)
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True



def overlay_on_image(display_image:numpy.ndarray, object_info_list:list, bluetooth, ser, car):
    """Overlays the boxes and labels onto the display image.
    :param display_image: the image on which to overlay the boxes/labels
    :param object_info_list: is a list of lists which have 6 values each
           these are the 6 values:
           [0] string that is network classification ie 'cat', or 'chair' etc
           [1] float value for box upper left X
           [2] float value for box upper left Y
           [3] float value for box lower right X
           [4] float value for box lower right Y
           [5] float value that is the probability 0.0 -1.0 for the network classification.
    :return: None
    """
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    global stop_flag, stop_count, stop_send, sign_flag, obstacle_flag, sign_start1, sign_stop1, sign_time1, h1

    for one_object in object_info_list:
##        if(sign_flag):
        if((one_object[0] == "person") or (one_object[0] == "car")):
##            print("STOP: ", sign_flag)
            caffe_flag = True
            if(obstacle_flag == False):
                obstacle_flag = True
                sign_start1 = cv2.getTickCount()
##            print("START: ", sign_start1)

            string = "0.png"
            if(one_object[0] == "person"):
                string = "3.png"
            elif(one_object[0] == "car"):
                string = "Car.png"
            string = string.encode("utf-8")
            ser.write(string)
            percentage = int(one_object[5] * 100)
        
            label_text = one_object[0] + " (" + str(percentage) + "%)"
            box_left =  int(one_object[1])  # int(object_info[base_index + 3] * source_image_width)
            box_top = int(one_object[2]) # int(object_info[base_index + 4] * source_image_height)
            box_right = int(one_object[3]) # int(object_info[base_index + 5] * source_image_width)
            box_bottom = int(one_object[4])# int(object_info[base_index + 6] * source_image_height)

            box_color = (255, 128, 0)  # box color
            box_thickness = 2
            cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

            v = box_top + (box_bottom - box_top) - 5
            # print x,y,x+w-5, y+h-5, w, h
            distance = DistanceToCamera(v/2.3,h1,300,display_image)
            
            scale_max = (100.0 - min_score_percent)
            scaled_prob = (percentage - min_score_percent)
            scale = scaled_prob / scale_max

            # draw the classification label string just above and to the left of the rectangle
            label_background_color = (0, int(scale * 175), 75)
            label_text_color = (255, 255, 255)  # white text

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = box_left
            label_top = box_top - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                          label_background_color, -1)

            # label text above the box
            cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
    sign_stop1 = cv2.getTickCount()
##    print("STOP: ", sign_stop1)
    sign_time1 = (sign_stop1 - sign_start1)/cv2.getTickFrequency()
##    print("Time: ", sign_time1)

    if(sign_time1 > 5):
        sign_time1 = 0
        obstacle_flag = False

def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask

    labels = SsdMobileNetProcessor.get_classification_labels()

    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).lower().startswith('exclude_classes=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                exclude_list = str(val).split(',')
                for exclude_id_str in exclude_list:
                    exclude_id = int(exclude_id_str)
                    if (exclude_id < 0 or exclude_id>len(labels)):
                        print("invalid exclude_classes= parameter")
                        return False
                    print("Excluding class ID " + str(exclude_id) + " : " + labels[exclude_id])
                    object_classifications_mask[int(exclude_id)] = 0
            except:
                print('Error with exclude_classes argument. ')
                return False;

        elif (str(an_arg).lower().startswith('init_min_score=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                init_min_score_str = val
                init_min_score = int(init_min_score_str)
                if (init_min_score < 0 or init_min_score > 100):
                    print('Error with init_min_score argument.  It must be between 0-100')
                    return False
                min_score_percent = init_min_score
                print ('Initial Minimum Score: ' + str(min_score_percent) + ' %')
            except:
                print('Error with init_min_score argument.  It must be between 0-100')
                return False;

        elif (str(an_arg).lower().startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True


def print_usage():
    """Prints usage information for the program.

    :return: None
    """
    labels = SsdMobileNetProcessor.get_classification_labels()

    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')
    print('  init_min_score - set the minimum score for a box to be recognized')
    print('                  must be a number between 0 and 100 inclusive.')
    print('                  Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print('  exclude_classes - comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)-1) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 init_min_score=50 exclude_classes=5,11')


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """
    ser = serial.Serial(port="/dev/ttyS0", baudrate=9600, parity=serial.PARITY_NONE, timeout=0.2)

    default_string = " 0.0"
    default_plus = "+"
    default_string = default_string.encode("utf-8")
    default_plus = default_plus.encode("utf-8")
    ser.write(default_string)
    ser.write(default_plus)
    ser.write(default_string)
    ser.write(default_plus)

    car = Car()
    bt = bluetooth_thread(car, ser)
    bt.start()

    global stop_flag, stop_count, stop_send, sign_flag, obstacle_flag

    sign_start = cv2.getTickCount()

    global resize_output, resize_output_width, resize_output_height, \
           obj_detector_proc, resize_output, resize_output_width, resize_output_height, video_proc

    if (not handle_args()):
        print_usage()
        return 1

##    # get list of all the .mp4 files in the image directory
##    input_video_filename_list = os.listdir(input_video_path)
##    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]
##    if (len(input_video_filename_list) < 1):
##        # no images to show
##        print('No video (.mp4) files found')
##        return 1

    # Set logging level to only log errors
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)

    devices = mvnc.enumerate_devices()
    if len(devices) < 1:
        print('No NCS device detected.')
        print('Insert device and try again!')
        return 1

    # Pick the first stick to run the network
    # use the first NCS device that opens for the object detection.
    dev_count = 0
    for one_device in devices:
        try:
            obj_detect_dev = mvnc.Device(one_device)
            obj_detect_dev.open()
            print("opened device " + str(dev_count))
            break;
        except:
            print("Could not open device " + str(dev_count) + ", trying next device")
            pass
        dev_count += 1

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    cv2.waitKey(1)

    obj_detector_proc = SsdMobileNetProcessor(NETWORK_GRAPH_FILENAME, obj_detect_dev,
                                              inital_box_prob_thresh=min_score_percent/100.0,
                                              classification_mask=object_classifications_mask)

    exit_app = False
    while (True):
##        for input_video_file in input_video_filename_list :

            # video processor that will put video frames images on the object detector's input FIFO queue
##            video_proc = VideoProcessor(input_video_path + '/' + input_video_file,
##                                         network_processor = obj_detector_proc)
            video_proc = VideoProcessor(0,
                                         network_processor = obj_detector_proc)
            video_proc.start_processing()

            frame_count = 0
            start_time = time.time()
            end_time = start_time

            while(True):
                try:
                    (filtered_objs, display_image) = obj_detector_proc.get_async_inference_result()
                except :
                    print("exception caught in main")
                    raise


                # check if the window is visible, this means the user hasn't closed
                # the window via the X button
                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    video_proc.stop_processing()
                    exit_app = True
                    break

                
                    
                overlay_on_image(display_image, filtered_objs, bt, ser, car)

                display_image = cv2.resize(display_image, (320, 240), cv2.INTER_LINEAR)

##                Line_detection(display_image)
##                cascade = cascade_thread(car, ser, display_image)
##                cascade.start()

                gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                stops = stopCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(50, 50),
            ##        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
##                crosses = crossCascade.detectMultiScale(
##                    gray,
##                    scaleFactor=1.5,
##                    minNeighbors=5,
##                    minSize=(50, 50),
##            ##        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
##                )
                if(len(stops) > 1):
                    sign_flag = True
                else:
                    sign_flag = False
                    
                for (x, y, w, h) in stops:
                    sign_flag = True
                    stop = "1.png"
                    stop = stop.encode("utf-8")
                    ser.write(stop)
                    cv2.rectangle(display_image, (x+5, y+5), (x+w-5, y+h-5), (0, 0, 255), 2)
                    cv2.putText(display_image, "STOP", (x,y-5), font, 0.5, (0, 0, 255), 1)
                    v = y + h - 5
                    # print x,y,x+w-5, y+h-5, w, h
                    distance = DistanceToCamera(v,h1,300,display_image)
##                    print("STOP DETECTED")
##                    print("Person: ", obstacle_flag)

                    if(stop_flag and stop_count == 0):
                        stop_flag = False
                        stop_count = 1
                        car.full_stop()
##                        print("STOP CAR")
                        sign_start = cv2.getTickCount()

                    sign_stop = cv2.getTickCount()
                    sign_time = (sign_stop - sign_start)/cv2.getTickFrequency()
##                    print("Stop time: %.2f" %sign_time)

                    if((sign_time >= 3) and (sign_time < 10)):
                        if(stop_count == 1):
                            if(obstacle_flag == True):
                                car.full_stop()
##                                print("STAY")
                            else:
                                car.full_forward()
                                stop_count = 2
                    
                    if(sign_time > 10):
##                        print("RESET")
                        sign_time = 0
                        stop_flag = True
                        stop_count = 0
                    
##                    print(stop_flag)
                    

##                for (x, y, w, h) in crosses:
##                    cv2.rectangle(display_image, (x+5, y+5), (x+w-5, y+h-5), (0, 0, 255), 2)
##                    cv2.putText(display_image, "CROSS", (x,y-5), font, 0.5, (0, 0, 255), 1)
##                    cross = "2.png"
##                    cross = cross.encode("utf-8")
##                    ser.write(cross)
            
                if (resize_output):
                    display_image = cv2.resize(display_image,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)
                cv2.imshow(cv_window_name, display_image)

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key, obj_detector_proc) == False):
                        end_time = time.time()
                        exit_app = True
                        video_proc.stop_processing()
                        continue

                frame_count += 1

##                if (obj_detector_proc.is_input_queue_empty()):
##                    end_time = time.time()
##                    print('Neural Network Processor has nothing to process, assuming video is finished.')
##                    break

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            throttling = obj_detect_dev.get_option(mvnc.DeviceOption.RO_THERMAL_THROTTLING_LEVEL)
            if (throttling > 0):
                print("\nDevice is throttling, level is: " + str(throttling))
                print("Sleeping for a few seconds....")
                cv2.waitKey(2000)

            #video_proc.stop_processing()
            cv2.waitKey(1)

            video_proc.cleanup()
            GPIO.cleanup()
            bt.stop()
            cascade.stop()
            car.stop()


    # Clean up the graph and the device
    obj_detector_proc.cleanup()
    obj_detect_dev.close()
    obj_detect_dev.destroy()

    cv2.destroyAllWindows()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
