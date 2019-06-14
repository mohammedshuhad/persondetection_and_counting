

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
#import gpss

# Variables
#total_passed_vehicle = 0  # using it to count vehicles

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object, fps, width, height):
        #initialize .csv
	with open('object_counting_report.csv', 'w') as f:
		writer = csv.writer(f)  
		csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
		writer.writerows([csv_line.split(',')])
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
	VIDEO_BASE_FOLDER = '/home/pi/person-detection-and-counting/out_videos/'
	def get_video_filename():
		return VIDEO_BASE_FOLDER+'video_out'+'_video.h264'        
# input video
	camera = PiCamera()
	camera.resolution = (640, 480)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(640, 480))
	#camera.start_recording(get_video_filename())
	out=cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (640,480))
	#cap = cv2.VideoCapture(0)
	the_result = "..."
	width_heigh_taken = True
	height = 0
	width = 0
	i=0
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
			for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
			#while(cap.isOpened()):
				blah=frame.array
				#ret,blah = cap.read()
				input_frame = blah
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
				(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
				font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
				counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(i,input_frame,1,is_color_recognition_enabled,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,targeted_objects=targeted_object,use_normalized_coordinates=True,line_thickness=4)
				#print(the_result)
				if(len(the_result) == 0):
					cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)     
					print("No person detected")                  
				else:
					cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
					print("Person found")
                    #gpss.gpslocation()
                    #sam.sample()
				#cv2.imshow('object counting',input_frame)
				i=i+1
				output_movie.write(input_frame)
				print ("writing frame")
                #if(len(the_result)!=0):
                #    print("Person found")
				out.write(input_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				rawCapture.truncate(0)
                #if(csv_line != "not_available"):
                        #with open('traffic_measurement.csv', 'a') as f:
                                #writer = csv.writer(f)                          
                                #size, direction = csv_line.split(',')                                             
                                #writer.writerows([csv_line.split(',')])         
			camera.stop_recording()
            #cap.release()
			cv2.destroyAllWindows()


