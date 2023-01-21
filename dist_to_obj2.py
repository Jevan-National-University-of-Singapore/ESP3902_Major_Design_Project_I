import pyrealsense2 as rs
import numpy as np
import math
import cv2
import os
import torch
import pandas
import numpy

#distance formula for find the distance between any two detections 
def distance(x1,z1,x2,z2):
    return math.sqrt((x2-x1)**2 + (z2-z1)**2)
 
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 320,240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
profile = pipeline.start(config)

# Getting the cameras depth scale (
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(depth_scale)

try:
    while True:
        
        #get camera frames (color and depth)
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        if not depth_frame or not color_frame:
            continue

     
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        #initialise the classifier 
        #detect and store pixel values
        #modify here for the yolo!!
        face_cascade = torch.hub.load('ultralytics/yolov5', 'yolov5n')
        face_cascade.classes = [0]
        face_cascade.conf = 0.5
  
        bodies=face_cascade(color_image)
        bodies = bodies.pandas().xyxy[0].values
        #find the depth and co-ordinates of the detected human in meters 
        if len(bodies) != 1:
            dict = {}
        
            
            for i in range(len(bodies)):
                row = bodies[i]
                x,y,h,w = int(row[0]),int(row[1]), int(row[2]),int(row[3])
                c_x,c_y = (x + int(w-x/2),h + int(y-h/2))
                depth = depth_image[x:w,y:h].astype(float)
                depth_scale = profile.get_dehvice().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale
                dist,_,_,_ = cv2.mean(depth)
                
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x,c_y], dist)
                dict[str(i)] = depth_point#depth of object detected,xyz
                #print(dict[str(i)])
            #get the distance value between all the detected bodies and store it in a dictionary 
            dist = {}
            for m in range(len(dict)):
                for n in range(len(dict)-1):
                    if n!=m:
                        x1,y1,z1 = dict[str(m)]
                        x2,y2,z2 = dict[str(n)]
                        dist[str(m)+str(n)] = distance(x1,z1,x2,z2)
                      
            #catogarise if social distancing is being followed and dipict this with red and blue boxes
            track = []
            for key in dist.keys():
                
                if (dist[key] < 1) and (int(key[0]) not in track):
                    row = bodies[int(key[0])]
                    (x,y,w,h) = int(row[0]),int(row[1]), int(row[2]),int(row[3])
                    cv2.rectangle(color_image,(x,y),(w,h),(0,0,255),2)
                    track.append(int(key[0]))
                    #print(dist[key])
                if (dist[key] < 1) and (int(key[1]) not in track):
                    row = bodies[int(key[1])]
                    (x,y,w,h) = int(row[0]),int(row[1]), int(row[2]),int(row[3])
                    cv2.rectangle(color_image,(x,y),(w,h),(0,0,255),2)
                    track.append(int(key[1]))
                    #print(dist[key])
                if (dist[key] > 1) and (int(key[1]) not in track) and (int(key[0]) not in track):
                    #(xm,ym,wm,hm) = bodies[int(key[0])]
                    #(xn,yn,wn,hn) = bodies[int(key[1])]
                    #cv2.rectangle(color_image,(xm,ym),(xm+wm,ym+hm),(255,0,0),2)
                    rowm = bodies[int(key[0])]
                    rown = bodies[int(key[1])]
                    (xm,ym,wm,hm) = int(rowm[0]),int(rowm[1]), int(rowm[2]),int(rowm[3])
                    (xn,yn,wn,hn) = int(rown[0]),int(rown[1]), int(rown[2]),int(rown[3])
                    cv2.rectangle(color_image,(xm,ym),(wm,hm),(255,0,0),2)
                    cv2.rectangle(color_image,(xn,yn),(wn,hn),(255,0,0),2)
                    #print(dist[key])
                    
        else:
            #x,y,w,h = bodies[0]
            row = bodies[0]
            x,y,h,w = int(row[0]),int(row[1]), int(row[2]),int(row[3])
            cv2.rectangle(color_image,(x,y),(w,h),(255,0,0),2)
        
        
        #Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
        
        

finally:

    # Stop streaming
    pipeline.stop()


'''

                dis = depth_frame.get_distance(c_x,c_y)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c_x,c_y], dis)
                [x3d,y3d,z3d] = rs.rs2_transform_point_to_point(depth_to_color_extrin,depth_point)
                dict[str(i)] = [x3d,y3d,z3d]
                print(dict[str(i)])
'''

                