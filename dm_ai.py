#**************************************************************************************
#
#   T4AV - Driver Monitoring Systems using AI (Assignment)
#
#   File: dm-ai.py
#   Author: Amir Sasanfar
#   Company: Politecnico di Torino
#
#
#**************************************************************************************

import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os
import math

DROWSY_EAR=0.8
# EAR baseline(the normal openness)
baseline_ear = 0.3

# Drowsiness detection (timing the drowsiness)
drowsy_start_time = None
drowsiness_timer = 0

#for PERCLOS
ear_history = []
interval_duration_perclos = 20 # measured in seconds
threshold_perclos = 0.25 # under this value the eyes are considered closes
update_time_perclos = time.time()
##############distraction
distracted_timeout = 0
distracted_msg_timeout = 0
init = 0

#for head angles
NORMAL_ANGLES = [0,0,0] # in order pith , yaw, roll
distracted = False # setting a flag for determining distraction
################functions##################
def ear_cal(coord_list: list): # in order p1_RE , p2_RE --> p6_RE the same for LE
    numerator_ear = abs(coord_list[1][1] - coord_list[5][1]) + abs(coord_list[2][1] - coord_list[4][1])
    denominator_ear = 2*abs(coord_list[0][0] - coord_list[3][0])
    return numerator_ear/denominator_ear
# function for computing PERCLOS
def perclos_cal(ear_history, threshold):
    if not ear_history:
        return 0
    closed_eyes_count = sum(1 for ear, t in ear_history if ear < threshold)
    return closed_eyes_count / len(ear_history)

#distraction criteria


################END functions##################

# 1 - Imports
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2 - Set the desired setting
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 3 - Open the video source
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)

start = time.time() # timing start before the loop
# 4 - Iterate (within an infinite loop)

while cap.isOpened(): 
    
    # 4.1 - Get the new frame
    success, image = cap.read() 
    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    #else: #needed with some cameras/video input format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    #these are tuples
    point_RER = [] # Right Eye Right # p1
    point_REB = [] # Right Eye Bottom
    point_REL = [] # Right Eye Left # p4 
    point_RET = [] # Right Eye Top
    p2_RE = [] #p2
    p3_RE = [] #p3
    p5_RE=[]
    p6_RE = [] #p6

    point_LER = [] # Left Eye Right # p1
    point_LEB = [] # Left Eye Bottom
    point_LEL = [] # Left Eye Left #p4
    point_LET = [] # Left Eye Top
    p2_LE =[]
    p3_LE =[]
    p5_LE= []
    p6_LE = []

    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center
    # added for gaze and distraction tracking
    face_2d = []
    face_3d = []
    left_eye_2d = []
    left_eye_3d = []
    right_eye_2d = []
    right_eye_3d = []
    #

    # 4.3 - Get the landmark coordinates
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):


                # Eye Gaze (Iris Tracking)
                # Left eye indices list
                LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                # Right eye indices list
                RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                LEFT_IRIS = [473, 474, 475, 476, 477]
                RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 33: #p1
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                # setting the rest of the eye coordinates for EAR
                # RE
                if idx == 160:
                    p2_RE = (lm.x *img_w , lm.y*img_h)
                if idx == 158:
                    p3_RE = (lm.x *img_w , lm.y*img_h)
                if idx == 153:
                    p5_RE = (lm.x *img_w , lm.y*img_h)
                if idx == 144:
                    p6_RE = (lm.x *img_w , lm.y*img_h)
                #
                if idx ==1:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 133: #p4
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                # EAR calculation coordinates Left Eye
                if idx == 362: # p1_LE
                    point_LER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 385:
                    p2_LE = (lm.x * img_w, lm.y * img_h)
                if idx == 387:
                    p3_LE = (lm.x * img_w, lm.y * img_h)
                if idx == 373:
                    p5_LE = (lm.x * img_w, lm.y * img_h)
                if idx == 380:
                    p6_LE = (lm.x * img_w, lm.y * img_h)
                #
                if idx == 374: 
                    point_LEB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 263: # p4 left eye
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 255, 0), thickness=-1)                    
                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 255), thickness=-1)
                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                
                
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            #converting into numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            # Calcuate camera matrix
            focal_length = 2 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            
            dist_matrix = np.zeros((4,1), dtype=np.float64)
            ## Calculate head gaze based on 3d points
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            # Get rotational matrices
            rmat, jac = cv2.Rodrigues(rot_vec)
        
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # Convert angles in degrees
            pitch = angles[0] * 1800
            yaw = -angles[1] * 1800
            roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
            if roll > 180:
                roll = roll - 360
            # Display head gaze angles
            cv2.putText(image, f"HEAD Roll: {roll:.2f} Pitch: {pitch:.2f} Yaw: {yaw:.2f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 

            head_distracted = 0
            if abs(roll) > 30 or abs(pitch) > 30 or abs(yaw) > 30:
                head_distracted = 1
            ## Calculate eyes gaze based on 2d points ##
            width_fraction = 0.5
            height_fraction = 0.4
            down_gaze_fraction = 0.3

            # compute some eye parameters
            r_eye_height = point_REB[1] - point_RET[1]
            l_eye_height = point_LEB[1] - point_LET[1]

            r_eye_center = [(point_REB[0] + point_RET[0])/2, (point_REL[1] + point_RER[1])/2]
            l_eye_center = [(point_LEB[0] + point_LET[0])/2, (point_LEL[1] + point_LER[1])/2]
            # calculate vertical distance between upper eye lid and eyebrow
            r_lid_eyebrow_dist = point_RET[1] - point_RET[1]
            l_lid_eyebrow_dist = p3_LE[1] - point_LET[1]
            r_semiaxes = (int(abs(point_469[0]-point_471[0])*0.5 * width_fraction), 
                          int(abs(point_472[1]-point_470[1])*0.5 * height_fraction))
            r_a = r_semiaxes[0]
            r_b = r_semiaxes[1] 
            r_xc = int(r_eye_center[0])
            r_yc = int(r_eye_center[1])
            r_x = point_REIC[0]
            r_y = point_REIC[1]
            ##
            l_semiaxes = (int(abs(point_474[0]-point_476[0])*0.5 * width_fraction), 
                          int(abs(point_477[1]-point_475[1])*0.5 * height_fraction))
            l_a = l_semiaxes[0]
            l_b = l_semiaxes[1]
            l_xc = int(l_eye_center[0])
            l_yc = int(l_eye_center[1])
            l_x = point_LEIC[0]
            l_y = point_LEIC[1]
            #checking eye angles
            # RIGHT EYE
            r_pitch = 0
            r_yaw = 0
            if r_y >= r_yc + r_b*down_gaze_fraction or r_lid_eyebrow_dist > 0.4*r_eye_height:
                r_pitch = -1
            if r_y <= r_yc - r_b:
                r_pitch = 1
            if r_x >= r_xc + r_a:
                r_yaw = 1
            if r_x <= r_xc - r_a:
                r_yaw = -1
            cv2.putText(image, f"Right_eye Pitch: {r_pitch:.2f} Yaw: {r_yaw:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 


            # LEFT EYE
            l_pitch = 0
            l_yaw = 0
            if l_y >= l_yc+ l_b*down_gaze_fraction or l_lid_eyebrow_dist > 0.4*l_eye_height:
                l_pitch = -1
            if l_y <= l_yc - l_b:
                l_pitch = 1
            if l_x >= l_xc + l_a:
                l_yaw = 1
            if l_x <= l_xc - l_a:
                l_yaw = -1
            cv2.putText(image, f"Left_eye Pitch: {l_pitch:.2f} Yaw: {l_yaw:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
            
            eyes_distracted = 0
            if abs(r_pitch) >= 1 or abs(l_pitch) >= 1 or abs(r_yaw) >= 1 or abs(l_yaw) >= 1:
                eyes_distracted = 1

            ######calculating EAR
            # Compute EAR for left and right eyes
            left_coord = [point_LER, p2_LE, p3_LE, point_LEL, p5_LE, p6_LE]
            ear_L = ear_cal(left_coord)
            right_coord = [point_RER, p2_RE, p3_RE, point_REL, p5_RE, p6_RE]
            ear_R = ear_cal(right_coord)
            # Average both eyes EAR
            avg_ear = (ear_L + ear_R) / 2
            # perclos logic
            

            # Get the current time for timing drowsiness
            current_time = time.time()

            # Define the threshold based on 80% of the baseline EAR.
            # According to literature, drowsiness is typically inferred when EAR falls
            # below a percentage of a person's normal (baseline) eye openness.
            threshold = 0.8 * baseline_ear

            # If the EAR remains below the threshold, then consider it a sign of drowsiness.
            if avg_ear > threshold:
                if drowsy_start_time is None:
                    drowsy_start_time = current_time  # start the drowsiness timer
                else:
                    drowsiness_timer = current_time - drowsy_start_time  # update timer
            else:
                drowsy_start_time = None  # reset if eye reopens
                drowsiness_timer = 0

            # Display the drowsiness timer in the bottom right of the image.
            # cv2.putText(image, f"Drowsiness: {drowsiness_timer:.2f} sec", 
            #             (250,  20), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # When the drowsiness timer exceeds 10 seconds, display an alarm message.
            if drowsiness_timer >= 10:
                cv2.putText(image, "DROWSINESS ALARM", 
                            (200,225), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            # PERCLOS LOGIC
            ear_history.append((avg_ear, current_time))
            ear_history = [(ear, t) for ear, t in ear_history \
                           if current_time - t <= interval_duration_perclos]
            if (current_time - update_time_perclos) >=1:
                perclos_val = perclos_cal(ear_history, threshold_perclos)
                update_time_perclos = current_time
                # cv2.putText(image, f'PERCLOS: {perclos_val: .2f}' , \
                #             (10,80), cv2.FONT_HERSHEY_SIMPLEX,1,(240,150,200),2)
                
            if perclos_val >= 0.8:
                cv2.putText(image, 'Possible Drowsiness', (200,260), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255),4)
            ##Distraction criteria
            
            

        
            
            #####################
            # 4.4. - Draw the positions on the frame
            l_eye_width = point_LEL[0] - point_LER[0]
            l_eye_height = point_LEB[1] - point_LET[1]
            l_eye_center = [(point_LEL[0] + point_LER[0])/2 ,(point_LEB[1] + point_LET[1])/2]
            
            #cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("Left eye: x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)))
            cv2.putText(image, "Left eye:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 

            r_eye_width = point_REL[0] - point_RER[0]
            r_eye_height = point_REB[1] - point_RET[1]
            r_eye_center = [(point_REL[0] + point_RER[0])/2 ,(point_REB[1] + point_RET[1])/2]
            ##############3
            

            ##################
            #cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)))
            cv2.putText(image, "Right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
            ###########my texts
            cv2.putText(image, "Left_EAR: " + str(np.round(ear_L, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(image, "Right_EAR: " + str(np.round(ear_R, 2)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(image, f'PERCLOS value: {perclos_val: .2f}' , \
                            (10,95), cv2.FONT_HERSHEY_SIMPLEX,0.5,(250,0,250),2)
            # speed reduction (comment out for full speed)
            cv2.putText(image, f'distraction flags: {head_distracted}_{eyes_distracted}' , \
                            (10,180), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            time.sleep(1/25) # [s]

        end = time.time()
        totalTime = end-start

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0
        
        # Detect if driver is distracted
        if head_distracted and eyes_distracted: 
            
            distracted_timeout += totalTime
            if distracted_timeout >= 200: # if the head and eye gaze continued for more than a certain amount of time
                
                cv2.putText(image, f'DISTRACTION ALARM', (200,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        else:
            distracted_timeout = 0
        cv2.putText(image, f'Distracted time: {int(distracted_timeout)}', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        
            
        
        
        #print("FPS:", fps)

        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)       
                    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly source and eventual log file
cap.release()
#log_file.close()
    
# [EOF]