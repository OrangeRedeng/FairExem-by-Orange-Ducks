#Import
import numpy as np
import cv2 as cv
import argparse
import time
import math
from openvino.inference_engine import IECore





#IECore
ie = IECore()


#Load the models
parser = argparse.ArgumentParser(description='Run models with OpenVINO')
parser.add_argument('-mFD', dest='face_detection_model', default='face-detection-adas-0001', help='Path to the model_1')
parser.add_argument('-mLR', dest='landmarks_regression_model', default='landmarks-regression-retail-0009', help='Path to the model_2')
parser.add_argument('-mAGR', dest='age_gender_recognition_model', default='age-gender-recognition-retail-0013', help='Path to the model_3')
parser.add_argument('-mER', dest='emotions_recognition_model', default='emotions-recognition-retail-0003', help='Path to the model_4')
parser.add_argument('-mSR', dest='super_resolution_model', default='single-image-super-resolution-1033', help='Path to the model_5')
parser.add_argument('-mPVB', dest='person_vehicle_bike_detection_model', default='person-vehicle-bike-detection-crossroad-1016', help='Path to the model_6')
parser.add_argument('-mHP', dest='head_pose_estimation_model', default='head-pose-estimation-adas-0001', help='Path to the model_7') 
args = parser.parse_args()





#Load&Setup face_detection_model
def face_detection_model_init():
    net_FD = ie.read_network(args.face_detection_model + '.xml', args.face_detection_model + '.bin')
    exec_net_FD = ie.load_network(net_FD, 'CPU')
    return net_FD, exec_net_FD

#Load&Setup landmarks_regression_model
def landmarks_regression_model_init():
    net_LR = ie.read_network(args.landmarks_regression_model + '.xml', args.landmarks_regression_model + '.bin')
    exec_net_LR = ie.load_network(net_LR, 'CPU')
    return net_LR, exec_net_LR
    
#Load&Setup age_gender_recognition_model
def age_gender_recognition_model_init():
    net_AGR = ie.read_network(args.age_gender_recognition_model + '.xml', args.age_gender_recognition_model + '.bin')
    exec_net_AGR = ie.load_network(net_AGR, 'CPU')
    return net_AGR, exec_net_AGR
    
#Load&Setup emotions_recognition_model
def emotions_recognition_model_init():
    net_ER = ie.read_network(args.emotions_recognition_model + '.xml', args.emotions_recognition_model + '.bin')
    exec_net_ER = ie.load_network(net_ER, 'CPU')
    return net_ER, exec_net_ER

#Load&Setup super_resolution_model
def super_resolution_model_init():
    net_SR = ie.read_network(args.super_resolution_model + '.xml', args.super_resolution_model + '.bin')
    return net_SR

#Load&Setup person_vehicle_bike_detection_model
def person_vehicle_bike_detection_model_init():
    net_PVB = ie.read_network(args.person_vehicle_bike_detection_model + '.xml', args.person_vehicle_bike_detection_model + '.bin')
    exec_net_PVB = ie.load_network(net_PVB, 'CPU')
    return net_PVB, exec_net_PVB

#Load&Setup head_pose_model
def head_pose_estimation_model_init():
    net_HP = ie.read_network(args.head_pose_estimation_model + '.xml', args.head_pose_estimation_model + '.bin')
    exec_net_HP = ie.load_network(net_HP, 'CPU')
    return net_HP, exec_net_HP





#person_vehicle_bike_detection_model
def person_vehicle_bike_detection (frame, net_PVB,  exec_net_PVB):

    #Prepare input
    dim = (512, 512)
    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    inp = resized.transpose(2,0,1) # interleaved to planar (HWC -> CHW)

    #Getting data from the network      
    input_name = next(iter(net_PVB.input_info))
    outputs = exec_net_PVB.infer({input_name:inp})

    #Processing data  
    outs = next(iter(outputs.values()))
    outs = outs[0][0]
    j = 0
    ROI_person = frame
    for out in outs:
        coords =[]
        if out[2] == 0.0:   
            break
        if out[2] > 0.6:
            x_min = out[3]
            y_min = out[4]
            x_max = out[5]
            y_max = out[6]
            coords.append([x_min,y_min,x_max,y_max])
            coord = coords[0]
            h=frame.shape[0]
            w=frame.shape[1]
            coord = coord* np.array([w, h, w, h])
            coord = coord.astype(np.int32)
            ROI_person = frame[coord[1]:coord[3], coord[0]:coord[2]]
            if out[1] == 2.0:
                #Drawing a rectangle 
                cv.rectangle(frame, (coord[0],coord[1]), (coord[2],coord[3]), color=(0, 0, 255))
            if out[1] == 1.0:
                #Drawing a rectangle 
                cv.rectangle(frame, (coord[0],coord[1]), (coord[2],coord[3]), color=(0, 255, 0))
            if out[1] == 0.0:
                #Drawing a rectangle 
                cv.rectangle(frame, (coord[0],coord[1]), (coord[2],coord[3]), color=(255, 0, 0))

                    
    return frame, ROI_person





    
#face_detection_model
def face_detection(frame, net_FD, exec_net_FD, age_gender_recognition_model_online, emotions_recognition_model_online, landmarks_regression_model_online):

    emotion_str = ''
    gender_str  = ''
    age = 0

    #Prepare input
    dim = (672, 384)
    resized_img = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    inp = resized_img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

    #Getting data from the network        
    input_name = next(iter(net_FD.input_info))          
    outputs = exec_net_FD.infer({input_name:inp})

    #Processing data    
    outs = next(iter(outputs.values()))
    outs = outs[0][0]
    j = 0
    ROI_face = frame
    for out in outs:
        coords =[]
        conf = out[2]
        if conf == 0.0: 
            break
        if conf > 0.8:
            x_min = out[3]
            y_min = out[4]
            x_max = out[5]
            y_max = out[6]
            coords.append([x_min,y_min,x_max,y_max])
            coord = coords[0]
            h=frame.shape[0]
            w=frame.shape[1]
            coord = coord* np.array([w, h, w, h])
            coord = coord.astype(np.int32)
            ROI_face = frame[coord[1]:coord[3], coord[0]:coord[2]]

            #Drawing a rectangle 
            cv.rectangle(frame,(coord[0],coord[1]), (coord[2],coord[3]), color=(0, 255, 0))

            broken_ROI_face = False

            try:
                resize = cv.resize(ROI_face, (64,64))
            except cv.error as err:
                broken_ROI_face = True
                print('Invalid frame!')


            #Models that require face_detaction_model
                
            if broken_ROI_face == False:

                if head_pose_estimation_model == True:
                    frame = head_pose_estimation(frame, ROI_face, net_HP, exec_net_HP, coord)

                if age_gender_recognition_model_online == True:
                    age, gender_str = age_gender_recognition(frame, ROI_face, net_AGR, exec_net_AGR)

                if emotions_recognition_model_online == True:     
                    emotion_str = emotions_recognition(frame, ROI_face, net_ER, exec_net_ER)

                if landmarks_regression_model_online == True:  
                    frame = landmarks_regression(frame, ROI_face, net_LR, exec_net_LR)


                #Put text on the frame
                j = j + 1
                text_1 = "ID[" + str(j) + "] Emotion: " + emotion_str
                text_2 = "Age: " + str(age)+ " Gender: " + gender_str  
                l_1 = len(text_1)*12 + 2
                l_2 = len(text_2)*12 + 2
                if l_2 > l_1:
                    l = l_2
                else:
                    l = l_1

                if ((coord[1] + coord[3])/2)>((frame.shape[0])/2):
                    #cv.rectangle(frame,(coord[0],coord[1]-52),(coord[0]+l,coord[1]),(0,255,0),0)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame,text_1,(coord[0],coord[1]-32), font, 0.7,(255, 255, 0),2,cv.LINE_AA)
                    cv.putText(frame,text_2,(coord[0],coord[1]-6), font, 0.7,(255, 255, 0),2,cv.LINE_AA)
                else:
                    #cv.rectangle(frame,(coord[0],coord[3]+52),(coord[0]+l,coord[3]),(0,255,0),0)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(frame,text_1,(coord[0],coord[3]+46), font, 0.7,(255, 255, 0),2,cv.LINE_AA)
                    cv.putText(frame,text_2,(coord[0],coord[3]+16), font, 0.7,(255, 255, 0),2,cv.LINE_AA)
                    

    return frame, ROI_face





#landmarks_regression_model
def landmarks_regression(frame, ROI_face, net_LR, exec_net_LR):
        
    # Getting the image size
    ROI_face_height = ROI_face.shape[0]
    ROI_face_width =  ROI_face.shape[1]
    difWH = ROI_face_width/ROI_face_height
    difHW = ROI_face_height/ROI_face_width

    # Prepare input
    dim = (48, 48)
    resized = cv.resize(ROI_face, dim, interpolation = cv.INTER_AREA)  
    inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

    #Getting data from the network    
    input_name = next(iter(net_LR.input_info))          
    outs = exec_net_LR.infer({input_name:inp})

    #Processing data
    out = next(iter(outs.values()))

    # Eye1
    x = int(out[0][0][0][0] * ROI_face.shape[0] *  difWH)
    y = int(out[0][1][0][0] * ROI_face.shape[1] *  difHW)

    cv.circle(ROI_face, (x,y),5,(0, 0, 255),-1)
        
    # Eye2
    x1 = int(out[0][2][0][0] * ROI_face.shape[0] *  difWH)
    y1 = int(out[0][3][0][0] * ROI_face.shape[1] *  difHW)

    cv.circle(ROI_face, (x1,y1),5,(0, 0, 255),-1)

    #Other points
    x2 = int(out[0][4][0][0] * ROI_face.shape[0] *  difWH)
    y2 = int(out[0][5][0][0] * ROI_face.shape[1] *  difHW)
    
    cv.circle(ROI_face, (x2,y2),5,(0, 0, 255),-1)

    x3 = int(out[0][6][0][0] * ROI_face.shape[0] *  difWH)
    y3 = int(out[0][7][0][0] * ROI_face.shape[1] *  difHW)

    cv.circle(ROI_face, (x3,y3),5,(0, 0, 255),-1)

    x4 = int(out[0][8][0][0] * ROI_face.shape[0] *  difWH)
    y4 = int(out[0][9][0][0] * ROI_face.shape[1] *  difHW)

    cv.circle(ROI_face, (x4,y4),5,(0, 0, 255),-1)

    return frame





#age_gender_recognition_model
def age_gender_recognition(frame, ROI_face, net_AGR, exec_net_AGR):

    #Prepare input
    dim = (62, 62)
    resized = cv.resize(ROI_face, dim, interpolation = cv.INTER_AREA) 
    inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

    #Getting data from the network 
    input_name = next(iter(net_AGR.input_info))            
    outs = exec_net_AGR.infer({input_name:inp})

    #Processing data
    age = outs['age_conv3'][0][0][0][0] * 100
    age = round(age)
    gender = np.argmax(outs['prob'])

    if gender == 0:
        gender_str = 'female'
    else:
        gender_str = 'male'

    return age, gender_str





#emotions_recognition_model
def emotions_recognition(frame, ROI_face, net_ER, exec_net_ER):

    #Prepare input
    dim = (64,64)    
    resized = cv.resize(ROI_face, dim, interpolation = cv.INTER_AREA)    
    inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

    #Getting data from the network 
    input_name = next(iter(net_ER.input_info))       
    outs = exec_net_ER.infer({input_name:inp})

    #Processing data
    out = next(iter(outs.values()))
    i = 0
    imax = i
    maxs = out[0][i][0][0]          
    for i in [0,1,2,3]:
        i = i + 1
        if maxs < out[0][i][0][0]:
            maxs = out[0][i][0][0]
            imax = i

    if imax == 0:
        emotion_str = 'neutral'
    if imax == 1:
        emotion_str = 'happy'
    if imax == 2:
        emotion_str = 'sad'
    if imax == 3:
        emotion_str = 'surprise'
    if imax == 4:
        emotion_str = 'anger'
        
    return emotion_str





#super_resolution_model
def super_resolution(frame, net_SR):

    img = frame

    inp_h, inp_w = img.shape[0], img.shape[1]
    out_h, out_w = inp_h * 3, inp_w * 3  # Do not change! This is how model works

    # Workaround for reshaping bug
    c1 = net_SR.layers['79/Cast_11815_const']
    c1.blobs['custom'][4] = inp_h
    c1.blobs['custom'][5] = inp_w

    c2 = net_SR.layers['86/Cast_11811_const']
    c2.blobs['custom'][2] = out_h
    c2.blobs['custom'][3] = out_w

    # Reshape network to specific size
    net_SR.reshape({'0': [1, 3, inp_h, inp_w], '1': [1, 3, out_h, out_w]})
    
    # Load network to device
    exec_net_SR = ie.load_network(net_SR, 'GPU')

    # Prepare input
    inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
    inp = inp.reshape(1, 3, inp_h, inp_w)
    inp = inp.astype(np.float32)

    # Prepare second input - bicubic resize of first input
    resized_img = cv.resize(img, (out_w, out_h), interpolation=cv.INTER_CUBIC)
    resized = resized_img.transpose(2, 0, 1)
    resized = resized.reshape(1, 3, out_h, out_w)
    resized = resized.astype(np.float32)

    
    #Getting data from the network 
    outs = exec_net_SR.infer({'0': inp, '1': resized})

    #Processing data
    out = next(iter(outs.values()))
    out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
    out = np.clip(out * 255, 0, 255)
    out = np.ascontiguousarray(out).astype(np.uint8)

    return img, resized_img, out





#head_pose_estimation_model
def head_pose_estimation(frame, ROI_face, net_HP, exec_net_HP, coord):
    global flag_head_pose
    global norm_pitch
    global norm_yaw
    global norm_down_pitch
    global photo_key
    if flag_head_pose == 0:
        if photo_key == 0:
            print('look at the monitor')
        else: 
            print('look at the notebook')
        timer = time.time()
        new_time = timer
        while new_time - timer < 3:
            new_time = time.time()
        ret, frame = cap.read()
        size = 100
        
        #Prepare input
        dim = (60,60)    
        resized = cv.resize(ROI_face, dim, interpolation = cv.INTER_AREA)    
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        
        #Getting data from the network 
        input_name = next(iter(net_HP.input_info))       
        outs = exec_net_HP.infer({input_name:inp})
        
        #Processing data
        yaw = outs['angle_y_fc'][0][0]
        pitch = outs['angle_p_fc'][0][0]
        roll = outs['angle_r_fc'][0][0]
        
        
        cx = ((coord[2]-coord[0])/2)+coord[0]
        cy = ((coord[3]-coord[1])/2)+coord[1]
        
        
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        
        
        # X-Axis pointing to right. drawn in blue
        x1 = size * (math.cos(yaw) * math.cos(roll)) + cx
        y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + cy
        
        # Y-Axis | drawn in green
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + cx
        y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + cy
        
        # Z-Axis (out of the screen) drawn in red
        x3 = size * (math.sin(yaw)) + cx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + cy
        
        cv.line(frame, (int(cx), int(cy)), (int(x1),int(y1)),(0,0,255),2)
        cv.line(frame, (int(cx), int(cy)), (int(x2),int(y2)),(0,255,0),2)
        cv.line(frame, (int(cx), int(cy)), (int(x3),int(y3)),(255,0,0),2)
        print('would you like take photo again?')
        cv.imshow('frame',frame)
        cv.waitKey(0)
        choise = 0  # я хз как сделать считывание выбора может через cv.waitKeyh()??
        if choise == 0: 
            photo_key += 1
            if photo_key == 1:
                norm_pitch = pitch
                norm_yaw = yaw
            if photo_key == 2:
                norm_down_pitch = pitch
                flag_head_pose = 1
            
        
        return frame
    else:
    
        size = 100
        
        #Prepare input
        dim = (60,60)    
        resized = cv.resize(ROI_face, dim, interpolation = cv.INTER_AREA)    
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        
        #Getting data from the network 
        input_name = next(iter(net_HP.input_info))       
        outs = exec_net_HP.infer({input_name:inp})
        
        #Processing data
        yaw = outs['angle_y_fc'][0][0]
        pitch = outs['angle_p_fc'][0][0]
        roll = outs['angle_r_fc'][0][0]
        
        
        cx = ((coord[2]-coord[0])/2)+coord[0]
        cy = ((coord[3]-coord[1])/2)+coord[1]
        
        
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        
        
        # X-Axis pointing to right. drawn in blue
        x1 = size * (math.cos(yaw) * math.cos(roll)) + cx
        y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + cy
        
        # Y-Axis | drawn in green
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + cx
        y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + cy
        
        # Z-Axis (out of the screen) drawn in red
        x3 = size * (math.sin(yaw)) + cx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + cy
        
        cv.line(frame, (int(cx), int(cy)), (int(x1),int(y1)),(0,0,255),2)
        cv.line(frame, (int(cx), int(cy)), (int(x2),int(y2)),(0,255,0),2)
        cv.line(frame, (int(cx), int(cy)), (int(x3),int(y3)),(255,0,0),2)
        
        
        
        #xyita_2
        global sum_deviation
        global times
        global max_devition_time
        
        global max_deviation_pitch

        global deviation_pitch
        if pitch < norm_pitch - max_deviation_pitch:
            deviation_pitch = time.time()
        if deviation_pitch != 0:
            if pitch >= norm_pitch + max_deviation_pitch:
                devition_time = time.time() - deviation_pitch
                sum_deviation += devition_time
                if devition_time >= max_devition_time:
                    times.append(devition_pitch)
                deviation_pitch = 0
        
        global deviation_down_pitch
        if pitch > norm_down_pitch + max_deviation_pitch:
            deviation_down_pitch = time.time()
        if deviation_down_pitch != 0:
            if pitch <= norm_down_pitch - max_deviation_pitch:
                devition_time = time.time() - deviation_down_pitch
                sum_deviation += devition_time
                if devition_time >= max_devition_time:
                    times.append(devition_down_pitch)
                deviation_down_pitch = 0
        
        global max_deviation_yaw
        
        global deviation_left
        if yaw > norm_yaw + max_deviation_yaw:
            deviation_left = time.time()
        if deviation_left != 0:
            if pitch <= norm_yaw + max_deviation_yaw:
                devition_time = time.time() - deviation_left
                sum_deviation += devition_time
                if devition_time >= max_devition_time:
                    times.append(devition_left)
                deviation_left = 0
        
        global deviation_right
        if yaw < norm_yaw - max_deviation_yaw:
             deviation_right = time.time()
        if deviation_right != 0:
             if yaw >= norm_yaw - max_deviation_yaw:
                 devition_time = time.time() - deviation_right
                 sum_deviation += devition_time
                 if devition_time >= max_devition_time:
                      times.append(devition_right)
                 deviation_right = 0 
             
        return frame





#main
net_PVB, exec_net_PVB = person_vehicle_bike_detection_model_init()
net_FD, exec_net_FD = face_detection_model_init()
net_AGR, exec_net_AGR = age_gender_recognition_model_init()
net_ER, exec_net_ER = emotions_recognition_model_init()
net_LR, exec_net_LR = landmarks_regression_model_init()
net_SR = super_resolution_model_init()
net_HP, exec_net_HP = head_pose_estimation_model_init()


person_vehicle_bike_detection_model_online = False
face_detection_model_online = True
age_gender_recognition_model_online = False
emotions_recognition_model_online = False
landmarks_regression_model_online = False
super_resolution_model_online = False
head_pose_estimation_model = True


#Photo

#frame = cv.imread('er3.jpg')
#if frame is None:
    #raise Exception('Image not found!')


#Webcam

# Reading the video
global start_time
global sum_deviation
sum_deviation = 0
#xyita_1
global norm_pitch
global norm_yaw
global norm_down_pitch
global times
times = []
global max_devition_time
max_devition_time = 10
global max_deviation_pitch
max_deviation_pitch = np.pi/30
global max_deviation_yaw
max_deviation_yaw = np.pi/20
global flag_head_pose

global deviation_pitch
deviation_pitch = 0
global deviation_down_pitch
deviation_down_pitch = 0
global deviation_left
deviation_left = 0
global deviation_rigth
deviation_right = 0
flag_head_pose = 0
global photo_key
photo_key = 0
start_time = time.time()
cap = cv.VideoCapture(0)
while(cap.isOpened()):


    # Reading the frame
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

    #Person_vihicle_bike_model 
    if person_vehicle_bike_detection_model_online == True:
        frame, ROI_person = person_vehicle_bike_detection (frame, net_PVB, exec_net_PVB)

    #Face_detection, age_gender, emotions, landmarks models       
    if (face_detection_model_online == True) or (age_gender_recognition_model_online == True) or (emotions_recognition_model_online == True) or (landmarks_regression_model_online == True):
        frame, ROI_face = face_detection(frame, net_FD, exec_net_FD, age_gender_recognition_model_online, emotions_recognition_model_online, landmarks_regression_model_online)
    cv.imshow('Frame', frame)

    #Superres model
    if super_resolution_model_online == True:
        img, resized_img, out = super_resolution(frame, net_SR)

        cv.imshow('Source image', img)
        cv.imshow('Bicubic interpolation', resized_img)
        cv.imshow('Super resolution', out)


    

cap.release()
cv.destroyAllWindows()    
cv.waitKey()




    








