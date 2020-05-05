import cv2
import numpy as np
import tflearn
import dlib
import math
import imutils
import threading
import random
from emotion_detection import facial_emotion_detector, emotion_for_filter, sprite_list


# Draws sprite over a image using alpha chanel to see which pixels need to be replaced
def sprite_over_image(frame, each_sprite, x_out_of_bound, y_out_of_bound):
    (height,width) = (each_sprite.shape[0], each_sprite.shape[1])
    (image_height,image_width) = (frame.shape[0], frame.shape[1])

    # if sprite gets out of image in the bottom
    if y_out_of_bound + height >= image_height: 
        each_sprite = each_sprite[0:image_height-y_out_of_bound,:,:]

    # if sprite gets out of image to the right
    if x_out_of_bound + width >= image_width:
        each_sprite = each_sprite[:,0:image_width-x_out_of_bound,:]

    # if sprite gets out of image to the left
    if x_out_of_bound < 0: 
        each_sprite = each_sprite[:,abs(x_out_of_bound)::,:]
        width = each_sprite.shape[1]
        x_out_of_bound = 0

    # chanel 4 is alpha channel - for backgroud
    # 0 being most transparent to 255 being least transparent
    for ch in range(0,3):
            frame[y_out_of_bound:y_out_of_bound + height, x_out_of_bound:x_out_of_bound + width, ch] =  \
            each_sprite[:,:,ch] * (each_sprite[:,:,3]/255.0) +  \
                frame[y_out_of_bound:y_out_of_bound+height, \
                    x_out_of_bound:x_out_of_bound+width, ch] * \
                        (1.0 - each_sprite[:,:,3]/255.0)
    return frame


# Adjust sprite in accordance to the head's position and width
# If the sprite doesn't fit the screen in the top, the sprite should be trimed
def set_sprite_to_head(each_sprite, head_width, head_y_position, apply_on_upper_half = True):
    (h_sprite,w_sprite) = (each_sprite.shape[0], each_sprite.shape[1])
    factor = 1.0 * head_width/w_sprite
    # resize with head's width
    each_sprite = cv2.resize(each_sprite, (0,0), fx=factor, fy=factor) 
    (h_sprite,w_sprite) = (each_sprite.shape[0], each_sprite.shape[1])

    # position the sprite to end where the head begins
    y_orig =  head_y_position-h_sprite if apply_on_upper_half else head_y_position
    # check for head being too close to the top of the image detected 
    # blocking the sprite from fitting in the screen
    if (y_orig < 0): 
            each_sprite = each_sprite[abs(y_orig)::,:,:]
            # make sprite begin at top of the detected image
            y_orig = 0
    return (each_sprite, y_orig)


# Applies sprite to image detected face's coordinates and adjust it to head
def deploy_sprite(image, path2sprite,w,x,y, angle, apply_on_upper_half = True):
    sprite = cv2.imread(path2sprite,-1)
    sprite = imutils.rotate_bound(sprite, angle)
    (sprite, y_final) = set_sprite_to_head(sprite, w, y, apply_on_upper_half)
    image = sprite_over_image(image,sprite,x, y_final)


# ReturnS angle between co-ordinates
def inclination_angle(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl

# Grabs boundbox's corners
def boundbox_coordinates(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])

    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    return (x,y,w,h)


def get_face_boundbox(points, face_part):
    # for jawline
    if face_part == 'J':
        (x,y,w,h) = boundbox_coordinates(points[6:12])
    # for left eyebrow
    elif face_part == 'LEB':
        (x,y,w,h) = boundbox_coordinates(points[17:22])
    # for right eyebrow
    elif face_part == 'REB':
        (x,y,w,h) = boundbox_coordinates(points[22:27]) 
    # for left eye        
    elif face_part == 'LE':
        (x,y,w,h) = boundbox_coordinates(points[36:42]) 
    # for right eye
    elif face_part == 'RE':
        (x,y,w,h) = boundbox_coordinates(points[42:48]) 
    # for nose
    elif face_part == 'N':
        (x,y,w,h) = boundbox_coordinates(points[29:36]) 
    # for mouth
    elif face_part == 'M':
        (x, y, w, h) = boundbox_coordinates(points[48:68]) 
   
    return (x,y,w,h)


# Main function
def face_filters(event):
    while True:
        video_capture = cv2.VideoCapture(0)
        # set initial values
        (x,y,w,h) = (0,0,10,10)

        # Detects human face in an image
        detector = dlib.get_frontal_face_detector()

        # Dlib facial keypoints (landmarks) detector
        model = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(model)

        # Running thread
        while event.is_set():
            ret, image = video_capture.read()
            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = detector(gray_scale, 0)

            # Setting sprites for each emotion
            if emotion_for_filter == "neutral":
                set_sprite(0)
                set_sprite(1)

            elif emotion_for_filter == "anger":
                set_sprite(1)
                set_sprite(3)

            elif emotion_for_filter == "disgust":
                set_sprite(2)

            elif emotion_for_filter == "fear":
                set_sprite(4)

            elif emotion_for_filter == "happy":
                set_sprite(2)
                set_sprite(3)

            elif emotion_for_filter == "sadness":
                set_sprite(1)
                set_sprite(5)

            elif emotion_for_filter == "surprise":
                set_sprite(6)


            (x, y, w, h) = (face.left(), face.top(),
                            face.width(), face.height())
            # Detecting facial keypoints (landmarks)
            shape = predictor(gray_scale, face)
            shape = imutils.face_utils.shape_to_np(shape)
            # inclination angle for eyebrows
            incl_angle = inclination_angle(shape[17], shape[26]) 

            # check if mouth is open
            jaw_drop = (shape[66][1] - shape[62][1]) >= 10

            # Crown
            if sprite_list[0]:
                deploy_sprite(image, "./sprites/crown.png",w,x,y, incl_angle)

            # Soul Patch
            if sprite_list[1]:
                (x0,y0,w0,h0) = get_face_boundbox(shape, 'J')
                deploy_sprite(image, "./sprites/soul_patch.png",w0,x0,y0, 
                                incl_angle, apply_on_upper_half=False)
            
            # Pig Nose
            if sprite_list[2]:
                (x1,y1,w1,h1) = get_face_boundbox(shape, 'M')
                deploy_sprite(image, "./sprites/pig_nose.png",w1,x1,y1, incl_angle)

            # Glasses - random glass sprites keep on flipping
            if sprite_list[3]:
                (x2,y2,_,h2) = get_face_boundbox(shape, 'LEB')
                deploy_sprite(image, "./sprites/sunglasses_{}.png".format(random.randint(1,5)),
                                w,x,y2, incl_angle, apply_on_upper_half = False)

            # Pill condition - if jaw is dropped, put pill in mouth          
            if sprite_list[4]:                
                if jaw_drop:
                    (x3,y3,w3,h3) = get_face_boundbox(shape, 'M')
                    deploy_sprite(image, "./sprites/pill.png",
                                    w3,x3,y3, incl_angle, apply_on_upper_half = False)

            # Tissues
            if sprite_list[5]:
                (x1,y1,w1,h1) = get_face_boundbox(shape, 'M')
                deploy_sprite(image, "./sprites/tissue.png",w1,x1,y1, incl_angle)

            # Cloud, Rain and Thunder
            if sprite_list[6]:
                deploy_sprite(image, "./sprites/cloudrain.png",w,x,y, incl_angle)


        video_capture.release()
        cv2.imshow("Real-time Facial Expression Detection with automated Face-Filters", cv2.resize(image,None,fx=1,fy=1))
        
        wait_key = cv2.waitKey(20)
        # Break on escape key
        if wait_key == 27:
            break

    video_capture.release()
    cv2.destroyWindow("Real-time Facial Expression Detection with automated Face-Filters")


# Set sprite to be drawn
def set_sprite(index):
    global sprite_list
    sprite_list[index] = (1 - sprite_list[index]) 


# Multi-threading to facilitate multiple filters
event = threading.Event()
event.set()
thread = threading.Thread(target=face_filters, args=(event,))
thread.setDaemon(True)
thread.start()


# Delete window and threads
def delete_window():
        global event, thread
        event.clear()
        tk.destroy()


if __name__ == "__main__":
    facial_emotion_detector()
    face_filters(event)
