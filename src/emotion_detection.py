from __future__ import division, absolute_import
import cv2
import numpy as np
import tflearn
import os

FACIAL_EXPRESSIONS = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
HAARCASCADE_PATH = os.path.abspath("") + "/haarcascade_frontalface_default.xml"


emotion_for_filter = "neutral"
sprite_list = [0,0,0,0,0,0,0] 


class NN:
  def __init__(self):
    pass

# Creating the Deep Convolutional Neural Network (DCNN)
  def nn_model(self):
      self.nn = tflearn.layers.core.input_data(shape = [None, 48, 48, 1])
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 64, 5, activation = "relu")
      self.nn = tflearn.layers.conv.max_pool_2d(self.nn, 3, strides = 2)
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 64, 5, activation = "relu")
      self.nn = tflearn.layers.conv.max_pool_2d(self.nn, 3, strides = 2)
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 128, 4, activation = "relu")
      self.nn = tflearn.layers.core.dropout(self.nn, 0.25)
      self.nn = tflearn.layers.core.fully_connected(self.nn, 3072, activation = "relu")
      self.nn = tflearn.layers.core.fully_connected(self.nn, len(FACIAL_EXPRESSIONS), activation = "softmax")
      self.nn = tflearn.layers.estimator.regression(self.nn,optimizer = "momentum",metric = "accuracy",loss = "categorical_crossentropy")
      self.model = tflearn.DNN(self.nn,checkpoint_path = "model",max_checkpoints = 1,tensorboard_verbose = 2)
      self.load_model()
    

# Loading the model
  def load_model(self):
    if not os.path.isfile("model.tfl.meta"):
        return None
    else:
      self.model.load("model.tfl")


# Resizing to predict the photo
  def predict(self, photo):
    if photo is not None:
      photo = photo.reshape([-1, 40, 40, 1])
      return self.model.predict(photo)
    else:
        return None


# Finding face in video and creating Bound-box around face (to later predict the emotion)
def bound_box(photo):
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    cascade_classifier = cv2.CascadeClassifier(HAARCASCADE_PATH)
    photo = cv2.resize(photo, (40,40), interpolation = cv2.INTER_CUBIC) / 255
    return photo


# Creating an instance of Neural Network
nn = NN()
nn.nn_model()
video_capture = cv2.VideoCapture(0)


# Facial emotion detector using OpenCV
def facial_emotion_detector():
    while True:
        # Classifier to draw bounding box around face
        _ , img = video_capture.read()
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        face = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)

        # Calculating the network prediction
        newtork_prediction = nn.predict(bound_box(img))
        if newtork_prediction is not None:
            # Put text different FACIAL_EXPRESSIONS with soft max numbers
            for counter, expression in enumerate(FACIAL_EXPRESSIONS):
                cv2.putText(img, expression, (5, counter * 20 + 20), cv2.cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, "{0:.5f}".format(newtork_prediction[0][counter]), (85, counter * 20 + 20), 
                    cv2.cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # Predict emotion with maximum probability (max value)
            cv2.putText(img,"Predicted emotion is: " + FACIAL_EXPRESSIONS[np.argmax(newtork_prediction[0])],(60,450), 
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1,cv2.LINE_AA) 

            (x,y,w,h) = face[0]
            img = cv2.rectangle(img,(x,y-50),(x+w,y+h+10),(0,255,0),1)
        
        global emotion_for_filter
        emotion_for_filter = FACIAL_EXPRESSIONS[np.argmax(newtork_prediction[0])]
        cv2.imshow("Real-time Facial Expression Detection with automated Face-Filters", cv2.resize(img,None,fx=1,fy=1))
        
        wait_key = cv2.waitKey(20)
        # Break on escape key
        if wait_key == 27:
            break

    video_capture.release()
    cv2.destroyWindow("Real-time Facial Expression Detection with automated Face-Filters")
