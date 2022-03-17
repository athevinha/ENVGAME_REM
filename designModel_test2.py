import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras import layers
from keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from flask import Blueprint
designModel = Blueprint('designModel', __name__, template_folder='templates')

class design:
    def __init__(self,w,h,d):
        print('__init__')
        self.model = Sequential()
        self.model.add(tf.keras.Input(shape=(w, h, d)) )
        @designModel.route("/hello")
        def getModelSummary():
            a = self.model.summary()
            print('====',a)
            return 'for fun'

    def addConv2D(self):
         self.model.add(Conv2D(64, 192, kernel_size=5, padding=2));
    
    def addMaxPooling2D(self,pool_size=(2,2)):
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=None, padding='valid', data_format=None))

    def addFlatten(self):
        self.model.add(Flatten(name='flatten'))()

    def addDropout(self, x):
        self.model.add(Dropout(x))
        
    def addDense(self, x, activation = 'relu'):
        self.model.add(Dense(x,activation=activation))

    def ComplieModel(self,opt = RMSprop(0.001) ,loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(opt,loss, metrics)

    def saveModel(self):
        self.model.save('user.h5')

    def trainModel(self):
        self.model.save('user.h5')

