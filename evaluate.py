import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from tensorflow.keras.optimizers import RMSprop
data_dir = "uploads/flowers"
img_height = 224
img_width = 224
batch_size = 64

train_model = tf.keras.models.load_model("models/user.h5")