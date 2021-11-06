from sys import path
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
name_model = "flowers"

class create_model:
  def __init__(self, data_dir, img_height,img_width,model_training,epoch = 5, name_model = "user",batch_size = 64):
    img_height = int(img_height)
    img_width = int(img_width)
    batch_size = int(batch_size)
    epoch = int(epoch)
    self.data_dir = data_dir
    self.img_height = int(img_height)
    self.img_width = int(img_width)
    self.batch_size = int(batch_size)
    self.name_model = name_model
    self.model_training = model_training
    self.epoch = int(epoch)
    print(self)
    self.train_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir,
                    seed=123,
                    validation_split=0.2,
                    image_size=(img_height, img_width),
                    batch_size=batch_size,
                    subset="training")
    self.val_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir,
                    seed=123,
                    subset="validation",
                    validation_split=0.2,
                    image_size=(img_height, img_width),
                    batch_size=batch_size)
    self.val_batches = tf.data.experimental.cardinality(self.val_ds)
    self.num_classes = len(self.train_ds.class_names)
    print(self.num_classes)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    AUTOTUNE = tf.data.AUTOTUNE
    self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
    self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

  #Mobilenet

  def mobileNet(self):
    base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(self.img_height, self.img_width, 3))
    x = base_model.layers[-6].output
    output = Dense(units=self.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    for layer in model.layers[:-23]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=self.epoch,
    )
    path_model = "models/"+ self.name_model + "_mobilenet.h5"
    model.save(path_model)
    # train_model = tf.keras.models.load_model(path_model)
    # loss0, accuracy0 = train_model.evaluate(self.val_ds)
    # print("initial loss: {:.2f}".format(loss0))
    # print("initial accuracy: {:.2f}".format(accuracy0))
    print(H.history)
    traning_result = {
      'log': H.history,
      'model': path_model,
      'name_model': self.name_model + "_mobilenet.h5",
      'type':'mobiletnet'
    }
    return traning_result

  # Resnet50

  def resnet50(self):
    base_model = tf.keras.applications.resnet50.ResNet50()
    x = base_model.layers[-2].output
    output = Dense(units=self.num_classes, activation='softmax')(x)
    base_model.summary()
    model = Model(inputs=base_model.input, outputs=output)

    for layer in model.layers[:-10]:
        layer.trainable = False

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=5,
    )
    model.save("uploads/"+ name_model + "_resnet50.h5")
    train_model = tf.keras.models.load_model("uploads/"+ name_model + "_resnet50.h5")
    loss0, accuracy0 = train_model.evaluate(self.val_ds)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

  # Mobilenetv2

  def mobileNetV2(self):
    base_model = tf.keras.applications.MobileNetV2()
    model = tf.keras.Sequential([base_model,
                    tf.keras.layers.GlobalAveragePooling2D(), 
                    Dense(5, activation='softmax')
                   ])
    model.summary()
    for layer in model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=5,
    )
    model.save("uploads/"+ name_model + "_mobilenet.h5")
    train_model = tf.keras.models.load_model("uploads/"+ name_model + "_mobilenet.h5")
    loss0, accuracy0 = train_model.evaluate(self.val_ds)
    print(" initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    

# train_ds = tf.keras.utils.image_dataset_from_directory(
# data_dir,
# seed=123,
# validation_split=0.2,
# image_size=(img_height, img_width),
# batch_size=batch_size,
# subset="training")

# val_ds = tf.keras.utils.image_dataset_from_directory(
# data_dir,
# seed=123,
# subset="validation",
# validation_split=0.2,
# image_size=(img_height, img_width),
# batch_size=batch_size)


# val_batches = tf.data.experimental.cardinality(val_ds)

# num_classes = len(train_ds.class_names)
# print(num_classes)


# #Configure data

# normalization_layer = tf.keras.layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.


# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# # Aug data

# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#   tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
# ])

# base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height,img_width,3),
#                                                include_top=False,
#                                                weights='imagenet')





# def MobileNet():
#   base_model = tf.keras.applications.mobilenet.MobileNet()
#   x = base_model.layers[-6].output
#   output = Dense(units=num_classes, activation='softmax')(x)
#   model = Model(inputs=base_model.input, outputs=output)
#   model.summary()
#   for layer in model.layers[:-23]:
#       layer.trainable = False

#   model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#   model.fit(train_ds,
#               steps_per_epoch=len(train_ds),
#               validation_data=val_ds,
#               validation_steps=val_batches,
#               epochs=5,
#   )
#   model.save("uploads/"+ name_model + "_mobilenet.h5")
#   train_model = tf.keras.models.load_model("uploads/"+ name_model + "_mobilenet.h5")
#   loss0, accuracy0 = train_model.evaluate(val_ds)
#   print("initial loss: {:.2f}".format(loss0))
#   print("initial accuracy: {:.2f}".format(accuracy0))

# Resnet50()