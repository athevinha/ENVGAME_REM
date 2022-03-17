from re import S
from sys import path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
# img_height = 224
# img_width = 224
# batch_size = 64
import os

class create_model:
  def __init__(self, data_dir, img_height,img_width,model_training,zip_files,epoch = 5, name_model = "user",batch_size = 64):
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
    self.zip_files = zip_files
    # print(self.zip_files)
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
    self.num_classes = len(self.train_ds.class_names)
    self.train_ds = self.train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=self.num_classes)))
    self.val_ds = self.val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=self.num_classes)))

    self.val_batches = tf.data.experimental.cardinality(self.val_ds)
    print(self.num_classes)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    AUTOTUNE = tf.data.AUTOTUNE


    self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
    self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)


    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.CenterCrop(self.img_height - 10, self.img_width-10 ),
        tf.keras.layers.RandomZoom(.5, .2)
    ])

    # self.train_ds = self.prepare(self.train_ds, shuffle=True, augment=True)
    
  #Mobilenet

  def mobileNet(self,type_model):
    base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(self.img_height, self.img_width, 3),include_top=False,weights='imagenet')
    fcHead = base_model.output
    fcHead = Flatten(name='flatten')(fcHead)
    # fcHead = Dense(256, activation='relu')(fcHead)
    fcHead = Dropout(0.5)(fcHead)
    fcHead = Dense(self.num_classes, activation='softmax')(fcHead)
    model = model = Model(inputs=base_model.input, outputs=fcHead)
    # print(self.num_classes)
    for layer in base_model.layers:
      layer.trainable = False
      
    opt = RMSprop(0.001)
    model.compile(opt, 'categorical_crossentropy', ['accuracy'])
    H = model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=self.epoch,
    )
    # print(H.history['val_accuracy'][-1])
    if(H.history['accuracy'][-1] <= 0.85):
      # print(en(base_model.layers))
      for layer in base_model.layers[-23:]:
          layer.trainable = True

      model.compile(opt, 'categorical_crossentropy', ['accuracy'])

      H = model.fit(self.train_ds,
                  steps_per_epoch=len(self.train_ds),
                  validation_data=self.val_ds,
                  validation_steps=self.val_batches,
                  epochs=self.epoch,
      )
      
    # test_loss, test_acc = model.evaluate(self.test_ds, verbose=2)
    path_model = "models/"+ self.name_model + "_mobilenet.h5"

    model.save(path_model)
    traning_result = {
      "log": H.history,
      "model": path_model,
      "name_model": self.name_model + "_"+ type_model + ".h5",
      "type":type_model,
    }
    with open("historys/" + self.name_model + "_" + type_model +".txt", 'w') as f:
      f.write(str(traning_result))
    return traning_result
    
  # Resnet50

  def resnet50(self):
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=(self.img_height, self.img_width, 3),include_top = False)
    for layer in base_model.layers:
      layer.trainable = False
    fcHead = base_model.output
    fcHead = Flatten(name='flatten')(fcHead)
    fcHead = Dense(256, activation='relu')(fcHead)
    fcHead = Dropout(0.5)(fcHead)
    fcHead = Dense(self.num_classes, activation='softmax')(fcHead)
    model  = Model(inputs=base_model.input, outputs=fcHead)
    # print(self.num_classes)
  
    model.compile(optimizer=RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    H = model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=self.epoch,
    )
    print(H.history['accuracy'][-1])
    if(H.history['accuracy'][-1] <= 0.85):
      print(len(base_model.layers))
      for layer in base_model.layers[-26:]:
          layer.trainable = True

      model.compile(RMSprop(0.001), 'categorical_crossentropy', ['accuracy'])

      H = model.fit(self.train_ds,
                  steps_per_epoch=len(self.train_ds),
                  validation_data=self.val_ds,
                  validation_steps=self.val_batches,
                  epochs=self.epoch,
      )

    path_model = "models/"+ self.name_model + "_resnet50.h5"
    model.save(path_model)
    print(H.history)
    traning_result = {
      "log": H.history,
      "model": path_model,
      "name_model": self.name_model + "_resnet50.h5",
      "type":"resnet50",
    }
    with open("historys/" + self.name_model + "_resnet50.txt", 'w') as f:
      f.write(str(traning_result))
    return traning_result

  # InceptionV3

  def inceptionV3(self):
    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(self.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    H = model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=self.epoch,
    )
    path_model = "models/"+ self.name_model + "_inceptionV3.h5"
    model.save(path_model)
    print(H.history)
    traning_result = {
      "log": H.history,
      "model": path_model,
      "name_model": self.name_model + "_inceptionV3.h5",
      "type":"inceptionV3",
    }
    with open("historys/" + self.name_model + "_inceptionV3.txt", 'w') as f:
      f.write(str(traning_result))
    return traning_result



  def envgame_leaf_disease(self):
    # # model.add(Dense(1024,activation=('relu'),input_dim=512))
    # model.add(Dense(self.num_classes,activation=('softmax')))
    base_model = tf.keras.models.load_model('envgame.h5')
    # input_img = tf.keras.Input(batch_shape=(None,self.img_height,self.img_width,3))
    # ouput_img = base_model(input_img)
    # base_model = Model(input_img, ouput_img)

    fcHead = base_model.layers[-2].output
    fcHead = Flatten(name='flatten')(fcHead)
    # fcHead = Dense(256, activation='relu')(fcHead)
    fcHead = Dense(self.num_classes, activation='softmax')(fcHead)
    model = Model(inputs=base_model.input, outputs=fcHead)
    # base_fake_model = Model(base_model.input, output_final)
    print(self.img_height)
    
    for layer in model.layers: # -23
        layer.trainable = True
       
    model.compile(optimizer=RMSprop(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(self.train_ds,
                steps_per_epoch=len(self.train_ds),
                validation_data=self.val_ds,
                validation_steps=self.val_batches,
                epochs=self.epoch,
    )
    # print(H.history['val_accuracy'][-1])
    if(H.history['accuracy'][-1] <= 0.85):
      # print(en(base_model.layers))
      for layer in base_model.layers[-23:]:
          layer.trainable = True
      model.compile(tf.keras.optimizers.Adam(), 'categorical_crossentropy', ['accuracy'])

      H = model.fit(self.train_ds,
                  steps_per_epoch=len(self.train_ds),
                  validation_data=self.val_ds,
                  validation_steps=self.val_batches,
                  epochs=self.epoch,
      )
    path_model = "models/"+ self.name_model + "_envgame_leaf_disease.h5"
    model.save(path_model)
    traning_result = {
      'log': H.history,
      'model': path_model,
      'name_model': self.name_model + "_envgame_leaf_disease.h5",
      'type':'envgame_leaf_disease',
    }
    with open("historys/" + self.name_model + "_envgame_leaf_disease.txt", 'w') as f:
      f.write(str(traning_result))
    return traning_result




  # def mobileNet(self,type_model):
  #   base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(self.img_height, self.img_width, 3))
  #   base_model.summary()
  #   x = base_model.layers[-6].output
  #   output = Dense(units=self.num_classes, activation='softmax')(x)
  #   model = Model(inputs=base_model.input, outputs=output)
  #   print("============ TRAINING =============")
  #   # model.summary()
  #   for layer in model.layers[:-23]: # -23
  #       layer.trainable = False

  #   model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  #   H = model.fit(self.train_ds,
  #               steps_per_epoch=len(self.train_ds),
  #               validation_data=self.val_ds,
  #               validation_steps=self.val_batches,
  #               epochs=self.epoch,
  #   )
  #   # type_model= "mobilenet"
  #   test_loss, test_acc = model.evaluate(self.test_ds, verbose=2)
  #   path_model = "models/"+ self.name_model + "_mobilenet.h5"
  #   model.save(path_model)
  #   traning_result = {
  #     "log": H.history,
  #     "model": path_model,
  #     "name_model": self.name_model + "_"+ type_model + ".h5",
  #     "type":type_model,
  #   }
  #   with open("historys/" + self.name_model + "_" + type_model +".txt", 'w') as f:
  #     f.write(str(traning_result))
  #   return traning_result
